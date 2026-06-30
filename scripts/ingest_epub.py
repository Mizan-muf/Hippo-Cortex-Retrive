"""
Ingest chapters from an EPUB file into the hippo-cortex knowledgebase.

Usage:
    python ingest_epub.py path/to/book.epub --list
    python ingest_epub.py path/to/book.epub --chapters 1-5
    python ingest_epub.py path/to/book.epub --chapters 1,3,5
    python ingest_epub.py path/to/book.epub --chapters all
    python ingest_epub.py path/to/book.epub --chapters 2-4 --dry-run
    python ingest_epub.py path/to/book.epub --chapters 1-5 --verbose
"""

import argparse
import hashlib
import logging
import re
import sys
import time
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path, PurePosixPath

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)-8s | %(name)-12s | %(message)s",
)
logger = logging.getLogger("ingest_epub")


# ── EPUB parsing ──────────────────────────────────────────────────────────────

def _opf_path(z: zipfile.ZipFile) -> str:
    raw = z.read("META-INF/container.xml").decode("utf-8")
    m = re.search(r'full-path=["\']([^"\']+\.opf)["\']', raw)
    if not m:
        raise ValueError("Cannot locate OPF file in container.xml")
    return m.group(1)


def _parse_opf(z: zipfile.ZipFile, opf_path: str) -> list[dict]:
    """Return spine-ordered list of content items from the OPF manifest."""
    raw = z.read(opf_path)

    def _local(tag: str) -> str:
        """Strip Clark-notation namespace: {uri}name → name"""
        return tag.split("}", 1)[-1] if "}" in tag else tag

    root = ET.fromstring(raw)
    opf_dir = str(PurePosixPath(opf_path).parent)

    manifest: dict[str, str] = {}
    for el in root.iter():
        if _local(el.tag) == "item":
            item_id = el.get("id", "")
            href = el.get("href", "")
            media = el.get("media-type", "")
            if media in ("application/xhtml+xml", "text/html"):
                manifest[item_id] = href

    items = []
    for el in root.iter():
        if _local(el.tag) == "itemref":
            idref = el.get("idref", "")
            if idref not in manifest:
                continue
            href = manifest[idref]
            abs_path = f"{opf_dir}/{href}" if opf_dir and opf_dir != "." else href
            abs_path = abs_path.replace("\\", "/")
            items.append({"id": idref, "href": href, "abs_path": abs_path})

    return items


def _extract_text(z: zipfile.ZipFile, abs_path: str) -> tuple[str, str]:
    """Return (title, plain_text) from an XHTML file inside the epub ZIP."""
    try:
        raw = z.read(abs_path).decode("utf-8", errors="replace")
    except KeyError:
        names = {n.lower(): n for n in z.namelist()}
        actual = names.get(abs_path.lower())
        if not actual:
            return Path(abs_path).stem, ""
        raw = z.read(actual).decode("utf-8", errors="replace")

    # Title: prefer <title>, then first heading
    title = ""
    m = re.search(r"<title[^>]*>(.*?)</title>", raw, re.I | re.S)
    if m:
        title = re.sub(r"<[^>]+>", "", m.group(1)).strip()
    if not title:
        m = re.search(r"<h[1-3][^>]*>(.*?)</h[1-3]>", raw, re.I | re.S)
        if m:
            title = re.sub(r"<[^>]+>", "", m.group(1)).strip()
    if not title:
        title = Path(abs_path).stem

    # Strip markup and normalize whitespace
    text = re.sub(r"<style[^>]*>.*?</style>", " ", raw, flags=re.S | re.I)
    text = re.sub(r"<script[^>]*>.*?</script>", " ", text, flags=re.S | re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&nbsp;", " ", text)
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"&lt;", "<", text)
    text = re.sub(r"&gt;", ">", text)
    text = re.sub(r"&#?\w+;", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    return title, text


def _stable_doc_id(epub_name: str, chapter_idx: int) -> int:
    """Return a stable, positive INT64-safe doc_id unique per (epub, chapter).

    Uses SHA-256 of '<epub_filename>:<chapter_index>' so the same chapter
    always gets the same ID regardless of ingest order, and two different
    books never share a doc_id even if they have the same number of chapters.
    """
    key = f"{epub_name}:{chapter_idx}".encode()
    return int(hashlib.sha256(key).hexdigest()[:14], 16) % (2**31 - 1)


# ── Chapter selection ──────────────────────────────────────────────────────────

def parse_chapter_selection(spec: str, total: int) -> list[int]:
    """
    Parse a human-readable chapter spec into sorted 0-based indices.
      '1-5'    -> [0,1,2,3,4]
      '1,3,5'  -> [0,2,4]
      '3'      -> [2]
      'all'    -> [0..total-1]
    """
    spec = spec.strip().lower()
    if spec == "all":
        return list(range(total))
    indices: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            indices.update(range(int(a) - 1, int(b)))
        else:
            indices.add(int(part) - 1)
    return sorted(i for i in indices if 0 <= i < total)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest EPUB chapters into hippo-cortex")
    parser.add_argument("epub", type=str, help="Path to .epub file")
    parser.add_argument("--list", action="store_true",
                        help="List all spine items with word counts and exit")
    parser.add_argument("--chapters", type=str, default=None,
                        help="Chapters to ingest: '1-5', '1,3,5', or 'all'")
    parser.add_argument("--dry-run", action="store_true",
                        help="Parse and show what would be ingested, skip writes")
    parser.add_argument("--verbose", action="store_true",
                        help="Print 150-char text preview per chapter")
    parser.add_argument("--min-words", type=int, default=100,
                        help="Skip chapters shorter than N words (default: 100)")
    args = parser.parse_args()

    epub_path = Path(args.epub)
    if not epub_path.exists():
        print(f"ERROR: File not found: {epub_path}")
        sys.exit(1)

    with zipfile.ZipFile(epub_path) as z:
        opf = _opf_path(z)
        all_items = _parse_opf(z, opf)

    total = len(all_items)
    print(f"EPUB : {epub_path.name}")
    print(f"Items: {total} spine items found\n")

    # --list or no --chapters: always print the chapter table
    if args.list or not args.chapters:
        print(f"{'#':<5} {'Words':<7} Title")
        print("-" * 72)
        with zipfile.ZipFile(epub_path) as z:
            for i, item in enumerate(all_items):
                title, text = _extract_text(z, item["abs_path"])
                wc = len(text.split())
                note = "  [too short]" if wc < args.min_words else ""
                print(f"{i + 1:<5} {wc:<7} {title[:55]}{note}")
        if not args.chapters:
            print("\nUse --chapters to select which to ingest.  Examples:")
            print("  --chapters 1-5       chapters 1 through 5")
            print("  --chapters 1,3,5     specific chapters")
            print("  --chapters all       everything")
            return

    selected = parse_chapter_selection(args.chapters, total)
    if not selected:
        print("ERROR: No valid chapter indices in selection.")
        sys.exit(1)

    print(f"Selected chapters (1-based): {[i + 1 for i in selected]}\n")

    # Dry run: extract + preview only
    if args.dry_run:
        print("DRY RUN -- no writes\n")
        with zipfile.ZipFile(epub_path) as z:
            for idx in selected:
                title, text = _extract_text(z, all_items[idx]["abs_path"])
                wc = len(text.split())
                skip = "  [SKIP -- too short]" if wc < args.min_words else ""
                print(f"  Chapter {idx + 1}: {title!r}  ({wc} words){skip}")
                if args.verbose and text:
                    print(f"    {text[:150]}...\n")
        return

    # Real ingest
    from src.storage.graph import init_graph, upsert_document
    from src.storage.memory import init_memory
    from src.ingestion.pipeline import ingest

    print("Initializing memory layer (Mem0 + Qdrant)...")
    if not init_memory():
        print("ERROR: Memory init failed. Check Qdrant and Ollama are running.")
        sys.exit(1)

    print("Initializing graph layer (Kuzu)...")
    if not init_graph():
        print("ERROR: Graph init failed.")
        sys.exit(1)

    stats = {"ok": 0, "failed": 0, "skipped": 0, "events": 0, "props": 0}
    t_start = time.time()
    print(f"\nIngesting {len(selected)} chapters...\n")

    with zipfile.ZipFile(epub_path) as z:
        for pos, idx in enumerate(selected, 1):
            title, text = _extract_text(z, all_items[idx]["abs_path"])
            wc = len(text.split())

            if wc < args.min_words:
                logger.info("[%d/%d] Chapter %d %r -- %d words, skipping",
                            pos, len(selected), idx + 1, title, wc)
                stats["skipped"] += 1
                continue

            # Stable ID: unique per (epub filename, chapter index) across all books
            doc_id = _stable_doc_id(epub_path.name, idx)
            chapter_title = f"{epub_path.stem} - Ch{idx + 1}: {title}"

            logger.info("[%d/%d] Chapter %d: %r  (%d words)  doc_id=%d",
                        pos, len(selected), idx + 1, title, wc, doc_id)
            if args.verbose:
                print(f"  Preview: {text[:150]}...")

            # Register the chapter as a Document node in the graph
            try:
                upsert_document(doc_id, chapter_title, source=str(epub_path))
            except Exception as exc:
                logger.warning("  -> Document node failed (non-fatal): %s", exc)

            try:
                result = ingest(text, doc_id=doc_id, title=chapter_title)
                stats["ok"] += 1
                stats["events"] += result["events"]
                stats["props"] += result["properties"]
                logger.info(
                    "  -> segments=%d events=%d props=%d ambiguous=%d%s",
                    result["segments"], result["events"], result["properties"],
                    result["ambiguous_entities"],
                    " [CONFLICT]" if result.get("conflict") else "",
                )
            except Exception as exc:
                stats["failed"] += 1
                logger.error("  -> FAILED chapter %d: %s", idx + 1, exc, exc_info=True)

    elapsed = time.time() - t_start
    print(f"""
=== EPUB ingest complete ===
  File               : {epub_path.name}
  Chapters selected  : {len(selected)}
  Ingested OK        : {stats['ok']}
  Skipped (short)    : {stats['skipped']}
  Failed             : {stats['failed']}
  Total events       : {stats['events']}
  Total properties   : {stats['props']}
  Elapsed            : {elapsed:.1f}s
""")


if __name__ == "__main__":
    main()
