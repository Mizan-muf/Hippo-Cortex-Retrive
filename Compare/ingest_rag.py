#!/usr/bin/env python3
"""
Compare/ingest_rag.py — Build the tsg_rag Qdrant collection for Standard RAG.

Reads EPUB chapters (samples/epub_extracted/OEBPS/Text/000N_Chapter__N.xhtml),
strips HTML, splits into overlapping text chunks, embeds via nomic-embed-text,
and upserts into the tsg_rag Qdrant collection.

Run from project root:
    python Compare/ingest_rag.py               # chapters 1-25, default settings
    python Compare/ingest_rag.py --chapters 10 # first 10 chapters only
    python Compare/ingest_rag.py --chunk-size 500 --overlap 100
    python Compare/ingest_rag.py --reset       # drop collection and rebuild
"""

import argparse
import re
import sys
import uuid
from pathlib import Path

# Allow imports from src/ when run from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from src.config import QDRANT_HOST, QDRANT_PORT, QDRANT_VECTOR_SIZE
from src.core.embed import embed

EPUB_TEXT_DIR = Path("samples/epub_extracted/OEBPS/Text")
RAG_COLLECTION = "tsg_rag"
BATCH_SIZE = 64


# ─── Text helpers ─────────────────────────────────────────────────────────────

def strip_html(html: str) -> str:
    text = re.sub(r"<[^>]+>", " ", html)
    text = (
        text.replace("&lt;", "<").replace("&gt;", ">")
            .replace("&amp;", "&").replace("&quot;", '"')
            .replace("&#39;", "'").replace("&nbsp;", " ")
            .replace("&mdash;", "—").replace("&ndash;", "–")
    )
    return re.sub(r"\s+", " ", text).strip()


def chunk_text(text: str, size: int = 400, overlap: int = 80) -> list[str]:
    """Sliding-window chunker that avoids splitting mid-word."""
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        # Walk back to nearest word boundary
        if end < len(text):
            boundary = end
            while boundary > start and text[boundary] not in " \n\t":
                boundary -= 1
            if boundary > start:
                end = boundary
        chunk = text[start:end].strip()
        if len(chunk) >= 40:       # skip tiny trailing fragments
            chunks.append(chunk)
        next_start = end - overlap
        if next_start <= start:    # safety: always advance
            next_start = start + 1
        start = next_start
        if end == len(text):
            break
    return chunks


def parse_chapter(path: Path) -> str:
    """Return clean text from one XHTML chapter file."""
    raw = path.read_text(encoding="utf-8", errors="replace")
    # Extract only <body> content, then strip all tags
    body_match = re.search(r"<body[^>]*>(.*?)</body>", raw, re.DOTALL | re.IGNORECASE)
    body = body_match.group(1) if body_match else raw
    return strip_html(body)


def chapter_files(n_chapters: int) -> list[tuple[int, Path]]:
    """Return (chapter_idx, path) pairs for chapters 1..n_chapters."""
    files = []
    for i in range(1, n_chapters + 1):
        pattern = f"{i:04d}_Chapter__{i}.xhtml"
        path = EPUB_TEXT_DIR / pattern
        if path.exists():
            files.append((i - 1, path))   # 0-indexed chapter id
        else:
            # Try alternate filenames (single underscore, etc.)
            alt = list(EPUB_TEXT_DIR.glob(f"{i:04d}_*.xhtml"))
            if alt:
                files.append((i - 1, alt[0]))
            else:
                print(f"  [WARN] chapter {i} not found — skipping")
    return files


# ─── Qdrant helpers ───────────────────────────────────────────────────────────

def ensure_collection(qc: QdrantClient, reset: bool) -> None:
    exists = qc.collection_exists(RAG_COLLECTION)
    if exists and reset:
        qc.delete_collection(RAG_COLLECTION)
        print(f"  Dropped existing '{RAG_COLLECTION}' collection")
        exists = False
    if not exists:
        qc.create_collection(
            collection_name=RAG_COLLECTION,
            vectors_config=VectorParams(size=QDRANT_VECTOR_SIZE, distance=Distance.COSINE),
        )
        print(f"  Created '{RAG_COLLECTION}' ({QDRANT_VECTOR_SIZE}d cosine)")


def upsert_batch(qc: QdrantClient, points: list[PointStruct]) -> None:
    qc.upsert(collection_name=RAG_COLLECTION, points=points)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest EPUB chapters into tsg_rag")
    parser.add_argument("--chapters", type=int, default=25, help="Number of chapters to ingest (default: 25)")
    parser.add_argument("--chunk-size", type=int, default=400, dest="chunk_size", help="Max chars per chunk (default: 400)")
    parser.add_argument("--overlap", type=int, default=80, help="Overlap chars between chunks (default: 80)")
    parser.add_argument("--reset", action="store_true", help="Drop and rebuild the collection")
    args = parser.parse_args()

    qc = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=30)

    print(f"tsg_rag ingestion")
    print(f"  chapters  : 1–{args.chapters}")
    print(f"  chunk_size: {args.chunk_size}  overlap: {args.overlap}")
    print(f"  collection: {RAG_COLLECTION}")
    print()

    ensure_collection(qc, args.reset)

    files = chapter_files(args.chapters)
    if not files:
        print("ERROR: no chapter files found in", EPUB_TEXT_DIR)
        sys.exit(1)

    total_chunks = 0
    batch: list[PointStruct] = []

    for chapter_idx, path in files:
        text = parse_chapter(path)
        chunks = chunk_text(text, size=args.chunk_size, overlap=args.overlap)
        print(f"  ch.{chapter_idx+1:02d}  {len(text):>6} chars  ->  {len(chunks):>3} chunks  ({path.name})")

        for chunk_idx, chunk in enumerate(chunks):
            vec = embed(chunk)
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"rag:{chapter_idx}:{chunk_idx}:{chunk[:40]}"))
            batch.append(PointStruct(
                id=point_id,
                vector=vec,
                payload={
                    "text": chunk,
                    "chapter": chapter_idx,
                    "chunk_idx": chunk_idx,
                    "source": path.name,
                },
            ))
            if len(batch) >= BATCH_SIZE:
                upsert_batch(qc, batch)
                batch.clear()

        total_chunks += len(chunks)

    if batch:
        upsert_batch(qc, batch)

    info = qc.get_collection(RAG_COLLECTION)
    print()
    print(f"Done — {total_chunks} chunks ingested")
    print(f"Collection '{RAG_COLLECTION}': {info.points_count} total points")


if __name__ == "__main__":
    main()
