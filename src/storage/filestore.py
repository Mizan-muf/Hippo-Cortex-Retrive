import logging
import re
from datetime import datetime, timezone
from pathlib import Path

from src.config import FILE_STORE_PATH

logger = logging.getLogger(__name__)

SENTINEL = "<!-- USER NOTES — system will never modify below this line -->"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ensure_type_dir(entity_type: str) -> Path:
    d = FILE_STORE_PATH / entity_type
    d.mkdir(parents=True, exist_ok=True)
    return d


def _slugify(name: str) -> str:
    return re.sub(r"[^\w\s-]", "", name).strip().replace(" ", "_")


def _find_record(entity_name: str, entity_type: str) -> Path | None:
    """Return the Path for an existing entity file, or None if not found.
    Checks filename match first, then aliases in frontmatter."""
    type_dir = FILE_STORE_PATH / entity_type
    if not type_dir.exists():
        return None

    normalized = entity_name.lower().strip()
    for f in type_dir.glob("*.md"):
        if f.stem.replace("_", " ").lower() == normalized:
            return f
        content = f.read_text(encoding="utf-8")
        fm_match = re.search(r"aliases:\s*\[([^\]]*)\]", content)
        if fm_match:
            aliases = [a.strip().strip('"\'').lower() for a in fm_match.group(1).split(",") if a.strip()]
            if normalized in aliases:
                return f
    return None


def _parse_record(content: str) -> tuple[str, str, str]:
    """Split a record file into (frontmatter_text, body, user_section)."""
    fm_match = re.match(r"^---\n(.*?)\n---\n", content, re.DOTALL)
    fm_text = fm_match.group(1) if fm_match else ""
    rest = content[fm_match.end():] if fm_match else content

    if SENTINEL in rest:
        idx = rest.index(SENTINEL)
        body = rest[:idx]
        user_section = rest[idx:]
    else:
        # Sentinel missing — inject it rather than aborting on first-ever parse
        body = rest
        user_section = f"{SENTINEL}\n\n## User Notes\n_Add notes here._\n"

    return fm_text, body, user_section


def _serialize_record(fm_text: str, body: str, user_section: str) -> str:
    return f"---\n{fm_text}\n---\n{body}{user_section}"


def _new_record_content(name: str, entity_type: str, fact: str, now: str) -> str:
    fm_text = f"name: {name}\ntype: {entity_type}\naliases: []\ncreated: {now}"
    body = f"\n## Profile\n### Key Facts\n- {fact}\n\n## Event Log\n\n"
    user_section = f"{SENTINEL}\n\n## User Notes\n_Add notes here._\n"
    return _serialize_record(fm_text, body, user_section)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def write_entity(name: str, entity_type: str, fact: str) -> Path:
    """Write a fact about an entity. Creates the file if new, appends if existing."""
    _ensure_type_dir(entity_type)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    existing = _find_record(name, entity_type)

    if existing:
        content = existing.read_text(encoding="utf-8")
        if SENTINEL not in content:
            logger.critical(
                "filestore | sentinel missing in '%s' — write aborted to protect user content",
                existing.name,
            )
            return existing

        fm_text, body, user_section = _parse_record(content)
        entry = f"### {now}\n- {fact}\n\n"
        if "## Event Log" in body:
            body += entry
        else:
            body += f"\n## Event Log\n\n{entry}"

        existing.write_text(_serialize_record(fm_text, body, user_section), encoding="utf-8")
        logger.info("filestore | updated '%s'", existing.name)
        return existing

    slug = _slugify(name)
    path = FILE_STORE_PATH / entity_type / f"{slug}.md"
    path.write_text(_new_record_content(name, entity_type, fact, now), encoding="utf-8")
    logger.info("filestore | created '%s'", path.name)
    return path


def grep_filestore(query: str) -> str:
    """Keyword grep across all entity files. Returns concatenated matching snippets.

    Keywords shorter than 4 characters are ignored (see F8 in README).
    """
    if not FILE_STORE_PATH.exists():
        return ""

    keywords = re.findall(r"[a-z]{4,}", query.lower())
    if not keywords:
        return ""

    hits: list[str] = []
    for md_file in FILE_STORE_PATH.rglob("*.md"):
        raw = md_file.read_text(encoding="utf-8")
        lower = raw.lower()
        if all(kw in lower for kw in keywords):
            relevant = [line for line in raw.splitlines() if any(kw in line.lower() for kw in keywords)]
            if relevant:
                hits.append(f"[{md_file.stem}]\n" + "\n".join(relevant[:10]))

    return "\n\n".join(hits)
