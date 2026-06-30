import logging
import re
from dataclasses import dataclass, field

from rapidfuzz import fuzz

from src.config import LINK_AMBIGUOUS_MIN, LINK_THRESHOLD
from src.core.llm import call as llm_call
from src.storage.graph import find_entity_by_alias, find_entity_by_name, get_all_entity_names
from src.storage.memory import search_knowledge

logger = logging.getLogger(__name__)

# In-memory HITL queue — flush with flush_ambiguous_queue()
_hitl_queue: list[dict] = []


@dataclass
class ResolutionResult:
    input_name: str
    node_id: str | None       # None = new entity (stub will be created by resolver)
    matched_by: str           # "exact" | "alias" | "semantic" | "llm" | "new"
    confidence: float
    is_ambiguous: bool = False


def _normalize_name(name: str) -> str:
    """Strip possessives, trailing punctuation, and encoding artifacts."""
    # Handle curly-apostrophe and straight-apostrophe possessives: "Chen Feng's" → "Chen Feng"
    name = re.sub(r"[’'‘]s$", "", name).strip()
    # Strip trailing punctuation that slips through NER (e.g. "Qilin Fist!")
    name = name.rstrip("!?,.")
    return name.strip()


_FUZZY_EDIT_THRESHOLD = 88   # rapidfuzz score 0-100; catches single-char edits
_FUZZY_TOKEN_MIN_TOKENS = 1  # minimum input tokens required for token-subset match


def _fuzzy_match_name(name: str) -> tuple[str, str] | None:
    """Pass 2.5: fuzzy match against all known non-stub entities.

    Strategy A — edit distance (rapidfuzz token_sort_ratio):
      catches typos and OCR errors, e.g. "Cheng Feng" → "Chen Feng".
    Strategy B — token subset:
      catches partial names, e.g. "Feng" → "Chen Feng".
      Only fires when exactly one entity contains all input tokens (no ambiguity).
    Returns (entity_id, entity_name) or None.
    """
    all_entities = get_all_entity_names()
    if not all_entities:
        return None

    name_lower = name.lower()
    name_tokens = set(name_lower.split())

    # Strategy A: edit distance
    best_score = 0
    best_edit: tuple[str, str] | None = None
    for eid, ename in all_entities:
        score = fuzz.token_sort_ratio(name_lower, ename.lower())
        if score >= _FUZZY_EDIT_THRESHOLD and score > best_score:
            best_score = score
            best_edit = (eid, ename)
    if best_edit:
        return best_edit

    # Strategy B: token subset (only when unambiguous)
    if len(name_tokens) >= _FUZZY_TOKEN_MIN_TOKENS:
        candidates = [
            (eid, ename) for eid, ename in all_entities
            if name_tokens.issubset(set(ename.lower().split()))
        ]
        if len(candidates) == 1:
            return candidates[0]

    return None


def resolve_name(name: str, context: str = "") -> ResolutionResult:
    """Four-pass resolution chain for a single entity name."""
    name = _normalize_name(name)
    # Pass 1 — exact name match in Kuzu
    entity = find_entity_by_name(name)
    if entity:
        logger.debug("entity_linker | pass1 exact: %r → %s", name, entity["id"])
        return ResolutionResult(name, entity["id"], "exact", 1.0)

    # Pass 2 — alias match in Kuzu
    entity = find_entity_by_alias(name)
    if entity:
        logger.debug("entity_linker | pass2 alias: %r → %s", name, entity["id"])
        return ResolutionResult(name, entity["id"], "alias", 0.95)

    # Pass 2.5 — fuzzy edit distance + token subset against known entities
    fuzzy = _fuzzy_match_name(name)
    if fuzzy:
        fid, fname = fuzzy
        entity = find_entity_by_name(fname)
        if entity:
            logger.info("entity_linker | pass2.5 fuzzy: %r → %r", name, fname)
            return ResolutionResult(name, entity["id"], "fuzzy", 0.87)

    # Pass 3 — semantic similarity via Mem0
    results = search_knowledge(name, top_k=3)
    best_score = 0.0
    best_text = ""
    for r in results:
        score = float(r.get("score", 0.0))
        if score > best_score:
            best_score = score
            best_text = r.get("memory", "")

    if best_score >= LINK_THRESHOLD and best_text:
        candidate_name = _extract_entity_name(best_text)
        if candidate_name:
            entity = find_entity_by_name(candidate_name)
            if entity:
                logger.info(
                    "entity_linker | pass3 semantic: %r → %s (score %.3f)",
                    name, entity["id"], best_score,
                )
                return ResolutionResult(name, entity["id"], "semantic", best_score)

    if LINK_AMBIGUOUS_MIN <= best_score < LINK_THRESHOLD and best_text:
        candidate_name = _extract_entity_name(best_text)
        logger.info(
            "entity_linker | ambiguous: %r ~ %r (score %.3f) → HITL queue",
            name, candidate_name, best_score,
        )
        _hitl_queue.append({
            "input_name": name,
            "candidate": candidate_name,
            "score": best_score,
            "context": context[:300],
        })
        return ResolutionResult(name, None, "new", best_score, is_ambiguous=True)

    # Pass 4 — LLM co-reference (only when we have candidates to compare)
    if results and context:
        candidates = [_extract_entity_name(r.get("memory", "")) for r in results]
        candidates = [c for c in candidates if c]
        if candidates:
            resolved = _llm_coreference(name, context, candidates)
            if resolved:
                entity = find_entity_by_name(resolved)
                if entity:
                    logger.info(
                        "entity_linker | pass4 llm: %r → %s", name, entity["id"]
                    )
                    return ResolutionResult(name, entity["id"], "llm", 0.8)

    logger.debug("entity_linker | new stub: %r", name)
    return ResolutionResult(name, None, "new", 0.0)


def _extract_entity_name(memory_text: str) -> str:
    """Pull entity name from stored format 'Entity: {name} | Type: ...'"""
    if memory_text.startswith("Entity: "):
        parts = memory_text[8:].split(" | ")
        return parts[0].strip()
    return ""


def _llm_coreference(input_name: str, context: str, candidates: list[str]) -> str | None:
    candidate_list = "\n".join(f"  - {c}" for c in candidates[:5])
    prompt = (
        f'Given this context:\n  "{context[:400]}"\n\n'
        f'Is "{input_name}" the same entity as any of these known entities?\n'
        f"{candidate_list}\n\n"
        "Reply with the EXACT name of the matching entity, or NULL if none match. "
        "Reply with only the name or NULL — no explanation."
    )
    try:
        answer = llm_call(prompt, temperature=0.0).strip()
        if answer.upper() == "NULL" or not answer:
            return None
        # Validate the LLM returned one of our candidates
        for c in candidates:
            if c.lower() == answer.lower():
                return c
        return None
    except Exception as exc:
        logger.warning("entity_linker | llm coreference failed: %s", exc)
        return None


def resolve_event(event: dict) -> dict:
    """Resolve all participant and location names in an event to node IDs.

    Adds 'participant_ids' and 'location_ids' to the event dict.
    Ambiguous matches are queued in the HITL queue and treated as new stubs.
    Keys in the output dicts are the normalized names (same normalization as
    resolve_name applies internally) so that resolver always writes clean names.
    """
    context = event.get("source_text", "")

    raw_participant_types: dict[str, str] = event.get("participant_types", {})

    participant_ids: dict[str, str | None] = {}
    normalized_participant_types: dict[str, str] = {}
    for name in event.get("participants", []):
        r = resolve_name(name, context)
        participant_ids[r.input_name] = r.node_id
        # Preserve NER label under the normalized name
        ner_label = raw_participant_types.get(name, raw_participant_types.get(r.input_name, "PERSON"))
        normalized_participant_types[r.input_name] = ner_label

    # Replace raw lists/dicts with normalized versions so all downstream code is consistent
    event["participants"] = list(participant_ids.keys())
    event["participant_types"] = normalized_participant_types

    location_ids: dict[str, str | None] = {}
    for name in event.get("locations", []):
        r = resolve_name(name, context)
        location_ids[r.input_name] = r.node_id

    event["participant_ids"] = participant_ids
    event["location_ids"] = location_ids
    return event


def flush_ambiguous_queue() -> list[dict]:
    """Return and clear all pending ambiguous HITL items."""
    items = list(_hitl_queue)
    _hitl_queue.clear()
    return items
