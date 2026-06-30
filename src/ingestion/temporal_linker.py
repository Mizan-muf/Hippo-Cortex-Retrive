import logging
import uuid

from src.storage.graph import get_events_for_entities

logger = logging.getLogger(__name__)


def assign_temporal_order(events: list[dict]) -> list[dict]:
    """Stage 4a: assign temporal_order and pre-allocate IDs by document position."""
    for i, event in enumerate(events):
        seg = int(event.get("segment_id", 0))
        event["temporal_order"] = seg * 1000 + i
        # Pre-assign a stable ID so later stages (4b, storage) can reference it
        if not event.get("id"):
            title = event.get("title", "unknown")
            slug = "".join(c if c.isalnum() else "_" for c in title.lower())[:20]
            event["id"] = f"evt_{slug}_{uuid.uuid4().hex[:8]}"
    return events


def link_within_document(events: list[dict]) -> list[tuple[str, str]]:
    """Generate PRECEDES edges between consecutive events that share a participant.

    For each participant, track the last event they appeared in and wire
    prev → current. This creates the narrative sequence chain inside a document.
    """
    edges: list[tuple[str, str]] = []
    last_event_for_participant: dict[str, str] = {}  # participant_name → event_id

    for event in events:
        event_id = event.get("id")
        if not event_id:
            continue
        participants = event.get("participants", [])
        for name in participants:
            prev_id = last_event_for_participant.get(name)
            if prev_id and prev_id != event_id:
                edges.append((prev_id, event_id))
                logger.debug(
                    "temporal_linker | intra-doc PRECEDES via %r: %s → %s",
                    name, prev_id[:20], event_id[:20],
                )
            last_event_for_participant[name] = event_id

    # Deduplicate (multiple shared participants can produce duplicate pairs)
    seen: set[tuple[str, str]] = set()
    unique: list[tuple[str, str]] = []
    for edge in edges:
        if edge not in seen:
            seen.add(edge)
            unique.append(edge)

    logger.info("temporal_linker | %d intra-doc PRECEDES edges", len(unique))
    return unique


def link_cross_document(
    events: list[dict],
    entity_id_map: dict[str, str],
) -> list[tuple[str, str]]:
    """Stage 4b: find PRECEDES edges from existing graph events to new events.

    Runs after entity linking so entity_id_map contains resolved node IDs.
    Returns list of (existing_event_id, new_event_id) tuples for PRECEDES edges.
    """
    edges: list[tuple[str, str]] = []

    for event in events:
        new_id = event.get("id")
        if not new_id:
            continue

        participant_ids = [
            entity_id_map[name]
            for name in event.get("participants", [])
            if name in entity_id_map and entity_id_map[name]
        ]
        location_ids = [
            entity_id_map[name]
            for name in event.get("locations", [])
            if name in entity_id_map and entity_id_map[name]
        ]
        all_ids = list(set(participant_ids + location_ids))
        if not all_ids:
            continue

        current_order = int(event.get("temporal_order", 0))
        existing = get_events_for_entities(all_ids)
        for ex in reversed(existing):
            if ex["id"] == new_id:
                continue
            if int(ex.get("temporal_order", 0)) < current_order:
                edges.append((ex["id"], new_id))
                logger.info(
                    "temporal_linker | cross-doc PRECEDES: %s → %s",
                    ex["title"][:50], event.get("title", new_id)[:50],
                )
                break

    return edges
