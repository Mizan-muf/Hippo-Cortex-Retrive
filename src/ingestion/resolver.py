import json
import logging
from datetime import datetime, timezone

from src.storage.graph import (
    add_causes,
    add_located_at,
    add_participant_in,
    add_precedes,
    find_entity_by_name,
    find_event_by_id,
    upsert_entity,
    upsert_event,
)
from src.storage.memory import search_knowledge, store_pathway
from src.storage.filestore import write_entity

logger = logging.getLogger(__name__)

# Maps spaCy NER label → Kuzu entity type
_NER_TO_ENTITY_TYPE: dict[str, str] = {
    "PERSON": "person",
    "ORG": "org",
    "NORP": "group",
}

# Maps spaCy NER label → filestore folder name
_NER_TO_FOLDER: dict[str, str] = {
    "PERSON": "People",
    "ORG": "Organizations",
    "NORP": "Groups",
}


def _entity_summary(name: str, entity_type: str, properties: dict) -> str:
    prop_str = ", ".join(f"{k}: {v}" for k, v in properties.items()) if properties else ""
    if prop_str:
        return f"Entity: {name} | Type: {entity_type} | Properties: {prop_str}"
    return f"Entity: {name} | Type: {entity_type}"


def _event_summary(event: dict) -> str:
    parts = [f"Event: {event.get('title', '')}"]
    if event.get("action"):
        parts.append(f"Action: {event['action']}")
    if event.get("outcome"):
        parts.append(f"Outcome: {event['outcome']}")
    if event.get("participants"):
        parts.append(f"Participants: {', '.join(event['participants'])}")
    if event.get("locations"):
        parts.append(f"Locations: {', '.join(event['locations'])}")
    if event.get("summary"):
        parts.append(f"Sentence: {event['summary']}")
    if event.get("source_text"):
        parts.append(f"Context: {event['source_text']}")
    return " | ".join(parts)


def _now_ts() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolve_trigger(event_id: str, entity_ids: list[str]) -> str:
    """Determine write trigger: AUTO_MISS, NEW_FACT, or SKIP."""
    if not find_event_by_id(event_id):
        return "AUTO_MISS"
    # Event exists — check if any entity participation edges are new
    # For simplicity: if the event already exists, treat as NEW_FACT (edge may be new)
    return "NEW_FACT"


def store_event(
    event: dict,
    precedes_edges: list[tuple[str, str]] | None = None,
    *,
    force: bool = False,
) -> None:
    """Stage 6: write an event and its resolved entities to Kuzu + Mem0 + filestore.

    event must have 'participant_ids' and 'location_ids' set by entity_linker.
    precedes_edges: list of (from_event_id, to_event_id) from temporal_linker.
    """
    participant_ids: dict[str, str | None] = event.get("participant_ids", {})
    location_ids: dict[str, str | None] = event.get("location_ids", {})
    participant_types: dict[str, str] = event.get("participant_types", {})

    # --- Ensure all participant entities exist in Kuzu ---
    resolved_participant_ids: dict[str, str] = {}
    for name, node_id in participant_ids.items():
        if node_id:
            resolved_participant_ids[name] = node_id
        else:
            ner_label = participant_types.get(name, "PERSON")
            entity_type = _NER_TO_ENTITY_TYPE.get(ner_label, "person")
            was_created, eid = upsert_entity(name, entity_type, is_stub=True)
            resolved_participant_ids[name] = eid
            if was_created:
                from src.core.terms import entity_key_terms
                terms = entity_key_terms(name, entity_type, {})
                try:
                    store_pathway(terms, [eid], {"entity_id": eid, "entity_name": name, "type": "entity", "doc_id": event.get("document_id")})
                except Exception as exc:
                    logger.warning("resolver | mem0 entity write failed for %r: %s", name, exc)

    resolved_location_ids: dict[str, str] = {}
    for name, node_id in location_ids.items():
        if node_id:
            resolved_location_ids[name] = node_id
        else:
            was_created, eid = upsert_entity(name, "location", is_stub=True)
            resolved_location_ids[name] = eid
            if was_created:
                from src.core.terms import entity_key_terms
                terms = entity_key_terms(name, "location", {})
                try:
                    store_pathway(terms, [eid], {"entity_id": eid, "entity_name": name, "type": "entity", "doc_id": event.get("document_id")})
                except Exception as exc:
                    logger.warning("resolver | mem0 entity write failed for %r: %s", name, exc)

    # --- Determine trigger and write event ---
    trigger = "EXPLICIT" if force else _resolve_trigger(event.get("id", ""), [])

    if trigger in ("AUTO_MISS", "EXPLICIT"):
        was_created, event_id = upsert_event(event)
        event["id"] = event_id

        if was_created or force:
            from src.core.terms import event_key_terms
            terms = event_key_terms(event)
            pathway = [event_id] + list(resolved_participant_ids.values())[:3]
            try:
                store_pathway(
                    terms,
                    pathway,
                    {
                        "event_id": event_id,
                        "title": event.get("title", ""),
                        "type": "event",
                        "doc_id": event.get("document_id"),
                    },
                )
            except Exception as exc:
                logger.warning("resolver | mem0 event write failed: %s", exc)

            # non-blocking filestore writes for participants, routed to the correct folder
            event_summary = _event_summary(event)
            for name in event.get("participants", []):
                ner_label = participant_types.get(name, "PERSON")
                folder = _NER_TO_FOLDER.get(ner_label, "People")
                try:
                    write_entity(name, folder, event_summary)
                except Exception as exc:
                    logger.debug("resolver | filestore write failed for %r: %s", name, exc)

    elif trigger == "NEW_FACT":
        # event exists — still ensure edges are wired
        event_id = event.get("id", "")

    else:
        logger.debug("resolver | skip event %s — already fully stored", event.get("id"))
        return

    event_id = event.get("id", "")

    # --- Write PARTICIPANT_IN edges ---
    for name, eid in resolved_participant_ids.items():
        add_participant_in(eid, event_id)

    # --- Write LOCATED_AT edges (specificity = position in locations list) ---
    for i, name in enumerate(event.get("locations", [])):
        loc_id = resolved_location_ids.get(name)
        if loc_id:
            add_located_at(event_id, loc_id, specificity=i)

    # --- Write PRECEDES edges from temporal linker ---
    if precedes_edges:
        for from_id, to_id in precedes_edges:
            if to_id == event_id:
                add_precedes(from_id, to_id)

    logger.info(
        "resolver | stored event [%s] %s | trigger=%s | participants=%d",
        event_id[:12], event.get("title", "")[:50], trigger, len(resolved_participant_ids),
    )


def store_properties(properties: list[dict]) -> None:
    """Write entity properties to Kuzu and Mem0."""
    for prop in properties:
        name = prop.get("entity", "")
        attr = prop.get("attribute", "")
        value = prop.get("value", "")
        if not name or not attr:
            continue

        existing = find_entity_by_name(name)
        if existing:
            upsert_entity(
                name, existing["type"],
                properties={attr: value},
                is_stub=existing["is_stub"],
                confidence=existing["confidence"],
                entity_id=existing["id"],
            )
        else:
            was_created, eid = upsert_entity(name, "unknown", properties={attr: value})

        # Update Mem0 summary
        current = find_entity_by_name(name)
        if current:
            from src.core.terms import entity_key_terms
            terms = entity_key_terms(name, current["type"], current.get("properties", {}))
            try:
                store_pathway(
                    terms,
                    [current["id"]],
                    {"entity_id": current["id"], "entity_name": name, "type": "entity", "doc_id": prop.get("document_id")},
                )
            except Exception as exc:
                logger.warning("resolver | mem0 property update failed for %r: %s", name, exc)

        logger.info("resolver | property | %s.%s = %s", name, attr, str(value)[:40])


def store_events_sequential(
    events: list[dict],
    precedes_edges: list[tuple[str, str]] | None = None,
) -> None:
    """Store a list of events one at a time. Failures skip to next event."""
    for i, event in enumerate(events):
        try:
            store_event(event, precedes_edges)
        except Exception as exc:
            logger.error(
                "resolver | event %d/%d failed — skipping. title=%r err=%s",
                i + 1, len(events), event.get("title", ""), exc,
            )
