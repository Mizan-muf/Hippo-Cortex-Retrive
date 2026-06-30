"""
Write pipeline orchestrator — seven stages in strict sequence.

Entry points:
  ingest(text, doc_id, title)  — full pipeline, stages 0-6
  store(event_dict)            — explicit write, stages 5-6 only

Stage sequence:
  0  Conflict check        memory.check_conflicts
  1b Coreference resolve   coref.resolve (full document, before segmentation)
  1  Segmentation          segmenter.segment (runs on coref-resolved text)
  2  Event extraction      decomposer.extract_events (spaCy NER + dep parse)
  3  Property extraction   decomposer.extract_properties (spaCy dep patterns)
  4  Temporal ordering     temporal_linker.assign_temporal_order
  5  Entity linking        entity_linker.resolve_event
  4b Cross-doc causal      temporal_linker.link_cross_document (needs resolved IDs)
  6  Storage               resolver.store_events_sequential + store_properties
"""

import logging

import src.ingestion.coref as coref
from src.ingestion.decomposer import extract_events, extract_properties
from src.ingestion.entity_linker import flush_ambiguous_queue, resolve_event
from src.storage.graph import is_initialized
from src.storage.memory import check_conflicts
from src.ingestion.resolver import store_event, store_events_sequential, store_properties
from src.ingestion.segmenter import segment
from src.ingestion.temporal_linker import assign_temporal_order, link_cross_document, link_within_document

logger = logging.getLogger(__name__)


def ingest(text: str, doc_id: int = 0, title: str = "") -> dict:
    """Run the full 7-stage write pipeline on a document.

    Returns a summary dict with counts of events, properties, and any conflicts.
    """
    if not is_initialized():
        raise RuntimeError("Graph not initialized — call init_graph() first")

    result = {
        "doc_id": doc_id,
        "title": title,
        "conflict": None,
        "segments": 0,
        "events": 0,
        "properties": 0,
        "ambiguous_entities": 0,
    }

    # --- Stage 0: Conflict check ---
    logger.info("pipeline | stage 0 | conflict check | doc_id=%d", doc_id)
    conflict = check_conflicts(text[:1200])
    if conflict:
        result["conflict"] = conflict
        logger.warning("pipeline | conflict detected: %s", conflict)

    # --- Stage 1b: Coreference resolution (full document, before segmentation) ---
    logger.info("pipeline | stage 1b | coreference resolution | doc_id=%d", doc_id)
    resolved_text = coref.resolve(text)

    # --- Stage 1: Segmentation ---
    logger.info("pipeline | stage 1 | segmentation | doc_id=%d", doc_id)
    segments = segment(resolved_text)
    result["segments"] = len(segments)
    logger.info("pipeline | %d segments produced", len(segments))

    all_events: list[dict] = []
    all_properties: list[dict] = []

    for seg in segments:
        seg_id = seg["segment_id"]
        seg_text = seg["text"]

        # --- Stage 2: Event extraction ---
        events = extract_events(seg_text, doc_id, seg_id)
        all_events.extend(events)

        # --- Stage 3: Property extraction ---
        props = extract_properties(seg_text, doc_id, seg_id)
        all_properties.extend(props)

    result["events"] = len(all_events)
    result["properties"] = len(all_properties)

    # --- Stage 4a: Temporal ordering + ID pre-assignment ---
    logger.info("pipeline | stage 4 | temporal ordering | %d events", len(all_events))
    all_events = assign_temporal_order(all_events)

    # --- Stage 4b (intra): Within-document PRECEDES edges ---
    intra_edges = link_within_document(all_events)
    logger.info("pipeline | intra-doc PRECEDES: %d edges", len(intra_edges))

    # --- Stage 5: Entity linking ---
    logger.info("pipeline | stage 5 | entity linking | %d events", len(all_events))
    for event in all_events:
        resolve_event(event)

    # --- Stage 4b (cross): Cross-document PRECEDES edges ---
    entity_id_map: dict[str, str] = {}
    for event in all_events:
        entity_id_map.update(event.get("participant_ids", {}))
        entity_id_map.update(event.get("location_ids", {}))
    entity_id_map = {k: v for k, v in entity_id_map.items() if v}

    cross_edges = link_cross_document(all_events, entity_id_map)
    logger.info("pipeline | cross-doc PRECEDES: %d edges", len(cross_edges))

    precedes_edges = intra_edges + cross_edges

    # --- Stage 6: Storage ---
    logger.info("pipeline | stage 6 | storage | %d events, %d properties", len(all_events), len(all_properties))
    store_events_sequential(all_events, precedes_edges)
    store_properties(all_properties)

    # Surface ambiguous entity links
    ambiguous = flush_ambiguous_queue()
    result["ambiguous_entities"] = len(ambiguous)
    if ambiguous:
        logger.warning(
            "pipeline | %d ambiguous entity links queued for HITL review:", len(ambiguous)
        )
        for item in ambiguous:
            logger.warning(
                "  ⚠ %r ~ %r (score %.2f)",
                item["input_name"], item.get("candidate", "?"), item.get("score", 0.0),
            )

    logger.info(
        "pipeline | done | doc_id=%d | segments=%d events=%d props=%d ambiguous=%d",
        doc_id, result["segments"], result["events"], result["properties"], result["ambiguous_entities"],
    )
    return result


def store(event_dict: dict, *, force: bool = True) -> None:
    """Explicit write path — Stage 5 + 6 only.

    Bypasses segmentation and extraction. Accepts a pre-formed event dict.
    entity_linker resolves names, resolver writes to stores.
    """
    if not is_initialized():
        raise RuntimeError("Graph not initialized — call init_graph() first")

    resolve_event(event_dict)
    store_event(event_dict, force=force)
    logger.info("pipeline | explicit store | %s", event_dict.get("title", "")[:60])
