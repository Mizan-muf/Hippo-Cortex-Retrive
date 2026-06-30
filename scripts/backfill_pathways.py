"""Backfill multi-path pathway signatures for existing Kuzu data into Qdrant.

For each Event, generates up to 5 pathway points (one per query angle):
  who_what, who_result, where_what, cause_effect, summary
For each Entity, generates up to 2+ pathway points:
  name_type, name_props, alias_N (one per alias)

Each point is a separate Qdrant vector embedding the relevant key-term bag,
with a `pathway` field listing the Kuzu node IDs to reconstruct from.
Safe to re-run — upserts are idempotent (UUID5 keyed on key_terms).

Estimated runtime: ~64k events × 5 variants × embed_time
  (nomic-embed-text via Ollama: ~30–60 min; fast GPU: ~10 min)

Usage:
    python scripts/backfill_pathways.py [--dry-run] [--limit N]
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _backfill_events(conn, dry_run: bool, limit: int | None = None) -> tuple[int, int]:
    """Returns (events_processed, pathway_points_written)."""
    from src.core.terms import event_key_terms_variants
    from src.storage.memory import store_pathway

    r = conn.execute(
        "MATCH (ev:Event) "
        "RETURN ev.id, ev.title, ev.summary, ev.action, ev.outcome, "
        "ev.participants, ev.locations, ev.document_id"
    )
    events_done = 0
    points_written = 0
    while r.has_next():
        if limit is not None and events_done >= limit:
            break
        row = r.get_next()
        event_id, title, summary, action, outcome, participants, locations, doc_id = row
        event = {
            "title": title or "",
            "summary": summary or "",
            "action": action or "",
            "outcome": outcome or "",
            "participants": participants or [],
            "locations": locations or [],
        }

        # fetch participant entity IDs for the pathway
        pr = conn.execute(
            "MATCH (e:Entity)-[:PARTICIPANT_IN]->(ev:Event {id: $id}) RETURN e.id LIMIT 3",
            {"id": event_id},
        )
        participant_ids = []
        while pr.has_next():
            participant_ids.append(pr.get_next()[0])

        pathway = [event_id] + participant_ids
        base_metadata = {
            "event_id": event_id,
            "title": title or "",
            "type": "event",
            "doc_id": doc_id,
        }

        variants = event_key_terms_variants(event)

        if dry_run:
            for label, terms in variants:
                logger.info(
                    "DRY event %s [%s]: %r → %d nodes",
                    event_id[:12], label, terms[:60], len(pathway),
                )
        else:
            for label, terms in variants:
                try:
                    store_pathway(terms, pathway, {**base_metadata, "variant": label})
                    points_written += 1
                except Exception as exc:
                    logger.warning("event %s [%s] failed: %s", event_id[:12], label, exc)

        events_done += 1
        if events_done % 1000 == 0:
            logger.info(
                "events processed: %d  pathway points written: %d",
                events_done, points_written,
            )

    return events_done, points_written


def _backfill_entities(conn, dry_run: bool) -> tuple[int, int]:
    """Returns (entities_processed, pathway_points_written)."""
    from src.core.terms import entity_key_terms_variants
    from src.storage.memory import store_pathway

    r = conn.execute(
        "MATCH (e:Entity) RETURN e.id, e.name, e.type, e.properties, e.aliases"
    )
    entities_done = 0
    points_written = 0
    errors = 0
    while r.has_next():
        row = r.get_next()
        entity_id, name, entity_type, props_json, aliases = row
        props = json.loads(props_json) if props_json else {}
        aliases = aliases or []

        base_metadata = {
            "entity_id": entity_id,
            "entity_name": name or "",
            "type": "entity",
        }

        variants = entity_key_terms_variants(name or "", entity_type or "", props, aliases)

        if dry_run:
            for label, terms in variants:
                logger.info(
                    "DRY entity %s [%s]: %r → [%s]",
                    entity_id[:12], label, terms[:60], entity_id[:12],
                )
        else:
            for label, terms in variants:
                try:
                    store_pathway(terms, [entity_id], {**base_metadata, "variant": label})
                    points_written += 1
                except Exception as exc:
                    logger.warning("entity %s [%s] failed: %s", entity_id[:12], label, exc)
                    errors += 1

        entities_done += 1
        if entities_done % 500 == 0:
            logger.info(
                "entities processed: %d  pathway points written: %d (errors: %d)",
                entities_done, points_written, errors,
            )

    return entities_done, points_written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Log what would be written, don't write")
    parser.add_argument("--limit", type=int, default=None, metavar="N",
                        help="Process only the first N events (useful for smoke-testing)")
    args = parser.parse_args()

    from src.storage.graph import init_graph, _conn_required
    from src.storage.memory import init_memory

    if not init_graph():
        logger.critical("graph init failed — aborting")
        sys.exit(1)
    if not args.dry_run and not init_memory():
        logger.critical("memory init failed — aborting")
        sys.exit(1)

    conn = _conn_required()

    logger.info("backfill starting (dry_run=%s, limit=%s)", args.dry_run, args.limit)

    n_events, pts_events = _backfill_events(conn, args.dry_run, args.limit)
    logger.info("events done: %d  points written: %d", n_events, pts_events)

    n_entities, pts_entities = _backfill_entities(conn, args.dry_run)
    logger.info("entities done: %d  points written: %d", n_entities, pts_entities)

    logger.info(
        "backfill complete — events=%d entities=%d  total_pathway_points=%d",
        n_events, n_entities, pts_events + pts_entities,
    )


if __name__ == "__main__":
    main()
