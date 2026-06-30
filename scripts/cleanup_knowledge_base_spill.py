"""Remove spilled pathway points from knowledge_base by reconstructing their IDs.

The first backfill run wrote ~4,588 pathway points to knowledge_base (the wrong
collection) before the architecture was fixed. Those points lack BM25 sparse
vectors, causing Qdrant scroll to panic. They also pollute search_knowledge()
results with sparse key-term bags instead of readable memories.

This script reconstructs each point ID (uuid5 of the key_terms string) using
the same logic as store_pathway(), then deletes them directly — no scroll needed.

Usage:
    python scripts/cleanup_knowledge_base_spill.py [--dry-run]
"""
import argparse
import logging
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _pathway_point_id(key_terms: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"pathway:{key_terms[:80]}"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    from src.storage.graph import init_graph, _conn_required
    from src.storage.memory import _get_qc
    from src.core.terms import event_key_terms_variants, entity_key_terms_variants
    import json

    if not init_graph():
        logger.critical("graph init failed"); sys.exit(1)

    conn = _conn_required()
    qc = _get_qc()

    ids_to_delete: list[str] = []

    # --- Reconstruct event pathway point IDs (first 2000 events = what was spilled) ---
    r = conn.execute(
        "MATCH (ev:Event) "
        "RETURN ev.id, ev.title, ev.summary, ev.action, ev.outcome, "
        "ev.participants, ev.locations LIMIT 2000"
    )
    event_count = 0
    while r.has_next():
        row = r.get_next()
        event_id, title, summary, action, outcome, participants, locations = row
        event = {
            "title": title or "", "summary": summary or "",
            "action": action or "", "outcome": outcome or "",
            "participants": participants or [], "locations": locations or [],
        }
        for _label, terms in event_key_terms_variants(event):
            ids_to_delete.append(_pathway_point_id(terms))
        event_count += 1

    logger.info("events scanned: %d  candidate IDs: %d", event_count, len(ids_to_delete))

    # --- Deduplicate ---
    ids_to_delete = list(set(ids_to_delete))
    logger.info("unique IDs to delete: %d", len(ids_to_delete))

    if args.dry_run:
        logger.info("DRY RUN — no deletions performed. Sample IDs: %s", ids_to_delete[:3])
        return

    # --- Delete in batches of 500 ---
    from qdrant_client.models import PointIdsList
    BATCH = 500
    deleted = 0
    for i in range(0, len(ids_to_delete), BATCH):
        batch = ids_to_delete[i:i + BATCH]
        try:
            qc.delete("knowledge_base", points_selector=PointIdsList(points=batch))
            deleted += len(batch)
            logger.info("deleted batch %d/%d (%d points)", i // BATCH + 1,
                        (len(ids_to_delete) + BATCH - 1) // BATCH, len(batch))
        except Exception as exc:
            logger.warning("batch %d failed: %s", i // BATCH + 1, exc)

    after = qc.get_collection("knowledge_base").points_count
    logger.info("done — deleted ~%d points | knowledge_base now: %d", deleted, after)


if __name__ == "__main__":
    main()