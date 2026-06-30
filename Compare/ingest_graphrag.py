#!/usr/bin/env python3
"""
Compare/ingest_graphrag.py — Build Graph-RAG artefacts from the existing Kuzu graph.

Produces two outputs from the live Kuzu database (no re-ingestion required):
  1. Compare/data/graph.json  — entity nodes + relationship edges for BFS traversal
  2. tsg_graphrag             — Qdrant collection with embedded entity vectors (seed lookup)

Node format in graph.json:
  {"name": "Chen Feng", "type": "Person", "description": "..."}

Edge format in graph.json:
  {"source": "Chen Feng", "target": "Luo Yuan", "relation": "CO_OCCURS"}

Run from project root:
    python Compare/ingest_graphrag.py               # all entities + events
    python Compare/ingest_graphrag.py --reset       # drop tsg_graphrag and rebuild
    python Compare/ingest_graphrag.py --no-embed    # write graph.json only, skip Qdrant
"""

import argparse
import json
import sys
import uuid
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import kuzu
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from src.config import GRAPH_STORE_PATH, QDRANT_HOST, QDRANT_PORT, QDRANT_VECTOR_SIZE
from src.core.embed import embed

GRAPHRAG_COLLECTION = "tsg_graphrag"
GRAPH_JSON_PATH = Path("Compare/data/graph.json")
BATCH_SIZE = 64


# ─── Kuzu extraction ──────────────────────────────────────────────────────────

def fetch_entities(conn: kuzu.Connection) -> list[dict]:
    """Return all Entity nodes as dicts."""
    r = conn.execute(
        "MATCH (e:Entity) "
        "RETURN e.id, e.name, e.type, e.aliases, e.is_stub, e.confidence, e.properties"
    )
    entities = []
    while r.has_next():
        row = r.get_next()
        props: dict = {}
        try:
            props = json.loads(row[6]) if row[6] else {}
        except (json.JSONDecodeError, TypeError):
            pass
        description = props.get("description") or props.get("summary") or ""
        if not description and props:
            # Build description from first few key-value pairs
            parts = [f"{k}: {v}" for k, v in list(props.items())[:4] if v]
            description = "; ".join(parts)
        entities.append({
            "id": row[0],
            "name": row[1],
            "type": row[2] or "Unknown",
            "aliases": row[3] or [],
            "is_stub": bool(row[4]),
            "confidence": float(row[5] or 0.5),
            "description": description,
        })
    return entities


def fetch_events(conn: kuzu.Connection) -> list[dict]:
    """Return all events that have at least one participant."""
    r = conn.execute(
        "MATCH (ev:Event) "
        "WHERE size(ev.participants) > 0 "
        "RETURN ev.id, ev.title, ev.participants, ev.locations, ev.action, ev.outcome"
    )
    events = []
    while r.has_next():
        row = r.get_next()
        events.append({
            "id": row[0],
            "title": row[1] or "",
            "participants": row[2] or [],
            "locations": row[3] or [],
            "action": row[4] or "",
            "outcome": row[5] or "",
        })
    return events


def fetch_alias_edges(conn: kuzu.Connection) -> list[dict]:
    """Return ALIAS_OF relationships between entities."""
    r = conn.execute(
        "MATCH (e1:Entity)-[:ALIAS_OF]->(e2:Entity) "
        "RETURN e1.name, e2.name"
    )
    edges = []
    while r.has_next():
        row = r.get_next()
        if row[0] and row[1] and row[0] != row[1]:
            edges.append({"source": row[0], "target": row[1], "relation": "ALIAS_OF"})
    return edges


# ─── Graph building ───────────────────────────────────────────────────────────

def build_graph(entities: list[dict], events: list[dict], alias_edges: list[dict]) -> dict:
    """
    Build graph.json structure.

    Nodes: all Entity records (non-stub preferred; stubs included if no better node).
    Edges:
      - CO_OCCURS: entity pairs that appear together in the same event's participants list
      - LOCATED_IN: entity appears in event's locations list
      - ALIAS_OF: from Kuzu ALIAS_OF relationships
    """
    # Index entity names for fast lookup
    entity_names: set[str] = {e["name"].lower() for e in entities}

    # Build nodes list (deduplicated by name)
    seen_names: set[str] = set()
    nodes: list[dict] = []
    for e in entities:
        key = e["name"].lower()
        if key in seen_names:
            continue
        seen_names.add(key)
        nodes.append({
            "name": e["name"],
            "type": e["type"],
            "description": e["description"],
        })

    # Build CO_OCCURS edges from event participant lists
    seen_co: set[tuple] = set()
    co_edges: list[dict] = []
    for ev in events:
        participants = [p for p in ev["participants"] if p and p.lower() in entity_names]
        if len(participants) < 2:
            continue
        for p1, p2 in combinations(participants, 2):
            key = tuple(sorted([p1.lower(), p2.lower()]))
            if key not in seen_co:
                seen_co.add(key)
                co_edges.append({"source": p1, "target": p2, "relation": "CO_OCCURS"})

    # Build LOCATED_IN edges from event location lists
    seen_loc: set[tuple] = set()
    loc_edges: list[dict] = []
    for ev in events:
        participants = [p for p in ev["participants"] if p and p.lower() in entity_names]
        locations = [l for l in ev["locations"] if l and l.lower() in entity_names]
        for participant in participants:
            for location in locations:
                if participant.lower() == location.lower():
                    continue
                key = (participant.lower(), location.lower())
                if key not in seen_loc:
                    seen_loc.add(key)
                    loc_edges.append({"source": participant, "target": location, "relation": "LOCATED_IN"})

    edges = co_edges + loc_edges + alias_edges

    return {"nodes": nodes, "edges": edges}


# ─── Qdrant helpers ───────────────────────────────────────────────────────────

def ensure_collection(qc: QdrantClient, reset: bool) -> None:
    exists = qc.collection_exists(GRAPHRAG_COLLECTION)
    if exists and reset:
        qc.delete_collection(GRAPHRAG_COLLECTION)
        print(f"  Dropped existing '{GRAPHRAG_COLLECTION}' collection")
        exists = False
    if not exists:
        qc.create_collection(
            collection_name=GRAPHRAG_COLLECTION,
            vectors_config=VectorParams(size=QDRANT_VECTOR_SIZE, distance=Distance.COSINE),
        )
        print(f"  Created '{GRAPHRAG_COLLECTION}' ({QDRANT_VECTOR_SIZE}d cosine)")


def embed_entities(qc: QdrantClient, entities: list[dict]) -> None:
    """Embed entity name + type + description and upsert into tsg_graphrag."""
    batch: list[PointStruct] = []
    for i, e in enumerate(entities, 1):
        text = f"{e['name']} ({e['type']})"
        if e["description"]:
            text += f": {e['description'][:300]}"
        vec = embed(text)
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"graphrag:{e['name'].lower()}"))
        batch.append(PointStruct(
            id=point_id,
            vector=vec,
            payload={
                "name": e["name"],
                "type": e["type"],
                "description": e["description"][:400] if e["description"] else "",
            },
        ))
        if len(batch) >= BATCH_SIZE:
            qc.upsert(collection_name=GRAPHRAG_COLLECTION, points=batch)
            batch.clear()
        if i % 100 == 0:
            print(f"    embedded {i}/{len(entities)} entities...")

    if batch:
        qc.upsert(collection_name=GRAPHRAG_COLLECTION, points=batch)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Build Graph-RAG artefacts from Kuzu")
    parser.add_argument("--reset", action="store_true", help="Drop and rebuild tsg_graphrag collection")
    parser.add_argument("--no-embed", action="store_true", dest="no_embed", help="Skip Qdrant embedding, write graph.json only")
    args = parser.parse_args()

    print("Graph-RAG ingestion")
    print(f"  source     : Kuzu at {GRAPH_STORE_PATH}")
    print(f"  graph.json : {GRAPH_JSON_PATH}")
    print(f"  collection : {GRAPHRAG_COLLECTION}")
    print()

    if not GRAPH_STORE_PATH.exists():
        print(f"ERROR: Kuzu graph not found at {GRAPH_STORE_PATH}")
        sys.exit(1)

    db = kuzu.Database(str(GRAPH_STORE_PATH))
    conn = kuzu.Connection(db)

    print("Fetching entities from Kuzu...")
    entities = fetch_entities(conn)
    print(f"  {len(entities)} entities")

    print("Fetching events (with participants) from Kuzu...")
    events = fetch_events(conn)
    print(f"  {len(events)} events")

    print("Fetching ALIAS_OF edges from Kuzu...")
    alias_edges = fetch_alias_edges(conn)
    print(f"  {len(alias_edges)} alias edges")

    print("Building graph...")
    graph = build_graph(entities, events, alias_edges)
    print(f"  {len(graph['nodes'])} nodes")
    print(f"  {len(graph['edges'])} edges  "
          f"(CO_OCCURS: {sum(1 for e in graph['edges'] if e['relation']=='CO_OCCURS')}, "
          f"LOCATED_IN: {sum(1 for e in graph['edges'] if e['relation']=='LOCATED_IN')}, "
          f"ALIAS_OF: {sum(1 for e in graph['edges'] if e['relation']=='ALIAS_OF')})")

    GRAPH_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    GRAPH_JSON_PATH.write_text(json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote {GRAPH_JSON_PATH}  ({GRAPH_JSON_PATH.stat().st_size // 1024} KB)")

    if not args.no_embed:
        print("\nEmbedding entities into Qdrant...")
        qc = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=30)
        ensure_collection(qc, args.reset)
        # Only embed non-stub entities (or all if count is small)
        to_embed = [e for e in entities if not e["is_stub"]] or entities
        print(f"  embedding {len(to_embed)} entities (non-stub)...")
        embed_entities(qc, to_embed)
        info = qc.get_collection(GRAPHRAG_COLLECTION)
        print(f"\nDone — {info.points_count} vectors in '{GRAPHRAG_COLLECTION}'")
    else:
        print("\nSkipped Qdrant embedding (--no-embed)")

    conn = None
    db = None


if __name__ == "__main__":
    main()
