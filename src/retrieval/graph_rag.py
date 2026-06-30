"""
Graph RAG retrieval — entity nodes in Qdrant + relationship edges in graph.json.
Collection: tsg_graphrag
"""
import json
import logging
from pathlib import Path

from qdrant_client import QdrantClient

from src.config import GRAPHRAG_SEED_THRESHOLD, QDRANT_HOST, QDRANT_PORT
from src.core.embed import embed

logger = logging.getLogger(__name__)

GRAPHRAG_COLLECTION = "tsg_graphrag"
GRAPH_PATH = Path("Compare/data/graph.json")

_qc: QdrantClient | None = None


def _get_qc() -> QdrantClient:
    global _qc
    if _qc is None:
        _qc = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=30)
    return _qc


# ---------------------------------------------------------------------------
# Graph I/O
# ---------------------------------------------------------------------------

def load_graph() -> dict:
    if not GRAPH_PATH.exists():
        return {"nodes": [], "edges": []}
    return json.loads(GRAPH_PATH.read_text(encoding="utf-8"))


def _build_indexes(graph: dict) -> tuple[dict, dict]:
    """Return (node_index by name.lower(), edge_index by source.lower())."""
    node_index = {n["name"].lower(): n for n in graph["nodes"]}
    edge_index: dict[str, list] = {}
    for edge in graph["edges"]:
        src = edge["source"].lower()
        edge_index.setdefault(src, []).append(edge)
    return node_index, edge_index


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def retrieve_graph_rag(
    query: str,
    top_k: int = 5,
    max_hops: int = 2,
    score_threshold: float | None = None,
) -> dict:
    """
    1. Embed query → semantic search for seed entity nodes in Qdrant.
    2. Filter seeds by score_threshold (defaults to GRAPHRAG_SEED_THRESHOLD).
    3. BFS expand up to max_hops over the relationship graph.
    4. Return collected nodes + edges + formatted context string.
    """
    threshold = score_threshold if score_threshold is not None else GRAPHRAG_SEED_THRESHOLD
    vec = embed(query)

    candidates = _get_qc().query_points(collection_name=GRAPHRAG_COLLECTION, query=vec, limit=top_k).points
    seed_results = [r for r in candidates if r.score >= threshold]
    if not seed_results:
        logger.info(
            "graph_rag | no seeds above threshold=%.2f (best=%.3f) | query=%r",
            threshold,
            candidates[0].score if candidates else 0.0,
            query[:60],
        )
        return {"nodes": [], "edges": [], "context": f"[GRAPH RAG] no seed nodes above threshold {threshold:.2f}"}

    graph = load_graph()
    node_index, edge_index = _build_indexes(graph)

    seed_names = {r.payload.get("name", "").lower() for r in seed_results if r.payload.get("name")}

    visited: set[str] = set()
    collected_nodes: list[dict] = []
    collected_edges: list[dict] = []

    frontier = list(seed_names)
    for _ in range(max_hops + 1):
        if not frontier:
            break
        next_frontier: list[str] = []
        for name in frontier:
            if name in visited:
                continue
            visited.add(name)
            if name in node_index:
                collected_nodes.append(node_index[name])
            for edge in edge_index.get(name, []):
                if edge not in collected_edges:
                    collected_edges.append(edge)
                tgt = edge["target"].lower()
                if tgt not in visited:
                    next_frontier.append(tgt)
        frontier = next_frontier

    context = _format_graph_context(collected_nodes, collected_edges, seed_results)
    logger.info("graph_rag | query=%r  nodes=%d  edges=%d", query[:60], len(collected_nodes), len(collected_edges))
    return {"nodes": collected_nodes, "edges": collected_edges, "context": context}


def _format_graph_context(nodes: list[dict], edges: list[dict], seed_results) -> str:
    lines = ["[GRAPH RAG]"]

    seed_scores = {r.payload.get("name", "").lower(): round(r.score, 4) for r in seed_results}

    lines.append("\nEntities:")
    for node in nodes:
        score = seed_scores.get(node["name"].lower(), "")
        score_str = f" | seed score {score}" if score else ""
        lines.append(f"  [{node.get('type', '?')}]{score_str} {node['name']}: {node.get('description', '')}")

    if edges:
        lines.append("\nRelationships:")
        for edge in edges:
            lines.append(f"  {edge['source']} --[{edge['relation']}]--> {edge['target']}")

    return "\n".join(lines)


def format_graph_context(result: dict) -> str:
    return result.get("context", "[GRAPH RAG] no context")
