import logging
import math
from dataclasses import dataclass, field
from functools import lru_cache

from src.config import (
    GRAPH_ANCHOR_MAX_STARTS,
    GRAPH_ANCHOR_THRESHOLD,
    PROPAGATION_DECAY,
    PROPAGATION_HOP_THRESHOLD,
    PROPAGATION_MAX_DEPTH,
    PROPAGATION_MAX_NODES,
    STUB_PENALTY,
)
from src.core.embed import embed
from src.storage.graph import (
    find_entity_by_name,
    get_entity_neighbors,
    get_event_causal_neighbors,
    get_event_entity_neighbors,
    get_event_location_neighbors,
)
from src.storage.memory import search_knowledge

logger = logging.getLogger(__name__)


@dataclass
class PropNode:
    node_id: str
    node_type: str        # "entity" | "event"
    name: str
    content: str
    is_stub: bool
    hop: int
    intrinsic_score: float
    prop_score: float


@dataclass
class PropagationResult:
    nodes: list[PropNode] = field(default_factory=list)
    start_entity: str = ""
    max_hop: int = 0


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return dot / (mag_a * mag_b)


@lru_cache(maxsize=4096)
def _cached_embed(name: str) -> tuple[float, ...]:
    return tuple(embed(name))


def _score_node(name: str, query_vec: list[float]) -> float:
    """Cosine similarity between a node name and a pre-computed query vector."""
    node_vec = list(_cached_embed(name))
    return _cosine(query_vec, node_vec)


def _get_neighbors(node_id: str, node_type: str) -> list[dict]:
    if node_type == "entity":
        return get_entity_neighbors(node_id)
    return (
        get_event_entity_neighbors(node_id)
        + get_event_location_neighbors(node_id)
        + get_event_causal_neighbors(node_id)
    )


def propagate(
    query: str,
    start_node_id: str,
    start_node_type: str,
    start_node_name: str,
    start_node_content: str,
    start_is_stub: bool = False,
) -> PropagationResult:
    """BFS graph traversal with decaying semantic scoring.

    Starts from the given node and walks connected edges in Kuzu.
    Each neighbor is scored against the query via Mem0, with a decay
    multiplier per hop. Nodes below HOP_THRESHOLD are pruned.
    """
    result = PropagationResult(start_entity=start_node_name)
    visited: set[str] = {start_node_id}

    query_vec = embed(query)
    start_intrinsic = _score_node(start_node_name, query_vec)
    start_prop = start_intrinsic * (STUB_PENALTY if start_is_stub else 1.0)

    start = PropNode(
        node_id=start_node_id,
        node_type=start_node_type,
        name=start_node_name,
        content=start_node_content,
        is_stub=start_is_stub,
        hop=0,
        intrinsic_score=start_intrinsic,
        prop_score=start_prop,
    )
    result.nodes.append(start)
    frontier = [start]

    for depth in range(1, PROPAGATION_MAX_DEPTH + 1):
        if not frontier or len(result.nodes) >= PROPAGATION_MAX_NODES:
            break

        next_frontier: list[PropNode] = []
        for current in frontier:
            for nbr in _get_neighbors(current.node_id, current.node_type):
                nid = nbr["node_id"]
                if nid in visited or len(result.nodes) >= PROPAGATION_MAX_NODES:
                    continue
                visited.add(nid)

                intrinsic = _score_node(nbr["name"], query_vec)
                decay = PROPAGATION_DECAY ** depth
                prop_score = intrinsic * decay
                if nbr.get("is_stub"):
                    prop_score *= STUB_PENALTY

                if prop_score < PROPAGATION_HOP_THRESHOLD:
                    continue

                node = PropNode(
                    node_id=nid,
                    node_type=nbr["node_type"],
                    name=nbr["name"],
                    content=nbr["content"],
                    is_stub=nbr.get("is_stub", False),
                    hop=depth,
                    intrinsic_score=intrinsic,
                    prop_score=prop_score,
                )
                result.nodes.append(node)
                next_frontier.append(node)

        frontier = next_frontier
        if frontier:
            result.max_hop = depth

    result.nodes.sort(key=lambda n: n.prop_score, reverse=True)
    logger.info(
        "propagator | query=%r  start=%r  nodes=%d  max_hop=%d",
        query[:60], start_node_name, len(result.nodes), result.max_hop,
    )
    return result


def find_start_node(query: str) -> dict | None:
    """Find the best graph start node for this query using Mem0 results.

    Checks both Entity and Event records. Returns the highest-scoring node
    that exists in the Kuzu graph.
    """
    from src.storage.graph import find_event_by_id

    results = search_knowledge(query, top_k=8)
    logger.debug(
        "find_start_node | query=%r | %d candidates | anchor_threshold=%.2f",
        query[:60], len(results), GRAPH_ANCHOR_THRESHOLD,
    )

    for r in results:
        score = float(r.get("score", 0.0))
        text = r.get("memory", "")
        meta = r.get("metadata", {}) or {}

        if score < GRAPH_ANCHOR_THRESHOLD:
            logger.debug(
                "find_start_node | SKIP score=%.3f < %.2f | %s",
                score, GRAPH_ANCHOR_THRESHOLD, text[:80],
            )
            continue

        # Entity anchor
        if text.startswith("Entity: "):
            name = text[8:].split(" | ")[0].strip()
            entity = find_entity_by_name(name)
            if entity:
                logger.info(
                    "find_start_node | ENTITY anchor score=%.3f stub=%s | %s",
                    score, entity.get("is_stub"), name,
                )
                return {
                    "node_id": entity["id"],
                    "node_type": "entity",
                    "name": entity["name"],
                    "content": text,
                    "is_stub": entity.get("is_stub", False),
                    "score": score,
                }
            logger.debug("find_start_node | entity %r not in graph — skip", name)

        # Event anchor — use stored event_id from metadata
        elif text.startswith("Event: ") and meta.get("event_id"):
            event = find_event_by_id(meta["event_id"])
            if event:
                logger.info(
                    "find_start_node | EVENT anchor score=%.3f | %s",
                    score, event.get("title", "")[:60],
                )
                return {
                    "node_id": event["id"],
                    "node_type": "event",
                    "name": event.get("title", ""),
                    "content": text,
                    "is_stub": False,
                    "score": score,
                }
            logger.debug(
                "find_start_node | event_id %r not in graph — skip", meta.get("event_id")
            )
        else:
            logger.debug(
                "find_start_node | SKIP no event_id in meta score=%.3f | %s",
                score, text[:80],
            )

    logger.info("find_start_node | no anchor found for query=%r", query[:60])
    return None


def find_start_nodes(query: str) -> list[dict]:
    """Return up to GRAPH_ANCHOR_MAX_STARTS anchor nodes for graph propagation.

    Search order:
      1. Direct Kuzu token match — finds entities whose name contains all query
         tokens (case-insensitive). This works even when Mem0 has no entity record.
      2. Mem0 semantic search — catches Event anchors and any Entity records that
         ARE stored in Qdrant.
    """
    from src.storage.graph import find_event_by_id, search_entities_by_query

    starts: list[dict] = []
    seen_node_ids: set[str] = set()

    # --- Step 1: direct Kuzu entity name match ---
    direct = search_entities_by_query(query, limit=GRAPH_ANCHOR_MAX_STARTS)
    for entity in direct:
        if len(starts) >= GRAPH_ANCHOR_MAX_STARTS:
            break
        logger.info(
            "find_start_nodes | DIRECT kuzu match stub=%s | %s",
            entity.get("is_stub"), entity["name"],
        )
        starts.append({
            "node_id": entity["id"],
            "node_type": "entity",
            "name": entity["name"],
            "content": f"Entity: {entity['name']} | Type: {entity['type']}",
            "is_stub": entity.get("is_stub", False),
            "score": 1.0,
        })
        seen_node_ids.add(entity["id"])

    # --- Step 2: Mem0 semantic search for additional Event/Entity anchors ---
    results = search_knowledge(query, top_k=8)
    logger.debug(
        "find_start_nodes | query=%r | %d mem0 candidates | threshold=%.2f | max_starts=%d",
        query[:60], len(results), GRAPH_ANCHOR_THRESHOLD, GRAPH_ANCHOR_MAX_STARTS,
    )

    for r in results:
        if len(starts) >= GRAPH_ANCHOR_MAX_STARTS:
            break

        score = float(r.get("score", 0.0))
        text = r.get("memory", "")
        meta = r.get("metadata", {}) or {}

        if score < GRAPH_ANCHOR_THRESHOLD:
            continue

        if text.startswith("Entity: "):
            name = text[8:].split(" | ")[0].strip()
            entity = find_entity_by_name(name)
            if entity and entity["id"] not in seen_node_ids:
                logger.info(
                    "find_start_nodes | MEM0 ENTITY anchor score=%.3f stub=%s | %s",
                    score, entity.get("is_stub"), name,
                )
                starts.append({
                    "node_id": entity["id"],
                    "node_type": "entity",
                    "name": entity["name"],
                    "content": text,
                    "is_stub": entity.get("is_stub", False),
                    "score": score,
                })
                seen_node_ids.add(entity["id"])

        elif text.startswith("Event: ") and meta.get("event_id"):
            event = find_event_by_id(meta["event_id"])
            if event and event["id"] not in seen_node_ids:
                logger.info(
                    "find_start_nodes | MEM0 EVENT anchor score=%.3f | %s",
                    score, event.get("title", "")[:60],
                )
                starts.append({
                    "node_id": event["id"],
                    "node_type": "event",
                    "name": event.get("title", ""),
                    "content": text,
                    "is_stub": False,
                    "score": score,
                })
                seen_node_ids.add(event["id"])

    logger.info("find_start_nodes | %d anchors found for query=%r", len(starts), query[:60])
    return starts


def format_propagation_result(result: PropagationResult) -> str:
    if not result.nodes:
        return "[no graph context found]"
    lines = [
        f"[PROPAGATED KNOWLEDGE — {len(result.nodes)} nodes across {result.max_hop} hops]"
    ]
    for node in result.nodes:
        lines.append(
            f"\n[hop {node.hop} | {node.node_type} | score {node.prop_score:.2f}] {node.name}"
        )
        lines.append(f"  → {node.content[:200]}")
    return "\n".join(lines)
