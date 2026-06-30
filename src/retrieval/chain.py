import logging
from dataclasses import dataclass
from typing import Literal

from src.config import CONFIDENT_THRESHOLD, START_THRESHOLD
from src.storage.filestore import grep_filestore
from src.storage.memory import search_knowledge, search_pathways, store_fact_direct, store_pathway

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    hits: list[str]
    source: Literal["tier1", "tier1+tier2", "tier2", "miss"]
    confidence: float          # highest Tier 1 score, or 0.0 on graph/file/miss
    reconsolidated: bool = False


def _tier1_pathway(query: str) -> "RetrievalResult | None":
    """Tier 1 fast path: match pathway signatures, fetch verbatim node content from Kuzu."""
    from src.storage.graph import fetch_nodes_by_ids, is_initialized

    # Pathway reconstruction requires Kuzu — skip gracefully if graph is unavailable
    # (e.g. another process holds the write lock during a backfill run).
    if not is_initialized():
        return None

    results = search_pathways(query, top_k=5)
    pathway_results = [r for r in results if r.get("pathway") and r.get("score", 0) >= START_THRESHOLD]

    if not pathway_results:
        return None

    node_ids: list[str] = []
    seen: set[str] = set()
    for r in sorted(pathway_results, key=lambda x: x["score"], reverse=True):
        for nid in r["pathway"]:
            if nid not in seen:
                seen.add(nid)
                node_ids.append(nid)

    nodes = fetch_nodes_by_ids(node_ids[:15])
    if not nodes:
        return None

    hits = [n["content"] for n in nodes if n.get("content")]
    best_score = pathway_results[0]["score"]
    return RetrievalResult(hits=hits, source="tier1", confidence=best_score)


def retrieve(query: str, *, deep_search: bool = False) -> RetrievalResult:
    """Three-tier retrieval: Mem0 → Kuzu graph propagation → file grep fallback.

    Tier 1 only:     score >= CONFIDENT_THRESHOLD and not deep_search
    Tier 1 + Tier 2: START_THRESHOLD <= score < CONFIDENT_THRESHOLD (soft hit)
    Tier 2 only:     score < START_THRESHOLD or deep_search
    """
    # --- Tier 1a: pathway fast path ---
    if not deep_search:
        pathway_result = _tier1_pathway(query)
        if pathway_result and pathway_result.confidence >= CONFIDENT_THRESHOLD:
            logger.info(
                "retrieve | tier1 pathway hit | score=%.3f | query=%r",
                pathway_result.confidence, query[:60],
            )
            return pathway_result

    # --- Tier 1: semantic search ---
    results = search_knowledge(query, top_k=5)
    tier1_hits = [r for r in results if r.get("score", 0.0) >= START_THRESHOLD]
    best_score = max((r.get("score", 0.0) for r in tier1_hits), default=0.0)

    very_confident = best_score >= CONFIDENT_THRESHOLD

    if very_confident and not deep_search:
        hits = [r.get("memory", "") for r in tier1_hits if r.get("memory")]
        logger.info(
            "retrieve | tier1 confident | score=%.3f | query=%r", best_score, query[:60]
        )
        return RetrievalResult(hits=hits, source="tier1", confidence=best_score)

    # Soft Tier 1 hit or deep_search: keep tier1 results and enrich with Tier 2
    tier1_memories = [r.get("memory", "") for r in tier1_hits if r.get("memory")]
    if tier1_memories:
        logger.info(
            "retrieve | tier1 soft hit score=%.3f — escalating to tier2 | query=%r",
            best_score, query[:60],
        )

    # --- Tier 2: Kuzu graph propagation ---
    tier2_result = _tier2_graph(query)
    if tier2_result is not None:
        if tier1_memories:
            merged_hits = tier2_result.hits + tier1_memories
            logger.info(
                "retrieve | tier1+tier2 merged | tier2_nodes=%d tier1_hits=%d | query=%r",
                len(tier2_result.hits), len(tier1_memories), query[:60],
            )
            return RetrievalResult(
                hits=merged_hits, source="tier1+tier2", confidence=best_score
            )
        return tier2_result

    # Tier 2 found nothing — fall back to whatever tier1 had
    if tier1_memories:
        logger.info("retrieve | tier1 fallback (tier2 miss) | score=%.3f | query=%r", best_score, query[:60])
        return RetrievalResult(hits=tier1_memories, source="tier1", confidence=best_score)

    # --- Tier 2 fallback: file store grep ---
    file_text = grep_filestore(query)
    if file_text:
        reconsolidated = False
        try:
            store_fact_direct(file_text, {"source": "filestore_reconsolidation", "query": query[:200]})
            reconsolidated = True
            logger.info("retrieve | tier2 file hit + reconsolidated | query=%r", query[:60])
        except Exception as exc:
            logger.warning("retrieve | reconsolidation failed: %s", exc)
        return RetrievalResult(
            hits=[file_text], source="tier2", confidence=0.0, reconsolidated=reconsolidated
        )

    logger.info("retrieve | miss | query=%r", query[:60])
    return RetrievalResult(hits=[], source="miss", confidence=0.0)


def _tier2_graph(query: str) -> RetrievalResult | None:
    """Attempt Tier 2 via Kuzu graph propagation from multiple start nodes."""
    try:
        from src.storage.graph import is_initialized
        from src.retrieval.propagator import (
            PropagationResult,
            find_start_nodes,
            format_propagation_result,
            propagate,
        )

        if not is_initialized():
            return None

        starts = find_start_nodes(query)
        if not starts:
            return None

        # Propagate from each anchor; keep the highest prop_score per node_id
        seen: dict[str, object] = {}
        for start in starts:
            prop_result = propagate(
                query,
                start["node_id"],
                start["node_type"],
                start["name"],
                start["content"],
                start_is_stub=start.get("is_stub", False),
            )
            for node in prop_result.nodes:
                if node.node_id not in seen or node.prop_score > seen[node.node_id].prop_score:
                    seen[node.node_id] = node

        if not seen:
            return None

        merged = sorted(seen.values(), key=lambda n: n.prop_score, reverse=True)
        combined = PropagationResult(
            nodes=merged,
            start_entity=", ".join(s["name"] for s in starts),
            max_hop=max(n.hop for n in merged),
        )

        context = format_propagation_result(combined)
        top_score = merged[0].prop_score

        try:
            top_node_ids = [n.node_id for n in merged[:10]]
            query_terms = " ".join(query.lower().split()[:12])
            store_pathway(
                query_terms,
                top_node_ids,
                {"source": "graph_reconsolidation", "query": query[:200], "type": "reconsolidation"},
            )
        except Exception:
            pass

        logger.info(
            "retrieve | tier2 graph hit | starts=%d nodes=%d | query=%r",
            len(starts), len(merged), query[:60],
        )
        return RetrievalResult(
            hits=[context], source="tier2", confidence=top_score, reconsolidated=True
        )
    except Exception as exc:
        logger.warning("retrieve | tier2 graph failed: %s", exc)
        return None


def format_context(result: RetrievalResult) -> str:
    """Format a RetrievalResult for injection into an LLM prompt."""
    if result.source == "miss":
        return "[no stored context — answer from base knowledge]"

    label = f"[{result.source.upper()} | confidence={'%.2f' % result.confidence if result.confidence else 'graph/file'}]"
    body = "\n---\n".join(result.hits)
    return f"{label}\n{body}"
