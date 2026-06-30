"""
Standard RAG retrieval — sliding-window chunks stored in Qdrant.
Collection: tsg_rag
"""
import logging

from qdrant_client import QdrantClient

from src.config import QDRANT_HOST, QDRANT_PORT
from src.core.embed import embed

logger = logging.getLogger(__name__)

RAG_COLLECTION = "tsg_rag"

_qc: QdrantClient | None = None


def _get_qc() -> QdrantClient:
    global _qc
    if _qc is None:
        _qc = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=30)
    return _qc


def retrieve_rag(query: str, top_k: int = 5) -> list[dict]:
    """Semantic search over stored chunks. Returns list of {text, score, chapter}."""
    qc = _get_qc()
    vec = embed(query)
    response = qc.query_points(collection_name=RAG_COLLECTION, query=vec, limit=top_k)
    hits = [
        {
            "text": r.payload.get("text", ""),
            "score": round(r.score, 4),
            "chapter": r.payload.get("chapter"),
            "chunk_idx": r.payload.get("chunk_idx"),
        }
        for r in response.points
    ]
    logger.info("rag | query=%r  hits=%d  top_score=%.3f", query[:60], len(hits), hits[0]["score"] if hits else 0)
    return hits


def format_rag_context(hits: list[dict]) -> str:
    if not hits:
        return "[RAG] no results"
    lines = ["[STANDARD RAG]"]
    for i, h in enumerate(hits, 1):
        lines.append(f"\n[{i} | ch.{h['chapter']} | score {h['score']:.3f}]")
        lines.append(h["text"])
    return "\n".join(lines)
