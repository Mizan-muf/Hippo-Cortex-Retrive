import hashlib
import logging
import uuid
from datetime import datetime, timezone

from mem0 import Memory
from qdrant_client import QdrantClient, models as qdrant_models
from qdrant_client.models import Distance, PointStruct, SparseVectorParams, VectorParams

from src.config import (
    EMBED_MODEL,
    MEM0_LLM,
    OLLAMA_BASE_URL,
    QDRANT_COLLECTION,
    QDRANT_HOST,
    QDRANT_PORT,
    QDRANT_VECTOR_SIZE,
)

PATHWAY_COLLECTION = "pathway_cache"
from src.core.llm import call as llm_call

logger = logging.getLogger(__name__)

_mem0: Memory | None = None
_qc: QdrantClient | None = None


def _get_qc() -> QdrantClient:
    global _qc
    if _qc is None:
        _qc = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=30)
    return _qc


def _sparse_config() -> dict:
    # modifier=IDF matches exactly what Mem0 v3 creates — without this,
    # Mem0's _has_bm25_slot check still passes but BM25 encoding behaviour differs.
    return {"bm25": SparseVectorParams(modifier=qdrant_models.Modifier.IDF)}


def _has_bm25(qc: QdrantClient, name: str) -> bool:
    try:
        info = qc.get_collection(name)
        sparse = info.config.params.sparse_vectors or {}
        return "bm25" in sparse
    except Exception:
        return False


def _ensure_collection(qc: QdrantClient, name: str) -> None:
    """Create or migrate a collection to include the BM25 sparse slot."""
    if not qc.collection_exists(name):
        qc.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=QDRANT_VECTOR_SIZE, distance=Distance.COSINE),
            sparse_vectors_config=_sparse_config(),
        )
        logger.info("memory | created collection '%s' with BM25 hybrid support", name)
        return

    info = qc.get_collection(name)
    # Check dense dimension
    actual = info.config.params.vectors.size
    if actual != QDRANT_VECTOR_SIZE:
        logger.critical(
            "Qdrant collection '%s' has %d dims but expected %d — "
            "delete it with: curl -X DELETE http://%s:%d/collections/%s",
            name, actual, QDRANT_VECTOR_SIZE,
            QDRANT_HOST, QDRANT_PORT, name,
        )
        raise RuntimeError(f"dimension mismatch in collection '{name}'")

    # Migrate: if BM25 slot missing, recreate the collection
    if not _has_bm25(qc, name):
        logger.warning(
            "memory | collection '%s' has no BM25 slot — recreating for hybrid search "
            "(existing vectors will be lost; re-run ingest to repopulate)",
            name,
        )
        qc.delete_collection(name)
        qc.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=QDRANT_VECTOR_SIZE, distance=Distance.COSINE),
            sparse_vectors_config=_sparse_config(),
        )
        logger.info("memory | collection '%s' recreated with BM25 hybrid support", name)


def _ensure_pathway_collection(qc: QdrantClient) -> None:
    """Create the pathway_cache collection if it doesn't exist.

    pathway_cache is a dense-only collection (no BM25 needed — pathway points
    are never hybrid-searched). Kept separate from knowledge_base so that
    key-term bags don't pollute Mem0 memory search results.
    """
    if not qc.collection_exists(PATHWAY_COLLECTION):
        qc.create_collection(
            collection_name=PATHWAY_COLLECTION,
            vectors_config=VectorParams(size=QDRANT_VECTOR_SIZE, distance=Distance.COSINE),
        )
        logger.info("memory | created collection '%s'", PATHWAY_COLLECTION)


def init_pathway_cache() -> bool:
    """Ensure the pathway_cache collection exists. Returns True on success."""
    try:
        _ensure_pathway_collection(_get_qc())
        return True
    except Exception as exc:
        logger.critical("memory | pathway_cache init failed: %s", exc)
        return False


def init_memory() -> bool:
    """Initialize Qdrant collections and Mem0. Returns True on success."""
    global _mem0, _qc
    if _mem0 is not None:
        return True
    try:
        qc = _get_qc()

        _ensure_collection(qc, QDRANT_COLLECTION)

        # Pre-create the Mem0 _entities collection to prevent first-write dimension mismatch (F3)
        entities_col = f"{QDRANT_COLLECTION}_entities"
        _ensure_collection(qc, entities_col)

        _ensure_pathway_collection(qc)

        config = {
            "llm": {
                "provider": "ollama",
                "config": {
                    "model": MEM0_LLM,
                    "ollama_base_url": OLLAMA_BASE_URL,
                },
            },
            "embedder": {
                "provider": "ollama",
                "config": {
                    "model": EMBED_MODEL,
                    "ollama_base_url": OLLAMA_BASE_URL,
                },
            },
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "host": QDRANT_HOST,
                    "port": QDRANT_PORT,
                    "collection_name": QDRANT_COLLECTION,
                },
            },
        }
        _mem0 = Memory.from_config(config)
        logger.info("memory | initialized — collection=%s dims=%d", QDRANT_COLLECTION, QDRANT_VECTOR_SIZE)
        return True

    except Exception as exc:
        logger.critical("memory | Qdrant unreachable or Mem0 init failed: %s", exc)
        return False


def _require_init() -> None:
    if _mem0 is None:
        raise RuntimeError("Memory not initialized — call init_memory() first")


def store_fact(content: str, metadata: dict | None = None) -> None:
    _require_init()
    _mem0.add(  # type: ignore[union-attr]
        [{"role": "user", "content": content}],
        user_id="app_user",
        metadata=metadata or {},
    )
    logger.debug("memory | stored fact: %s", content[:80])


def search_knowledge(query: str, top_k: int = 3) -> list[dict]:
    """Direct Qdrant vector search over knowledge_base — bypasses Mem0's internal LLM.

    Returns only Mem0/fact memories. Pathway routing points live in pathway_cache
    and are never mixed here — use search_pathways() for those.
    """
    _require_init()
    from src.core.embed import embed

    vec = embed(query)
    response = _get_qc().query_points(
        collection_name=QDRANT_COLLECTION,
        query=vec,
        limit=top_k,
        with_payload=True,
        query_filter=qdrant_models.Filter(
            must=[qdrant_models.FieldCondition(
                key="user_id",
                match=qdrant_models.MatchValue(value="app_user"),
            )]
        ),
    )
    # store_fact_direct writes "data"; mem0.add writes "memory" — handle both
    results = [
        {
            "memory": r.payload.get("data") or r.payload.get("memory", ""),
            "score": r.score,
            "metadata": r.payload.get("metadata", {}),
            "pathway": r.payload.get("metadata", {}).get("pathway", []),
        }
        for r in response.points
    ]
    logger.debug("memory | search returned %d results for %r", len(results), query[:60])
    return results


def search_pathways(query: str, top_k: int = 5) -> list[dict]:
    """Search pathway_cache for pathway signatures matching the query.

    Returns list of dicts with: pathway (list of Kuzu node IDs), score, metadata.
    Used exclusively by _tier1_pathway() in chain.py.
    """
    from src.core.embed import embed

    qc = _get_qc()
    if not qc.collection_exists(PATHWAY_COLLECTION):
        return []

    vec = embed(query)
    response = qc.query_points(
        collection_name=PATHWAY_COLLECTION,
        query=vec,
        limit=top_k,
        with_payload=True,
    )
    results = [
        {
            "pathway": r.payload.get("metadata", {}).get("pathway", []),
            "score": r.score,
            "metadata": r.payload.get("metadata", {}),
            "key_terms": r.payload.get("data", ""),
        }
        for r in response.points
        if r.payload.get("metadata", {}).get("pathway")
    ]
    logger.debug("memory | pathway search returned %d results for %r", len(results), query[:60])
    return results


def check_conflicts(new_input: str) -> str | None:
    """Stage 0: check whether new_input contradicts stored knowledge.

    Returns a one-sentence conflict description, or None if no conflict found.
    """
    _require_init()
    results = search_knowledge(new_input, top_k=5)
    snippets = [r.get("memory", "") for r in results if r.get("memory")]
    if not snippets:
        return None

    stored_facts = "\n".join(f"- {s}" for s in snippets)
    prompt = (
        f"Stored knowledge:\n{stored_facts}\n\n"
        f"New input:\n{new_input[:800]}\n\n"
        "Does the new input directly and clearly contradict any stored fact above "
        "(e.g. a changed attribute, a reversed state, a mutually exclusive claim)? "
        "Reply NONE if there is no contradiction, or describe the specific conflict "
        "in one sentence."
    )
    try:
        answer = llm_call(prompt, temperature=0.0).strip()
        if answer.upper() == "NONE" or not answer:
            return None
        return answer
    except Exception as exc:
        logger.warning("memory | conflict check LLM call failed: %s", exc)
        return None


def store_fact_direct(content: str, metadata: dict | None = None) -> None:
    """Embed and upsert directly to Qdrant, bypassing Mem0's internal LLM pipeline.

    Use this for bulk seeding. Writes Mem0-compatible payloads so search_knowledge()
    finds these vectors normally. Avoids the LLM consolidation calls that Mem0's
    store_fact() triggers — which cause JSON parse errors with some local models.
    """
    from src.core.embed import embed

    vec = embed(content)
    now = datetime.now(timezone.utc).isoformat()
    content_hash = hashlib.md5(content.encode()).hexdigest()
    point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"app_user:{content_hash}"))

    payload = {
        "user_id": "app_user",
        "hash": content_hash,
        "data": content,
        "created_at": now,
        "updated_at": now,
        "metadata": metadata or {},
    }

    _get_qc().upsert(
        collection_name=QDRANT_COLLECTION,
        points=[PointStruct(id=point_id, vector=vec, payload=payload)],
    )
    logger.debug("memory | direct store: %s", content[:80])


def store_pathway(
    key_terms: str,
    pathway: list[str],
    metadata: dict | None = None,
) -> None:
    """Embed key_terms and store pathway (list of Kuzu node IDs) in Qdrant.

    key_terms  — space-separated bag of salient tokens; this is what gets embedded.
    pathway    — ordered list of Kuzu node IDs activated by this pattern.
    metadata   — standard metadata dict; pathway list is merged in automatically.
    """
    from src.core.embed import embed

    vec = embed(key_terms)
    now = datetime.now(timezone.utc).isoformat()
    point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"pathway:{key_terms[:80]}"))
    merged_meta = {**(metadata or {}), "pathway": pathway}

    payload = {
        "user_id": "app_user",
        "hash": point_id,
        "data": key_terms,
        "created_at": now,
        "updated_at": now,
        "metadata": merged_meta,
    }

    _get_qc().upsert(
        collection_name=PATHWAY_COLLECTION,
        points=[PointStruct(id=point_id, vector=vec, payload=payload)],
    )
    logger.debug("memory | pathway stored: %s → %d nodes", key_terms[:60], len(pathway))
