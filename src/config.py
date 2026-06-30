import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# LLM provider: "local" (Ollama) or "gemini" (Google Gemini API)
LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "local").lower()
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")

OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LIGHTWEIGHT_LLM: str = os.getenv("LIGHTWEIGHT_LLM", "llama3.2")
QUALITY_LLM: str = os.getenv("QUALITY_LLM", "") or os.getenv("LIGHTWEIGHT_LLM", "llama3.2")
MEM0_LLM: str = os.getenv("MEM0_LLM", "phi3.5")
EMBED_MODEL: str = os.getenv("EMBED_MODEL", "nomic-embed-text")

QDRANT_HOST: str = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT: int = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_COLLECTION: str = os.getenv("QDRANT_COLLECTION", "knowledge_base")
QDRANT_VECTOR_SIZE: int = int(os.getenv("QDRANT_VECTOR_SIZE", "768"))

FILE_STORE_PATH: Path = Path(os.getenv("FILE_STORE_PATH", "./knowledge-store"))
GRAPH_STORE_PATH: Path = Path(os.getenv("GRAPH_STORE_PATH", "./data/graph"))

# Retrieval thresholds
# Tier 1 floor — results below this score are ignored entirely
START_THRESHOLD: float = float(os.getenv("START_THRESHOLD", "0.55"))
# Score above which Tier 1 short-circuits without escalating to the graph
CONFIDENT_THRESHOLD: float = float(os.getenv("CONFIDENT_THRESHOLD", "0.92"))
# Minimum Mem0 score for a result to be used as a Kuzu graph anchor
GRAPH_ANCHOR_THRESHOLD: float = float(os.getenv("GRAPH_ANCHOR_THRESHOLD", "0.35"))
# Max number of graph start nodes to propagate from simultaneously
GRAPH_ANCHOR_MAX_STARTS: int = int(os.getenv("GRAPH_ANCHOR_MAX_STARTS", "3"))
PROPAGATION_DECAY: float = float(os.getenv("PROPAGATION_DECAY", "0.90"))
PROPAGATION_HOP_THRESHOLD: float = float(os.getenv("PROPAGATION_HOP_THRESHOLD", "0.25"))
PROPAGATION_MAX_DEPTH: int = int(os.getenv("PROPAGATION_MAX_DEPTH", "4"))
PROPAGATION_MAX_NODES: int = int(os.getenv("PROPAGATION_MAX_NODES", "25"))
STUB_PENALTY: float = float(os.getenv("STUB_PENALTY", "0.50"))
# Minimum Qdrant score for graph_rag.py seed nodes
GRAPHRAG_SEED_THRESHOLD: float = float(os.getenv("GRAPHRAG_SEED_THRESHOLD", "0.35"))

# Entity linker thresholds
LINK_THRESHOLD: float = float(os.getenv("LINK_THRESHOLD", "0.88"))
# 0.82: high enough to exclude cross-entity co-occurrence noise from nomic-embed-text
# (org/location names were falsely matching person entries at 0.70-0.74)
LINK_AMBIGUOUS_MIN: float = float(os.getenv("LINK_AMBIGUOUS_MIN", "0.82"))

# Pipeline settings
SEGMENT_MODE: str = os.getenv("SEGMENT_MODE", "paragraph")
SEMANTIC_SPLIT_THRESHOLD: float = float(os.getenv("SEMANTIC_SPLIT_THRESHOLD", "0.75"))
MIN_SEGMENT_CHARS: int = int(os.getenv("MIN_SEGMENT_CHARS", "150"))
WEB_SEARCH_ENABLED: bool = os.getenv("WEB_SEARCH_ENABLED", "true").lower() == "true"
HITL_MODE: str = os.getenv("HITL_MODE", "async")
