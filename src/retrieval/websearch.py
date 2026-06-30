import logging
import re

logger = logging.getLogger(__name__)

_FACTUAL_SIGNALS = re.compile(
    r"\b(what is|who is|when did|where is|how many|tell me about|define|what are)\b",
    re.IGNORECASE,
)
_NON_FACTUAL_SIGNALS = re.compile(
    r"\b(write|imagine|create|invent|pretend|make up|generate|design|compose)\b",
    re.IGNORECASE,
)


def is_factual_query(query: str) -> bool:
    if _NON_FACTUAL_SIGNALS.search(query):
        return False
    return bool(_FACTUAL_SIGNALS.search(query))


def search(query: str, top_k: int = 3) -> list[dict]:
    """Tier 3 web search — stub. Returns empty list until a search backend is wired up.

    To activate: install a web search library (e.g. duckduckgo-search) and replace
    this stub with real search logic. The return format is:
        [{"title": "...", "url": "...", "snippet": "..."}]
    """
    logger.info("websearch | stub called for query=%r — returning empty", query[:60])
    return []


def format_web_results(results: list[dict]) -> str:
    if not results:
        return "[web search returned no results]"
    lines = ["[WEB SEARCH RESULTS]"]
    for i, r in enumerate(results, 1):
        lines.append(f"\n[{i}] {r.get('title', '')}")
        lines.append(f"  {r.get('url', '')}")
        lines.append(f"  {r.get('snippet', '')}")
    return "\n".join(lines)
