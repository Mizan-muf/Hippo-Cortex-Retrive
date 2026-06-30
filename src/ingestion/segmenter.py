import logging
import math
import re

from src.config import MIN_SEGMENT_CHARS, SEGMENT_MODE, SEMANTIC_SPLIT_THRESHOLD
from src.core.llm import call as llm_call

logger = logging.getLogger(__name__)

SCENE_BREAK_RE = re.compile(r"\n\s*(\*{3,}|---+|#{1,3}\s+\w)", re.MULTILINE)


def segment(text: str, mode: str | None = None) -> list[dict]:
    """Split text into segments. Returns list of {segment_id, text}."""
    mode = mode or SEGMENT_MODE
    if mode == "semantic":
        return _by_semantic(text)
    elif mode == "paragraph":
        return _by_paragraph(text)
    elif mode == "scene_break":
        return _by_scene_break(text)
    elif mode == "llm":
        return _by_llm(text)
    else:
        logger.warning("segmenter | unknown mode '%s', falling back to paragraph", mode)
        return _by_paragraph(text)


def _by_semantic(text: str) -> list[dict]:
    """Split by cosine similarity drops between adjacent sentence embeddings."""
    from src.core.embed import embed as embed_text
    from src.core.nlp import get_nlp

    nlp = get_nlp()
    doc = nlp(text)
    sentences = [s.text.strip() for s in doc.sents if s.text.strip()]

    if not sentences:
        return _by_paragraph(text)

    vecs = [embed_text(s) for s in sentences]

    def cosine(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(x * x for x in b))
        return dot / (na * nb) if na and nb else 0.0

    chunks: list[str] = []
    current = [sentences[0]]
    for i in range(1, len(sentences)):
        if cosine(vecs[i - 1], vecs[i]) < SEMANTIC_SPLIT_THRESHOLD:
            chunks.append(" ".join(current))
            current = [sentences[i]]
        else:
            current.append(sentences[i])
    chunks.append(" ".join(current))

    # Merge micro-segments shorter than MIN_SEGMENT_CHARS into the preceding chunk
    merged: list[str] = []
    for chunk in chunks:
        if merged and len(chunk) < MIN_SEGMENT_CHARS:
            merged[-1] += " " + chunk
        else:
            merged.append(chunk)

    return [{"segment_id": i, "text": s} for i, s in enumerate(merged) if s.strip()]


def _by_paragraph(text: str) -> list[dict]:
    parts = [p.strip() for p in re.split(r"\n\s*\n", text) if len(p.strip()) >= 80]
    return [{"segment_id": i, "text": p} for i, p in enumerate(parts)]


def _by_scene_break(text: str) -> list[dict]:
    parts = [p.strip() for p in SCENE_BREAK_RE.split(text) if p.strip()]
    # filter out bare break markers captured by split groups
    segments = [p for p in parts if len(p) > 20]
    if not segments:
        return _by_paragraph(text)
    return [{"segment_id": i, "text": s} for i, s in enumerate(segments)]


def _by_llm(text: str) -> list[dict]:
    prompt = (
        "Split the following text into coherent scenes or passages. "
        "Each scene should be a complete narrative unit. "
        "Return ONLY the scenes separated by the marker: <<<SCENE_BREAK>>>. "
        "Do not add any other text.\n\n"
        f"{text[:6000]}"
    )
    try:
        raw = llm_call(prompt, temperature=0.1)
        parts = [p.strip() for p in raw.split("<<<SCENE_BREAK>>>") if p.strip()]
        if parts:
            return [{"segment_id": i, "text": p} for i, p in enumerate(parts)]
    except Exception as exc:
        logger.warning("segmenter | llm split failed: %s — falling back to semantic", exc)
    return _by_semantic(text)
