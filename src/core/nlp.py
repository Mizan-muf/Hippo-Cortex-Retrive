import logging

import spacy

logger = logging.getLogger(__name__)

_nlp = None


def get_nlp():
    """Return the shared en_core_web_trf model (lazy-loaded, ~500MB on first call)."""
    global _nlp
    if _nlp is None:
        logger.info("nlp | loading en_core_web_trf")
        _nlp = spacy.load("en_core_web_trf")
        logger.info("nlp | model ready")
    return _nlp
