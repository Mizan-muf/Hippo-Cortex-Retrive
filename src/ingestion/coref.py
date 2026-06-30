import logging

logger = logging.getLogger(__name__)

# Pronouns that can be replaced with their antecedent
_PRONOUNS = frozenset({
    "he", "him", "his", "she", "her", "hers", "it", "its",
    "they", "them", "their", "theirs", "we", "us", "our", "ours",
    "i", "me", "my", "mine", "you", "your", "yours",
    "this", "that", "these", "those",
})

# Third-person singular pronouns targeted by the heuristic fallback
_THIRD_SINGULAR = frozenset({
    "he", "him", "his", "himself",
    "she", "her", "hers", "herself",
})

_model = None
_model_failed = False  # set True after a permanent load failure to skip retries


# ---------------------------------------------------------------------------
# Transformers 5.x compatibility patch
# ---------------------------------------------------------------------------

def _patch_fcoref_for_transformers5() -> None:
    """Ensure FCorefModel has the instance attrs transformers 5.x expects.

    In transformers 5.x, `all_tied_weights_keys` is assigned as an instance
    attr inside PreTrainedModel.__init__.  FCorefModel.__init__ mutates
    base_model_prefix at the *class* level after super().__init__() returns,
    which confuses the post-init tie_weights() call that accesses the attr.
    Adding it at the class level acts as a safe fallback.
    """
    try:
        from fastcoref.coref_models.modeling_fcoref import FCorefModel
        if not hasattr(FCorefModel, "all_tied_weights_keys"):
            FCorefModel.all_tied_weights_keys = {}
        if not hasattr(FCorefModel, "get_expanded_tied_weights_keys"):
            FCorefModel.get_expanded_tied_weights_keys = lambda self, **kw: {}
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Neural model loader
# ---------------------------------------------------------------------------

def _get_model():
    global _model, _model_failed
    if _model_failed:
        return None
    if _model is None:
        try:
            from fastcoref import FCoref
            _patch_fcoref_for_transformers5()
            logger.info("coref | loading FCoref model (downloads on first use)")
            _model = FCoref(device="cpu")
            logger.info("coref | FCoref model ready")
        except Exception as exc:
            logger.warning("coref | FCoref unavailable (%s) — heuristic fallback active", exc)
            _model_failed = True
    return _model


# ---------------------------------------------------------------------------
# Heuristic fallback
# ---------------------------------------------------------------------------

def _heuristic_coref(text: str) -> str:
    """Pronoun substitution via spaCy entity tracking.

    Tracks the most recently mentioned PERSON entity across sentences and
    replaces third-person singular pronouns (he/him/his, she/her/hers) with
    that name.  Works well for single-protagonist narratives where the
    protagonist is introduced by name and then referred to as 'he' or 'she'.
    """
    from src.core.nlp import get_nlp
    nlp = get_nlp()
    doc = nlp(text)

    last_person: str | None = None
    replacements: list[tuple[int, int, str]] = []  # (start, end, replacement)

    for sent in doc.sents:
        # First pass: update the known person from named entities in this sentence.
        # Doing this before the pronoun scan handles "Chen Feng smiled as he left."
        for ent in sent.ents:
            if ent.label_ == "PERSON":
                last_person = ent.text.strip()

        # Second pass: replace pronouns if we have a known antecedent
        if last_person:
            for token in sent:
                if token.text.lower() in _THIRD_SINGULAR and token.ent_type_ == "":
                    replacements.append(
                        (token.idx, token.idx + len(token.text), last_person)
                    )

    if not replacements:
        return text

    # Apply right-to-left so earlier character offsets stay valid
    result = text
    for start, end, rep in sorted(replacements, key=lambda x: -x[0]):
        result = result[:start] + rep + result[end:]

    logger.debug("coref | heuristic resolved %d pronouns", len(replacements))
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def resolve(text: str) -> str:
    """Stage 1b: rewrite pronouns with their full-name antecedents.

    Tries fastcoref (neural, globally accurate).  Falls back to the spaCy
    heuristic if the neural model is unavailable or produces no clusters.
    Returns original text if both approaches find nothing to replace.
    """
    model = _get_model()

    if model is not None:
        try:
            preds = model.predict(texts=[text])
            clusters = preds[0].get_clusters(as_strings=False)
        except Exception as exc:
            logger.warning("coref | prediction failed (%s) — using heuristic", exc)
            return _heuristic_coref(text)

        if not clusters:
            return _heuristic_coref(text)

        replacements: dict[tuple[int, int], str] = {}
        for cluster in clusters:
            if not cluster:
                continue
            rep_span = max(cluster, key=lambda s: s[1] - s[0])
            rep_text = text[rep_span[0]:rep_span[1]]
            for span in cluster:
                if span == rep_span:
                    continue
                if text[span[0]:span[1]].lower() in _PRONOUNS:
                    replacements[span] = rep_text

        if not replacements:
            return _heuristic_coref(text)

        result = text
        for (start, end), rep_text in sorted(replacements.items(), key=lambda x: -x[0][0]):
            result = result[:start] + rep_text + result[end:]

        logger.debug("coref | neural resolved %d pronoun mentions", len(replacements))
        return result

    return _heuristic_coref(text)
