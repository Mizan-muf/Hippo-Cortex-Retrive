import re

_STOP = frozenset(
    "the a an and or but of in on at to for with by from is was are were be been "
    "being has have had do does did will would could should may might shall can "
    "this that these those it its he she they we you i s t".split()
)


def _tokenize(text: str) -> list[str]:
    tokens = re.split(r"[^a-z0-9]+", text.lower())
    return [t for t in tokens if t and t not in _STOP and len(t) > 1]


def _dedup_cap(tokens: list[str], cap: int = 20) -> str:
    """Deduplicate tokens (preserving order) and join up to `cap` of them."""
    seen: set[str] = set()
    unique: list[str] = []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            unique.append(t)
        if len(unique) == cap:
            break
    return " ".join(unique)


def entity_key_terms(name: str, entity_type: str, properties: dict) -> str:
    """Build a bag of key tokens for a person/org/location entity.

    Example output: "chen feng person gene warrior protagonist"
    """
    tokens: list[str] = []
    tokens.extend(_tokenize(name))
    tokens.extend(_tokenize(entity_type))
    for v in properties.values():
        tokens.extend(_tokenize(str(v)))
    return _dedup_cap(tokens)


def event_key_terms(event: dict) -> str:
    """Build a bag of key tokens for an event.

    Uses: participants, action, outcome, locations, title.
    Example output: "chen feng mo lei fight win arena defeat"
    """
    tokens: list[str] = []
    for participant in event.get("participants", []):
        tokens.extend(_tokenize(participant))
    tokens.extend(_tokenize(event.get("action", "")))
    tokens.extend(_tokenize(event.get("outcome", "")))
    for loc in event.get("locations", []):
        tokens.extend(_tokenize(loc))
    tokens.extend(_tokenize(event.get("title", "")))
    return _dedup_cap(tokens)


def event_key_terms_variants(event: dict) -> list[tuple[str, str]]:
    """Return multiple (label, key_terms) pairs for an event, one per query angle.

    Covers five angles so queries phrased different ways all resolve to the same pathway:
      who_what     — participants + action   ("who did what?")
      who_result   — participants + outcome  ("what happened to who?")
      where_what   — locations + action      ("what happened where?")
      cause_effect — action + outcome        ("what caused / what was the result?")
      summary      — title + summary tokens  (broad catch-all)

    Any variant that resolves to an empty string is omitted.
    """
    p_tok = [t for p in event.get("participants", []) for t in _tokenize(p)]
    a_tok = _tokenize(event.get("action", ""))
    o_tok = _tokenize(event.get("outcome", ""))
    l_tok = [t for loc in event.get("locations", []) for t in _tokenize(loc)]
    t_tok = _tokenize(event.get("title", ""))
    s_tok = _tokenize(event.get("summary", ""))

    candidates = [
        ("who_what",     p_tok + a_tok),
        ("who_result",   p_tok + o_tok),
        ("where_what",   l_tok + a_tok),
        ("cause_effect", a_tok + o_tok),
        ("summary",      t_tok + s_tok + p_tok),
    ]

    variants: list[tuple[str, str]] = []
    for label, tokens in candidates:
        terms = _dedup_cap(tokens)
        if terms:
            variants.append((label, terms))
    return variants


def entity_key_terms_variants(
    name: str,
    entity_type: str,
    properties: dict,
    aliases: list[str],
) -> list[tuple[str, str]]:
    """Return multiple (label, key_terms) pairs for an entity.

    Covers:
      name_type  — name + type             (identity / "what is X?" queries)
      name_props — name + property values  (attribute / "what are X's stats?" queries)
      alias_N    — one per alias + type    (alternate-name queries)

    Any variant that resolves to an empty string is omitted.
    """
    n_tok = _tokenize(name)
    type_tok = _tokenize(entity_type)
    prop_tok = [t for v in properties.values() for t in _tokenize(str(v))]

    candidates: list[tuple[str, list[str]]] = [
        ("name_type",  n_tok + type_tok),
        ("name_props", n_tok + prop_tok),
    ]
    for i, alias in enumerate(aliases or []):
        a_tok = _tokenize(alias)
        if a_tok:
            candidates.append((f"alias_{i}", a_tok + type_tok))

    variants: list[tuple[str, str]] = []
    for label, tokens in candidates:
        terms = _dedup_cap(tokens)
        if terms:
            variants.append((label, terms))
    return variants
