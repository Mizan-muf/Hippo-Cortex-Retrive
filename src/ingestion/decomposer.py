import logging
import re
from datetime import datetime, timezone

from src.core.nlp import get_nlp

logger = logging.getLogger(__name__)

# Entity label sets for NER-based extraction
_PARTICIPANT_LABELS = {"PERSON", "ORG", "NORP"}
_LOCATION_LABELS = {"GPE", "LOC", "FAC"}

# Pronouns, articles, and generic nouns that spaCy (especially sm/md) mis-tags as entities
_ENTITY_NOISE = {
    "i", "me", "my", "mine", "myself",
    "you", "your", "yours", "yourself",
    "he", "him", "his", "himself",
    "she", "her", "hers", "herself",
    "it", "its", "itself",
    "we", "us", "our", "ours", "ourselves",
    "they", "them", "their", "theirs", "themselves",
    "who", "whom", "whose", "which", "that",
    "this", "these", "those",
    "one", "ones", "thing", "things", "everyone", "someone", "anyone", "nobody",
    "man", "men", "woman", "women", "person", "people", "guy", "guys",
    "all", "both", "each", "other", "another", "others",
}


def _is_noise(text: str) -> bool:
    return text.lower().strip() in _ENTITY_NOISE or len(text.strip()) <= 1


_LEADING_DET = re.compile(r"^(all|the|a|an|this|these|those|some|any)\s+", re.IGNORECASE)


def _normalize_entity_text(text: str) -> str:
    """Strip possessives, leading determiners, and trailing punctuation from NER span text."""
    text = re.sub(r"[''']s\b", "", text).strip()
    text = _LEADING_DET.sub("", text)
    return text.rstrip("!?,. ").strip()


def _sent_entities(sent, labels: set[str]) -> dict[str, str]:
    """Named entities in the sentence, filtered and normalized.

    Returns {normalized_name: ner_label}. Possessives are stripped so
    'Chen Feng's' becomes 'Chen Feng' and de-dupes with any direct mention.
    """
    result: dict[str, str] = {}
    for e in sent.ents:
        if e.label_ in labels:
            name = _normalize_entity_text(e.text)
            if name and not _is_noise(name):
                result[name] = e.label_
    return result


def extract_events(text: str, doc_id: int, seg_id: int) -> list[dict]:
    """Stage 2: extract Event objects from a text segment using spaCy NER + dep parsing."""
    nlp = get_nlp()
    doc = nlp(text)
    now = datetime.now(timezone.utc).isoformat()

    # Segment-level entities — only used for the no-verb fallback event
    seg_part_types: dict[str, str] = {}
    seg_loc_types: dict[str, str] = {}
    for e in doc.ents:
        name = _normalize_entity_text(e.text)
        if not name or _is_noise(name):
            continue
        if e.label_ in _PARTICIPANT_LABELS:
            seg_part_types[name] = e.label_
        elif e.label_ in _LOCATION_LABELS:
            seg_loc_types[name] = e.label_

    sentences = list(doc.sents)

    events: list[dict] = []
    for sent_idx, sent in enumerate(sentences):
        # Scope entities strictly to the sentence — never fall back to segment-wide list,
        # since that contaminates every pronoun-subject event with all characters in the passage.
        sent_part_types = _sent_entities(sent, _PARTICIPANT_LABELS)
        sent_loc_types = _sent_entities(sent, _LOCATION_LABELS)

        # 5-sentence context window: 2 before, the sentence itself, 2 after
        ctx_start = max(0, sent_idx - 2)
        ctx_end = min(len(sentences), sent_idx + 3)
        context_text = " ".join(s.text.strip() for s in sentences[ctx_start:ctx_end])

        for token in sent:
            if token.dep_ != "ROOT" or token.pos_ != "VERB":
                continue
            subjs = [c.text for c in token.children if c.dep_ in ("nsubj", "nsubjpass")]
            objs = [c.text for c in token.children if c.dep_ in ("dobj", "attr", "pobj")]
            if not subjs:
                continue

            # Prefer a named entity over a pronoun/generic noun for the title subject
            title_subj = subjs[0]
            if _is_noise(title_subj) and sent_part_types:
                title_subj = next(iter(sent_part_types))

            events.append({
                "title": f"{title_subj} {token.lemma_}"[:80],
                "action": token.lemma_,
                "outcome": " ".join(objs),
                "source_text": context_text,
                "summary": sent.text.strip(),
                "participants": list(sent_part_types.keys()),
                "participant_types": sent_part_types,
                "locations": list(sent_loc_types.keys()),
                "caused_by": [],
                "leads_to": [],
                "document_id": doc_id,
                "segment_id": seg_id,
                "timestamp": now,
            })
            logger.info(
                "decomposer | EVENT | doc%02d seg%02d | %s",
                doc_id, seg_id, events[-1]["title"],
            )

    # Fallback: no root verb found — emit one event covering the whole segment
    if not events:
        events.append({
            "title": text[:80],
            "action": "",
            "outcome": "",
            "source_text": text,
            "summary": text,
            "participants": list(seg_part_types.keys()),
            "participant_types": seg_part_types,
            "locations": list(seg_loc_types.keys()),
            "caused_by": [],
            "leads_to": [],
            "document_id": doc_id,
            "segment_id": seg_id,
            "timestamp": now,
        })

    logger.info("decomposer | extracted %d events from doc%02d seg%02d", len(events), doc_id, seg_id)
    return events


def extract_properties(text: str, doc_id: int, seg_id: int) -> list[dict]:
    """Stage 3: extract static entity properties using spaCy dependency patterns."""
    nlp = get_nlp()
    doc = nlp(text)
    props: list[dict] = []

    for sent in doc.sents:
        for token in sent:
            # 1. Attributive: "X is a Y"  — nsubj → copula head → attr
            if token.dep_ == "attr":
                for subj in (c for c in token.head.children if c.dep_ == "nsubj"):
                    # Only emit for tokens that are part of a named entity, not pronouns/common nouns
                    if _is_noise(subj.text) or subj.ent_type_ == "":
                        continue
                    props.append({
                        "entity": _normalize_entity_text(subj.text), "attribute": "type", "value": token.text,
                        "document_id": doc_id, "segment_id": seg_id,
                    })

            # 2. Adjectival: "X is corroded"  — nsubj + amod modifier
            elif token.dep_ == "amod":
                for subj in (c for c in token.head.children if c.dep_ == "nsubj"):
                    if _is_noise(subj.text) or subj.ent_type_ == "":
                        continue
                    props.append({
                        "entity": _normalize_entity_text(subj.text), "attribute": "condition", "value": token.text,
                        "document_id": doc_id, "segment_id": seg_id,
                    })

            # 3. Prepositional: "X is from Y"  — nsubj + prep + pobj
            elif token.dep_ == "prep" and token.head.dep_ in ("ROOT", "relcl"):
                pobjs = [c.text for c in token.children if c.dep_ == "pobj"]
                for subj in (c for c in token.head.children if c.dep_ == "nsubj"):
                    if _is_noise(subj.text) or subj.ent_type_ == "":
                        continue
                    for pobj in pobjs:
                        props.append({
                            "entity": _normalize_entity_text(subj.text), "attribute": token.text, "value": pobj,
                            "document_id": doc_id, "segment_id": seg_id,
                        })

            # 4. Appositive: "X, a Y"  — appos
            elif token.dep_ == "appos":
                head = token.head
                if _is_noise(head.text) or head.ent_type_ == "":
                    continue
                props.append({
                    "entity": _normalize_entity_text(head.text), "attribute": "role", "value": token.text,
                    "document_id": doc_id, "segment_id": seg_id,
                })

    for p in props:
        logger.info(
            "decomposer | PROP  | doc%02d seg%02d | %s.%s = %s",
            doc_id, seg_id, p["entity"], p["attribute"], str(p["value"])[:40],
        )

    logger.info("decomposer | extracted %d properties from doc%02d seg%02d", len(props), doc_id, seg_id)
    return props
