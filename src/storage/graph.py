import atexit
import json
import logging
import uuid
from datetime import datetime, timezone

import kuzu

from src.config import GRAPH_STORE_PATH

logger = logging.getLogger(__name__)

_db: kuzu.Database | None = None
_conn: kuzu.Connection | None = None


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _gen_id(prefix: str, name: str) -> str:
    slug = "".join(c if c.isalnum() else "_" for c in name.lower())[:20]
    uid = str(uuid.uuid4())[:8]
    return f"{prefix}_{slug}_{uid}"


def close_graph() -> None:
    """Release Kuzu connection and database, dropping the file lock."""
    global _db, _conn
    _conn = None
    _db = None
    logger.info("graph | closed")


def init_graph() -> bool:
    global _db, _conn
    if _conn is not None:
        return True
    try:
        # Ensure the parent exists but let Kuzu manage the database path itself.
        # If an empty directory was left from a previous failed init, remove it
        # so Kuzu can create its own structure there.
        GRAPH_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
        if GRAPH_STORE_PATH.is_dir() and not any(GRAPH_STORE_PATH.iterdir()):
            GRAPH_STORE_PATH.rmdir()
        _db = kuzu.Database(str(GRAPH_STORE_PATH))
        _conn = kuzu.Connection(_db)
        _create_schema()
        atexit.register(close_graph)
        logger.info("graph | initialized at %s", GRAPH_STORE_PATH)
        return True
    except Exception as exc:
        logger.critical("graph | init failed: %s", exc)
        return False


def is_initialized() -> bool:
    return _conn is not None


def _conn_required() -> kuzu.Connection:
    if _conn is None:
        raise RuntimeError("Graph not initialized — call init_graph() first")
    return _conn


def _exec(query: str, params: dict | None = None) -> kuzu.QueryResult:
    return _conn_required().execute(query, parameters=params or {})


def _create_schema() -> None:
    stmts = [
        """CREATE NODE TABLE IF NOT EXISTS Entity(
            id STRING,
            name STRING,
            type STRING,
            aliases STRING[],
            is_stub BOOLEAN,
            confidence DOUBLE,
            source_count INT64,
            first_seen STRING,
            last_updated STRING,
            properties STRING,
            PRIMARY KEY(id)
        )""",
        """CREATE NODE TABLE IF NOT EXISTS Event(
            id STRING,
            title STRING,
            summary STRING,
            source_text STRING,
            participants STRING[],
            locations STRING[],
            action STRING,
            outcome STRING,
            caused_by STRING[],
            leads_to STRING[],
            temporal_order INT64,
            document_id INT64,
            segment_id INT64,
            timestamp STRING,
            PRIMARY KEY(id)
        )""",
        """CREATE NODE TABLE IF NOT EXISTS Document(
            id INT64,
            title STRING,
            date STRING,
            source STRING,
            PRIMARY KEY(id)
        )""",
        "CREATE REL TABLE IF NOT EXISTS PARTICIPANT_IN(FROM Entity TO Event, role STRING)",
        "CREATE REL TABLE IF NOT EXISTS LOCATED_AT(FROM Event TO Entity, specificity INT64)",
        "CREATE REL TABLE IF NOT EXISTS CAUSES(FROM Event TO Event)",
        "CREATE REL TABLE IF NOT EXISTS PRECEDES(FROM Event TO Event)",
        "CREATE REL TABLE IF NOT EXISTS ALIAS_OF(FROM Entity TO Entity)",
        "CREATE REL TABLE IF NOT EXISTS CONTAINED_IN(FROM Event TO Document)",
        "CREATE REL TABLE IF NOT EXISTS MENTIONED_IN(FROM Entity TO Document)",
    ]
    for stmt in stmts:
        try:
            _exec(stmt)
        except Exception as exc:
            if "already exist" not in str(exc).lower():
                logger.warning("graph | schema: %s", exc)


# ---------------------------------------------------------------------------
# Document operations
# ---------------------------------------------------------------------------

def upsert_document(doc_id: int, title: str, *, source: str = "", date: str = "") -> bool:
    """Create a Document node if it does not already exist. Returns True if created."""
    r = _exec("MATCH (d:Document {id: $id}) RETURN d.id", {"id": doc_id})
    if r.has_next():
        return False
    _exec(
        "CREATE (:Document {id: $id, title: $title, date: $date, source: $source})",
        {"id": doc_id, "title": title, "date": date or _now(), "source": source},
    )
    logger.info("graph | document created: %s (id=%d)", title[:60], doc_id)
    return True


# ---------------------------------------------------------------------------
# Entity operations
# ---------------------------------------------------------------------------

def _entity_row(row: list) -> dict:
    return {
        "id": row[0], "name": row[1], "type": row[2],
        "aliases": row[3] or [], "is_stub": bool(row[4]),
        "confidence": float(row[5] or 0.5), "source_count": int(row[6] or 1),
        "properties": json.loads(row[7]) if row[7] else {},
    }


def find_entity_by_name(name: str) -> dict | None:
    r = _exec(
        "MATCH (e:Entity) WHERE e.name = $name "
        "RETURN e.id, e.name, e.type, e.aliases, e.is_stub, e.confidence, e.source_count, e.properties",
        {"name": name},
    )
    return _entity_row(r.get_next()) if r.has_next() else None


def find_entity_by_alias(alias: str) -> dict | None:
    r = _exec(
        "MATCH (e:Entity) WHERE $alias IN e.aliases "
        "RETURN e.id, e.name, e.type, e.aliases, e.is_stub, e.confidence, e.source_count, e.properties",
        {"alias": alias},
    )
    return _entity_row(r.get_next()) if r.has_next() else None


def upsert_entity(
    name: str,
    entity_type: str,
    *,
    properties: dict | None = None,
    aliases: list[str] | None = None,
    is_stub: bool = True,
    confidence: float = 0.4,
    entity_id: str | None = None,
) -> tuple[bool, str]:
    """Upsert entity. Returns (was_created, entity_id)."""
    existing = find_entity_by_name(name)
    if existing:
        eid = existing["id"]
        new_count = existing["source_count"] + 1
        new_stub = existing["is_stub"] and new_count < 2
        new_conf = min(0.97, existing["confidence"] + 0.05) if not new_stub else existing["confidence"]
        _exec(
            "MATCH (e:Entity {id: $id}) SET e.last_updated = $date, "
            "e.source_count = $count, e.is_stub = $stub, e.confidence = $conf",
            {"id": eid, "date": _now(), "count": new_count, "stub": new_stub, "conf": new_conf},
        )
        if properties:
            merged = {**existing.get("properties", {}), **properties}
            _exec(
                "MATCH (e:Entity {id: $id}) SET e.properties = $props",
                {"id": eid, "props": json.dumps(merged)},
            )
        if aliases:
            current = existing.get("aliases", [])
            updated = list(set(current + aliases))
            _exec(
                "MATCH (e:Entity {id: $id}) SET e.aliases = $aliases",
                {"id": eid, "aliases": updated},
            )
        logger.debug("graph | entity updated: %s", name)
        return False, eid

    eid = entity_id or _gen_id("entity", name)
    _exec(
        """CREATE (:Entity {
            id: $id, name: $name, type: $type, aliases: $aliases,
            is_stub: $is_stub, confidence: $confidence, source_count: 1,
            first_seen: $date, last_updated: $date, properties: $properties
        })""",
        {
            "id": eid, "name": name, "type": entity_type,
            "aliases": aliases or [],
            "is_stub": is_stub, "confidence": confidence,
            "date": _now(),
            "properties": json.dumps(properties or {}),
        },
    )
    logger.info("graph | entity created: %s (%s)", name, entity_type)
    return True, eid


def add_alias(entity_id: str, alias: str) -> None:
    r = _exec("MATCH (e:Entity {id: $id}) RETURN e.aliases", {"id": entity_id})
    if r.has_next():
        current = r.get_next()[0] or []
        if alias not in current:
            _exec(
                "MATCH (e:Entity {id: $id}) SET e.aliases = $aliases",
                {"id": entity_id, "aliases": current + [alias]},
            )


# ---------------------------------------------------------------------------
# Event operations
# ---------------------------------------------------------------------------

def _event_row(row: list) -> dict:
    return {
        "id": row[0], "title": row[1], "summary": row[2],
        "source_text": row[3], "participants": row[4] or [],
        "locations": row[5] or [], "temporal_order": int(row[6] or 0),
    }


def find_event_by_id(event_id: str) -> dict | None:
    r = _exec(
        "MATCH (ev:Event {id: $id}) "
        "RETURN ev.id, ev.title, ev.summary, ev.source_text, ev.participants, ev.locations, ev.temporal_order",
        {"id": event_id},
    )
    return _event_row(r.get_next()) if r.has_next() else None


def upsert_event(event: dict) -> tuple[bool, str]:
    """Upsert event. Returns (was_created, event_id)."""
    eid = event.get("id") or _gen_id("evt", event.get("title", "unknown"))
    event["id"] = eid
    if find_event_by_id(eid):
        logger.debug("graph | event exists: %s", eid)
        return False, eid

    _exec(
        """CREATE (:Event {
            id: $id, title: $title, summary: $summary, source_text: $source_text,
            participants: $participants, locations: $locations, action: $action,
            outcome: $outcome, caused_by: $caused_by, leads_to: $leads_to,
            temporal_order: $temporal_order, document_id: $document_id,
            segment_id: $segment_id, timestamp: $timestamp
        })""",
        {
            "id": eid,
            "title": event.get("title", ""),
            "summary": event.get("summary", ""),
            "source_text": event.get("source_text", ""),
            "participants": event.get("participants", []),
            "locations": event.get("locations", []),
            "action": event.get("action", ""),
            "outcome": event.get("outcome", ""),
            "caused_by": event.get("caused_by", []),
            "leads_to": event.get("leads_to", []),
            "temporal_order": int(event.get("temporal_order", 0)),
            "document_id": int(event.get("document_id", 0)),
            "segment_id": int(event.get("segment_id", 0)),
            "timestamp": event.get("timestamp", _now()),
        },
    )
    logger.info("graph | event created: %s", event.get("title", eid))
    return True, eid


# ---------------------------------------------------------------------------
# Edge operations
# ---------------------------------------------------------------------------

def _edge_exists(query: str, params: dict) -> bool:
    r = _exec(query, params)
    return r.has_next() and (r.get_next()[0] or 0) > 0


def add_participant_in(entity_id: str, event_id: str, role: str = "") -> bool:
    if _edge_exists(
        "MATCH (e:Entity {id: $eid})-[:PARTICIPANT_IN]->(ev:Event {id: $evid}) RETURN count(*)",
        {"eid": entity_id, "evid": event_id},
    ):
        return False
    try:
        _exec(
            "MATCH (e:Entity {id: $eid}), (ev:Event {id: $evid}) "
            "CREATE (e)-[:PARTICIPANT_IN {role: $role}]->(ev)",
            {"eid": entity_id, "evid": event_id, "role": role},
        )
        return True
    except Exception as exc:
        logger.warning("graph | add_participant_in: %s", exc)
        return False


def add_located_at(event_id: str, entity_id: str, specificity: int = 0) -> bool:
    if _edge_exists(
        "MATCH (ev:Event {id: $evid})-[:LOCATED_AT]->(e:Entity {id: $eid}) RETURN count(*)",
        {"evid": event_id, "eid": entity_id},
    ):
        return False
    try:
        _exec(
            "MATCH (ev:Event {id: $evid}), (e:Entity {id: $eid}) "
            "CREATE (ev)-[:LOCATED_AT {specificity: $s}]->(e)",
            {"evid": event_id, "eid": entity_id, "s": specificity},
        )
        return True
    except Exception as exc:
        logger.warning("graph | add_located_at: %s", exc)
        return False


def add_precedes(from_event_id: str, to_event_id: str) -> bool:
    if _edge_exists(
        "MATCH (a:Event {id: $a})-[:PRECEDES]->(b:Event {id: $b}) RETURN count(*)",
        {"a": from_event_id, "b": to_event_id},
    ):
        return False
    try:
        _exec(
            "MATCH (a:Event {id: $a}), (b:Event {id: $b}) CREATE (a)-[:PRECEDES]->(b)",
            {"a": from_event_id, "b": to_event_id},
        )
        return True
    except Exception as exc:
        logger.warning("graph | add_precedes: %s", exc)
        return False


def add_causes(from_event_id: str, to_event_id: str) -> bool:
    if _edge_exists(
        "MATCH (a:Event {id: $a})-[:CAUSES]->(b:Event {id: $b}) RETURN count(*)",
        {"a": from_event_id, "b": to_event_id},
    ):
        return False
    try:
        _exec(
            "MATCH (a:Event {id: $a}), (b:Event {id: $b}) CREATE (a)-[:CAUSES]->(b)",
            {"a": from_event_id, "b": to_event_id},
        )
        return True
    except Exception as exc:
        logger.warning("graph | add_causes: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Neighbor queries (for propagation)
# ---------------------------------------------------------------------------

def get_entity_neighbors(entity_id: str) -> list[dict]:
    """Events this entity participated in."""
    r = _exec(
        "MATCH (e:Entity {id: $id})-[:PARTICIPANT_IN]->(ev:Event) "
        "RETURN ev.id, ev.title, ev.source_text, ev.temporal_order",
        {"id": entity_id},
    )
    results = []
    while r.has_next():
        row = r.get_next()
        results.append({
            "node_id": row[0], "node_type": "event",
            "name": row[1], "content": row[2] or row[1],
            "is_stub": False,
        })
    return results


def get_event_entity_neighbors(event_id: str) -> list[dict]:
    """Entities that participated in this event."""
    r = _exec(
        "MATCH (e:Entity)-[:PARTICIPANT_IN]->(ev:Event {id: $id}) "
        "RETURN e.id, e.name, e.type, e.is_stub, e.confidence, e.properties",
        {"id": event_id},
    )
    results = []
    while r.has_next():
        row = r.get_next()
        props = json.loads(row[5]) if row[5] else {}
        prop_str = ", ".join(f"{k}: {v}" for k, v in props.items())
        content = f"{row[1]} ({row[2]}): {prop_str}" if prop_str else f"{row[1]} ({row[2]})"
        results.append({
            "node_id": row[0], "node_type": "entity",
            "name": row[1], "content": content,
            "is_stub": bool(row[3]), "confidence": float(row[4] or 0.5),
        })
    return results


def get_event_location_neighbors(event_id: str) -> list[dict]:
    """Location entities for this event."""
    r = _exec(
        "MATCH (ev:Event {id: $id})-[:LOCATED_AT]->(e:Entity) "
        "RETURN e.id, e.name, e.type, e.is_stub, e.confidence",
        {"id": event_id},
    )
    results = []
    while r.has_next():
        row = r.get_next()
        results.append({
            "node_id": row[0], "node_type": "entity",
            "name": row[1], "content": f"{row[1]} (location, {row[2]})",
            "is_stub": bool(row[3]), "confidence": float(row[4] or 0.5),
        })
    return results


def get_event_causal_neighbors(event_id: str) -> list[dict]:
    """Events temporally or causally linked to this event."""
    results = []
    for query in [
        "MATCH (a:Event)-[:PRECEDES]->(b:Event {id: $id}) RETURN a.id, a.title, a.source_text",
        "MATCH (a:Event {id: $id})-[:PRECEDES]->(b:Event) RETURN b.id, b.title, b.source_text",
        "MATCH (a:Event)-[:CAUSES]->(b:Event {id: $id}) RETURN a.id, a.title, a.source_text",
        "MATCH (a:Event {id: $id})-[:CAUSES]->(b:Event) RETURN b.id, b.title, b.source_text",
    ]:
        r = _exec(query, {"id": event_id})
        while r.has_next():
            row = r.get_next()
            results.append({
                "node_id": row[0], "node_type": "event",
                "name": row[1], "content": row[2] or row[1],
                "is_stub": False,
            })
    return results


def fetch_nodes_by_ids(node_ids: list[str]) -> list[dict]:
    """Direct fetch of Entity + Event nodes by their IDs.

    Returns dicts with: node_id, node_type, name, content.
    Order matches node_ids where possible.
    """
    if not node_ids:
        return []

    id_index = {nid: i for i, nid in enumerate(node_ids)}
    results: dict[str, dict] = {}

    # Entities
    r = _exec(
        "MATCH (e:Entity) WHERE e.id IN $ids "
        "RETURN e.id, e.name, e.type, e.properties",
        {"ids": node_ids},
    )
    while r.has_next():
        row = r.get_next()
        props = json.loads(row[3]) if row[3] else {}
        prop_str = ", ".join(f"{k}: {v}" for k, v in props.items())
        content = f"{row[1]} ({row[2]}): {prop_str}" if prop_str else f"{row[1]} ({row[2]})"
        results[row[0]] = {"node_id": row[0], "node_type": "entity", "name": row[1], "content": content}

    # Events
    r = _exec(
        "MATCH (ev:Event) WHERE ev.id IN $ids "
        "RETURN ev.id, ev.title, ev.source_text, ev.summary",
        {"ids": node_ids},
    )
    while r.has_next():
        row = r.get_next()
        content = row[2] or row[3] or row[1]  # source_text > summary > title
        results[row[0]] = {"node_id": row[0], "node_type": "event", "name": row[1], "content": content}

    # Return in node_ids order, skipping missing
    return [results[nid] for nid in node_ids if nid in results]


def get_all_entity_names() -> list[tuple[str, str]]:
    """Return [(id, name)] for all non-stub entities."""
    r = _exec("MATCH (e:Entity) WHERE e.is_stub = false RETURN e.id, e.name")
    results: list[tuple[str, str]] = []
    while r.has_next():
        row = r.get_next()
        results.append((row[0], row[1]))
    return results


def search_entities_by_query(query: str, limit: int = 3) -> list[dict]:
    """Case-insensitive token match against entity names in Kuzu.

    Requires ALL query tokens to appear in the entity name.
    Non-stub entities are returned first; stubs as fallback if nothing else matches.
    Tokens shorter than 2 chars and non-alphanumeric chars are stripped to prevent
    injection into the Cypher string.
    """
    import re
    tokens = [
        re.sub(r"[^a-z0-9 ]", "", t.lower())
        for t in query.split()
        if len(t) >= 2
    ]
    tokens = [t for t in tokens if t]
    if not tokens:
        return []

    conditions = " AND ".join(f"toLower(e.name) CONTAINS '{t}'" for t in tokens)

    def _run(stub_filter: str) -> list[dict]:
        try:
            res = _exec(
                f"MATCH (e:Entity) WHERE {conditions} {stub_filter}"
                f" RETURN e.id, e.name, e.type, e.is_stub LIMIT {limit}"
            )
            rows = []
            while res.has_next():
                row = res.get_next()
                rows.append({"id": row[0], "name": row[1], "type": row[2], "is_stub": row[3]})
            return rows
        except Exception:
            return []

    results = _run("AND e.is_stub = false")
    if not results:
        results = _run("")  # fall back to stubs
    return results


def get_events_for_entities(entity_ids: list[str]) -> list[dict]:
    """Events involving any of the given entities, ordered by temporal_order."""
    if not entity_ids:
        return []
    seen: set[str] = set()
    results = []
    for eid in entity_ids:
        r = _exec(
            "MATCH (e:Entity {id: $id})-[:PARTICIPANT_IN]->(ev:Event) "
            "RETURN ev.id, ev.title, ev.temporal_order ORDER BY ev.temporal_order",
            {"id": eid},
        )
        while r.has_next():
            row = r.get_next()
            if row[0] not in seen:
                seen.add(row[0])
                results.append({"id": row[0], "title": row[1], "temporal_order": row[2] or 0})
    results.sort(key=lambda x: x["temporal_order"])
    return results
