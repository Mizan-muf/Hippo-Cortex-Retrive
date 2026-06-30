# Hippo-Cortex Retrieval

A three-tier hybrid retrieval engine for persistent AI memory. Combines semantic search (Mem0 + Qdrant), structured graph traversal (Kuzu), and filesystem fallback to store, link, and query factual knowledge extracted from documents.

---

## Architecture

```
Input document
      │
      ▼
[PIPELINE]
  Stage 0 — Conflict check          (semantic search for contradictions)
  Stage 1 — Segmentation            (paragraph-level chunks)
  Stage 2 — Extraction              (LLM → events + properties)
  Stage 3 — Temporal linking        (date extraction + chronological sort)
  Stage 4 — Entity linking          (name resolution, fuzzy match, HITL queue)
  Stage 5 — Storage                 (three tiers in parallel)
      │
      ├── Tier 1: Mem0 + Qdrant     (semantic vectors, fast, lossy)
      ├── Tier 2: Kuzu graph        (entity/event nodes + edges, structured)
      └── Tier 3: File store        (Markdown records, permanent, grep-searchable)

Query
      │
      ▼
[RETRIEVAL CHAIN]
  Tier 1a pathway  →  known pattern? → fetch node IDs → pull verbatim source_text from Kuzu → return
  Tier 1b dense    →  score ≥ 0.85?  → return results
                   →  score 0.50–0.85 → expand via graph BFS (propagator)
                   →  score < 0.50   → grep file store → reconsolidate to Tier 1
```

**Neither tier replaces the others.** Tier 1 compresses old entries over time. Tier 2 preserves structured relationships. Tier 3 never forgets — when a semantic hit is a miss, the file fallback finds it and reconsolidates it back into Tier 1 automatically.

---

## Project Structure

```
Hippo-Cortex-Retrieval/
├── src/
│   ├── config.py               — env loading, all thresholds and settings
│   ├── core/
│   │   ├── llm.py              — Ollama / Gemini dispatcher
│   │   ├── embed.py            — embedding interface
│   │   ├── nlp.py              — spaCy model loader (lazy)
│   │   └── terms.py            — key-term extraction for pathway embeddings
│   ├── ingestion/
│   │   ├── pipeline.py         — 7-stage ingest orchestrator
│   │   ├── segmenter.py        — paragraph / semantic / scene-break chunking
│   │   ├── decomposer.py       — spaCy event + property extraction
│   │   ├── coref.py            — fastcoref pronoun resolution
│   │   ├── temporal_linker.py  — temporal ordering + cross-doc causal links
│   │   ├── entity_linker.py    — fuzzy / semantic entity resolution, HITL queue
│   │   └── resolver.py         — writes resolved facts to all three tiers
│   ├── retrieval/
│   │   ├── chain.py            — three-tier cascade retrieve()
│   │   ├── propagator.py       — graph BFS expansion with decay scoring
│   │   ├── rag.py              — standard chunked RAG path
│   │   ├── graph_rag.py        — graph-aware RAG variant
│   │   └── websearch.py        — web search stub (future)
│   └── storage/
│       ├── memory.py           — Mem0 + Qdrant init, search, store_pathway, store_fact_direct
│       ├── graph.py            — Kuzu schema, entity/event/property CRUD, fetch_nodes_by_ids
│       └── filestore.py        — Markdown record manager (three-layer structure)
├── scripts/
│   ├── ingest_epub.py                  — CLI: ingest EPUB chapters into the pipeline
│   ├── backfill_pathways.py            — populate pathway_cache (multi-path: 5 angles/event)
│   └── cleanup_knowledge_base_spill.py — remove accidentally spilled pathway points from knowledge_base

├── main.py                     — smoke test + interactive REPL
├── docker-compose.yml
├── requirements.txt
└── .env                        — (not committed) see .env-example
```

---

## Prerequisites

| Service | Version | Purpose |
|---------|---------|---------|
| Docker | any | runs Qdrant |
| Ollama | any | local LLM + embedding server |
| Python | 3.10+ | runtime |

Pull required models before starting:

```bash
ollama pull nomic-embed-text      # 768-dim embedder (required)
ollama pull <your-fast-model>     # lightweight LLM (decomposer, Mem0 internals)
ollama pull <your-quality-model>  # optional, for document processing
```

---

## Setup

**1. Clone and install dependencies**

```bash
pip install -r requirements.txt
```

**2. Configure environment**

Copy `.env.example` to `.env` (or create `.env`) and fill in the values (see [Environment Variables](#environment-variables)).

**3. Start Qdrant**

```bash
docker-compose up -d

# or manually (Windows CMD):
docker run -d --name app-qdrant -p 6333:6333 ^
  -v "%CD%\data\qdrant":/qdrant/storage qdrant/qdrant
```

Verify it is running:

```bash
curl http://localhost:6333/healthz
```

**4. Run the smoke test**

```bash
python main.py
```

This initialises all three tiers, writes three facts, and queries them. Healthy output ends with `memory_available: true`.

---

## Usage

### Interactive REPL

```bash
python main.py
```

Type any query at the `>` prompt. The retrieval chain runs all three tiers and returns results with confidence scores.

### Ingest an EPUB

```bash
# List chapters (no writes)
python scripts/ingest_epub.py samples/The_Strongest_Gene.epub --list

# Ingest a range of chapters
python scripts/ingest_epub.py samples/The_Strongest_Gene.epub --chapters 1-5

# Ingest all chapters
python scripts/ingest_epub.py samples/The_Strongest_Gene.epub --chapters all

# Dry run (parse only, no writes)
python scripts/ingest_epub.py samples/The_Strongest_Gene.epub --chapters all --dry-run
```

---

## Environment Variables

```env
# LLM
OLLAMA_BASE_URL=http://localhost:11434
LIGHTWEIGHT_LLM=<your-fast-model>       # used by REPL, decomposer, Mem0 internals
QUALITY_LLM=<your-quality-model>        # used only for document processing
EMBED_MODEL=nomic-embed-text            # must produce 768-dim vectors

# Tier 1 — Qdrant
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION=knowledge_base        # rename for your domain
QDRANT_VECTOR_SIZE=768                  # must match EMBED_MODEL output dims

# Tier 2 — Kuzu graph
GRAPH_PATH=./data/graph

# Tier 3 — file store
FILE_STORE_PATH=./knowledge-store       # relative to project root

# Retrieval thresholds
CONFIDENCE_THRESHOLD=0.85               # Tier 1 confident-hit cutoff
GRAPH_ANCHOR_THRESHOLD=0.50             # minimum score to trigger graph expansion
PROPAGATION_DECAY=0.90                  # BFS score decay per hop
LINK_THRESHOLD=0.88                     # entity linker fuzzy-match threshold
```

`QDRANT_VECTOR_SIZE` must always match the embedding model's output dimensions. If you switch models, drop both Qdrant collections and restart — `init_memory()` recreates them at the correct dimensions.

---

## File Store Record Structure

Every entity gets a Markdown record with three immutable layers:

```
┌──────────────────────────────────────┐
│  YAML Frontmatter                    │  name, type, aliases, status, dates
├──────────────────────────────────────┤
│  ## Profile          (Layer 1)       │  structured facts — system writes
│  ## Relationships    (Layer 1)       │  entity links — system maintains
├──────────────────────────────────────┤
│  ## Event Log        (Layer 2)       │  timestamped entries — system appends only
├──────────────────────────────────────┤
│  <!-- USER NOTES sentinel line -->   │  ← hard boundary
│  ## User Notes       (Layer 3)       │  user owns — system NEVER touches
└──────────────────────────────────────┘
```

The sentinel `<!-- USER NOTES — system will never modify below this line -->` is checked on every write. If it is missing, the write is aborted and a CRITICAL is logged to protect user content.

---

## Visual Inspection

### Qdrant Web UI

Qdrant ships with a built-in dashboard at `http://localhost:6333/dashboard`. No extra container needed — start the existing service and open the URL.

```powershell
# Start (if not already running)
docker-compose up -d

# Verify
curl http://localhost:6333/healthz
```

Then open **http://localhost:6333/dashboard** in your browser. From there you can browse collections, run searches, and inspect individual points.

**Useful REST queries (run in terminal or paste into the dashboard console):**

```bash
# List all collections
curl http://localhost:6333/collections

# Collection info (vector count, dimensions, status)
curl http://localhost:6333/collections/knowledge_base
curl http://localhost:6333/collections/knowledge_base_entities

# Scroll all points in a collection (first 10)
curl -X POST http://localhost:6333/collections/knowledge_base/points/scroll \
  -H "Content-Type: application/json" \
  -d '{"limit": 10, "with_payload": true, "with_vector": false}'

# Search by text — embed your query first with Ollama, then:
# (easier to use the dashboard's built-in search UI for ad-hoc queries)
```

---

### Kuzu Graph Explorer

Kuzu has an official browser-based graph explorer. Add it to `docker-compose.yml` to run it alongside Qdrant:

```yaml
# Add this service to docker-compose.yml
  kuzu-explorer:
    image: kuzudb/explorer:latest
    container_name: app-kuzu-explorer
    ports:
      - "8000:8000"
    volumes:
      - ./data/graph:/database
    environment:
      - MODE=READ_ONLY
    restart: unless-stopped
```

> **Important:** the pipeline must not be running when the explorer is open — Kuzu uses a file lock and only one process can hold it at a time. Stop the pipeline first, then start the explorer.

```powershell
docker run --rm -p 8000:8000 -v "${PWD}/data:/database" -e KUZU_FILE=graph kuzudb/explorer:latest
```

Open **http://localhost:8000** in your browser.

**Cypher queries to run in the explorer:**

```cypher
-- Full graph: all entities and events (limit to avoid overload)
MATCH (n) RETURN n LIMIT 100;

-- All entities with their type
MATCH (e:Entity) RETURN e.name, e.type, e.confidence ORDER BY e.name LIMIT 50;

-- All events with participants
MATCH (en:Entity)-[:PARTICIPANT_IN]->(ev:Event)
RETURN en.name, ev.title, ev.summary LIMIT 50;

-- Single entity — everything connected to it
MATCH (e:Entity {name: "Chen Feng"})-[r]-(n)
RETURN e, r, n;

-- Entity and 2-hop neighbourhood
MATCH (e:Entity {name: "Chen Feng"})-[r1]-(n1)-[r2]-(n2)
RETURN e, r1, n1, r2, n2;

-- Events for a specific document
MATCH (ev:Event)-[:CONTAINED_IN]->(d:Document)
WHERE d.title CONTAINS "Chapter"
RETURN ev.title, ev.summary, d.title ORDER BY ev.temporal_order;

-- All relationship types present
MATCH ()-[r]->() RETURN DISTINCT label(r), count(*) ORDER BY count(*) DESC;

-- Stub entities (incomplete, low confidence)
MATCH (e:Entity) WHERE e.is_stub = true RETURN e.name, e.type, e.confidence;
```

---

### Knowledge Store (File Store)

The file store at `./knowledge-store/` is plain Markdown — no tooling required. Open it directly in VS Code or any file explorer. Each entity has its own `.md` file with structured frontmatter, an event log, and a protected user notes section.

```powershell
# Count total entity records
(Get-ChildItem -Recurse knowledge-store -Filter "*.md").Count

# Search across all records for a name or term
grep -r "Chen Feng" knowledge-store/
# or in PowerShell:
Select-String -Path "knowledge-store\**\*.md" -Pattern "Chen Feng" -Recurse
```

---

## Troubleshooting

### Qdrant dimension mismatch

```
CRITICAL | memory | Qdrant collection 'knowledge_base' has 1536 dims but expected 768.
```

A previous run created the collection at OpenAI's default (1536). Fix:

```bash
curl -X DELETE http://localhost:6333/collections/knowledge_base
curl -X DELETE http://localhost:6333/collections/knowledge_base_entities
# restart backend — init_memory() recreates both at 768 dims
```

### Qdrant unreachable

```
CRITICAL | memory | Qdrant unreachable or Mem0 init failed — memory layer disabled
```

```bash
docker start app-qdrant
```

The pipeline degrades gracefully — retrieval returns empty but the REPL still functions. Check `GET /health` for `memory_available: false`.

### Decomposer returns zero facts

```
INFO | decomposer | Decomposer: extracted 0 facts from doc01 seg01
```

Possible causes: LLM returned invalid JSON, empty input text, or all subjects were pronouns. Add `format="json"` to the Ollama call to enforce structured output. Check logs for the raw LLM response.

### File store write aborted (sentinel missing)

```
CRITICAL | filestore | Sentinel missing in 'Entity_Name.md' — write aborted
```

Add the sentinel line manually between the Event Log and User Notes sections of the affected file:

```markdown
<!-- USER NOTES — system will never modify below this line -->
```

### Duplicate entity records

Two files created for the same entity (e.g. `Full_Name.md` and `Short_Name.md`). Add the short name to the canonical record's `aliases:` frontmatter field, then delete the duplicate. The alias matcher routes future facts to the canonical file.

### Qdrant scroll panic (`OffsetOutOfBounds` / `LiteralOutOfBounds`)

```
Service internal error: task panicked with message "OffsetOutOfBounds"
```

Caused by pathway points (dense-only) written into `knowledge_base` (BM25 hybrid). The BM25 index panics when it encounters points without a sparse vector. Fix:

```bash
python scripts/cleanup_knowledge_base_spill.py
```

This reconstructs the spilled point IDs via UUID5 and deletes them directly — no scroll needed.

---

## Startup Checklist

```
[ ] docker ps | grep app-qdrant          — Qdrant container running
[ ] curl http://localhost:6333/healthz   — Qdrant responds
[ ] ollama list                          — Ollama running
[ ] ollama pull nomic-embed-text         — embedding model available
[ ] ollama pull <LIGHTWEIGHT_LLM>        — lightweight model available
[ ] python main.py                       — smoke test passes, memory_available: true
[ ] curl http://localhost:6333/collections/knowledge_base         — 768 dims, BM25 hybrid
[ ] curl http://localhost:6333/collections/knowledge_base_entities — 768 dims
[ ] curl http://localhost:6333/collections/pathway_cache           — 768 dims, dense-only
```

Always start the backend from the project root so `FILE_STORE_PATH` and `GRAPH_PATH` resolve correctly.
