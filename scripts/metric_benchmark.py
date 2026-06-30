"""
Deep retrieval metric benchmark — OAT parameter sweep + combination configs.

Strategy
--------
One-At-a-Time (OAT): hold all parameters at baseline, sweep each one
independently through its full meaningful range. This surfaces the sensitivity
profile of every knob without combinatorial explosion.

After OAT, a set of multi-parameter combination configs tests synergies.

Each run:
  1. Patches all metric values into the live module namespaces
  2. Runs three-tier retrieve() for "Who is Chen Feng?"
  3. Runs graph propagation separately to capture deep PropNode-level metrics
  4. Calls Gemini with the retrieved context
  5. Computes a composite context-quality score (0–100)

The final report (Markdown) contains:
  - Table of Contents
  - Scoring formula
  - Baseline reference
  - Executive Summary (key findings, top/bottom 5, histograms)
  - Cross-Parameter Impact Analysis (ranked by score spread)
  - Per-parameter sensitivity tables
  - Timing analysis
  - Tier distribution
  - Combination config results
  - Full ranked results (all configs, detailed)
  - Optimal configuration recommendation
  - Appendix: full parameter table

Usage:
    python scripts/metric_benchmark.py
    python scripts/metric_benchmark.py --dry-run         # skip Gemini
    python scripts/metric_benchmark.py --max-configs 20  # quick test
"""

import argparse
import logging
import os
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

logging.basicConfig(level=logging.WARNING)

QUERY = "Who is Chen Feng?"

# ── Baseline ──────────────────────────────────────────────────────────────────
BASELINE: dict[str, Any] = {
    "START_THRESHOLD":           0.55,
    "CONFIDENT_THRESHOLD":       0.92,
    "GRAPH_ANCHOR_THRESHOLD":    0.35,
    "GRAPH_ANCHOR_MAX_STARTS":   3,
    "PROPAGATION_DECAY":         0.90,
    "PROPAGATION_HOP_THRESHOLD": 0.25,
    "PROPAGATION_MAX_DEPTH":     4,
    "PROPAGATION_MAX_NODES":     25,
    "STUB_PENALTY":              0.50,
    "GRAPHRAG_SEED_THRESHOLD":   0.35,
}

PARAM_ROLES: dict[str, str] = {
    "START_THRESHOLD":           "Min Tier-1 score — Qdrant hits below this are discarded entirely",
    "CONFIDENT_THRESHOLD":       "Tier-1 short-circuit — graph escalation skipped if score ≥ this",
    "GRAPH_ANCHOR_THRESHOLD":    "Min Mem0 score for a result to qualify as a graph BFS entry node",
    "GRAPH_ANCHOR_MAX_STARTS":   "Max simultaneous BFS start nodes (breadth of graph entry)",
    "PROPAGATION_DECAY":         "Per-hop score multiplier: prop_score × decay^hop",
    "PROPAGATION_HOP_THRESHOLD": "Min prop_score to keep a node — nodes below this are pruned",
    "PROPAGATION_MAX_DEPTH":     "Maximum BFS hops from each start node",
    "PROPAGATION_MAX_NODES":     "Maximum total nodes collected across all BFS walks",
    "STUB_PENALTY":              "Score multiplier for incomplete/stub entities (< 1.0 = penalty)",
    "GRAPHRAG_SEED_THRESHOLD":   "Min Qdrant score for graph-RAG seed nodes",
}

# ── OAT sweep ranges (12–21 values per parameter) ────────────────────────────
PARAM_SWEEP: dict[str, list[Any]] = {
    "START_THRESHOLD": [
        0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40,
        0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90,
    ],
    "CONFIDENT_THRESHOLD": [
        0.50, 0.55, 0.60, 0.65, 0.70, 0.74, 0.78, 0.82,
        0.85, 0.88, 0.90, 0.92, 0.94, 0.96, 0.98, 0.99,
    ],
    "GRAPH_ANCHOR_THRESHOLD": [
        0.05, 0.10, 0.15, 0.18, 0.20, 0.23, 0.25, 0.28,
        0.30, 0.33, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70,
    ],
    "GRAPH_ANCHOR_MAX_STARTS": [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15],
    "PROPAGATION_DECAY": [
        0.20, 0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70,
        0.75, 0.80, 0.83, 0.86, 0.88, 0.90, 0.92, 0.94,
        0.96, 0.97, 0.98, 0.99, 1.00,
    ],
    "PROPAGATION_HOP_THRESHOLD": [
        0.01, 0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20,
        0.22, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70,
    ],
    "PROPAGATION_MAX_DEPTH": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15],
    "PROPAGATION_MAX_NODES": [
        3, 5, 8, 10, 12, 15, 18, 20, 25, 30,
        35, 40, 50, 60, 75, 100,
    ],
    "STUB_PENALTY": [
        0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35,
        0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75,
        0.80, 0.85, 0.90, 0.95, 1.00,
    ],
    "GRAPHRAG_SEED_THRESHOLD": [
        0.05, 0.10, 0.15, 0.18, 0.20, 0.23, 0.25, 0.28,
        0.30, 0.33, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70,
    ],
}

# ── Combination configs (12 multi-parameter interactions) ─────────────────────
COMBO_CONFIGS: list[dict[str, Any]] = [
    {
        "name": "combo_max_recall",
        "description": "All thresholds at minimum — widest possible net, maximum noise",
        "START_THRESHOLD": 0.05, "CONFIDENT_THRESHOLD": 0.99,
        "GRAPH_ANCHOR_THRESHOLD": 0.05, "GRAPH_ANCHOR_MAX_STARTS": 15,
        "PROPAGATION_DECAY": 0.99, "PROPAGATION_HOP_THRESHOLD": 0.01,
        "PROPAGATION_MAX_DEPTH": 15, "PROPAGATION_MAX_NODES": 100,
        "STUB_PENALTY": 1.00, "GRAPHRAG_SEED_THRESHOLD": 0.05,
    },
    {
        "name": "combo_max_precision",
        "description": "All thresholds at maximum — only the highest-confidence signals pass",
        "START_THRESHOLD": 0.90, "CONFIDENT_THRESHOLD": 0.50,
        "GRAPH_ANCHOR_THRESHOLD": 0.70, "GRAPH_ANCHOR_MAX_STARTS": 1,
        "PROPAGATION_DECAY": 0.20, "PROPAGATION_HOP_THRESHOLD": 0.70,
        "PROPAGATION_MAX_DEPTH": 1, "PROPAGATION_MAX_NODES": 3,
        "STUB_PENALTY": 0.01, "GRAPHRAG_SEED_THRESHOLD": 0.70,
    },
    {
        "name": "combo_deep_permissive",
        "description": "Deep traversal + permissive thresholds + slow decay",
        "START_THRESHOLD": 0.25, "CONFIDENT_THRESHOLD": 0.99,
        "GRAPH_ANCHOR_THRESHOLD": 0.10, "GRAPH_ANCHOR_MAX_STARTS": 10,
        "PROPAGATION_DECAY": 0.98, "PROPAGATION_HOP_THRESHOLD": 0.05,
        "PROPAGATION_MAX_DEPTH": 15, "PROPAGATION_MAX_NODES": 100,
        "STUB_PENALTY": 0.85, "GRAPHRAG_SEED_THRESHOLD": 0.10,
    },
    {
        "name": "combo_graph_focused",
        "description": "Force graph escalation — balanced decay, stub-free",
        "START_THRESHOLD": 0.40, "CONFIDENT_THRESHOLD": 0.99,
        "GRAPH_ANCHOR_THRESHOLD": 0.20, "GRAPH_ANCHOR_MAX_STARTS": 6,
        "PROPAGATION_DECAY": 0.90, "PROPAGATION_HOP_THRESHOLD": 0.15,
        "PROPAGATION_MAX_DEPTH": 6, "PROPAGATION_MAX_NODES": 50,
        "STUB_PENALTY": 0.05, "GRAPHRAG_SEED_THRESHOLD": 0.20,
    },
    {
        "name": "combo_tier1_only",
        "description": "Disable graph — confident threshold below start threshold",
        "START_THRESHOLD": 0.55, "CONFIDENT_THRESHOLD": 0.30,
        "GRAPH_ANCHOR_THRESHOLD": 0.90, "GRAPH_ANCHOR_MAX_STARTS": 1,
        "PROPAGATION_DECAY": 0.20, "PROPAGATION_HOP_THRESHOLD": 0.90,
        "PROPAGATION_MAX_DEPTH": 1, "PROPAGATION_MAX_NODES": 3,
        "STUB_PENALTY": 0.50, "GRAPHRAG_SEED_THRESHOLD": 0.90,
    },
    {
        "name": "combo_no_stub_deep",
        "description": "No stub penalty + deep traversal — include all incomplete entities",
        "START_THRESHOLD": 0.40, "CONFIDENT_THRESHOLD": 0.92,
        "GRAPH_ANCHOR_THRESHOLD": 0.20, "GRAPH_ANCHOR_MAX_STARTS": 6,
        "PROPAGATION_DECAY": 0.96, "PROPAGATION_HOP_THRESHOLD": 0.10,
        "PROPAGATION_MAX_DEPTH": 12, "PROPAGATION_MAX_NODES": 75,
        "STUB_PENALTY": 1.00, "GRAPHRAG_SEED_THRESHOLD": 0.20,
    },
    {
        "name": "combo_balanced_wide",
        "description": "Moderate all settings — wider than baseline but not extreme",
        "START_THRESHOLD": 0.40, "CONFIDENT_THRESHOLD": 0.95,
        "GRAPH_ANCHOR_THRESHOLD": 0.25, "GRAPH_ANCHOR_MAX_STARTS": 5,
        "PROPAGATION_DECAY": 0.93, "PROPAGATION_HOP_THRESHOLD": 0.18,
        "PROPAGATION_MAX_DEPTH": 6, "PROPAGATION_MAX_NODES": 40,
        "STUB_PENALTY": 0.60, "GRAPHRAG_SEED_THRESHOLD": 0.25,
    },
    {
        "name": "combo_aggressive_anchoring",
        "description": "Many start nodes + very low anchor threshold + moderate depth",
        "START_THRESHOLD": 0.35, "CONFIDENT_THRESHOLD": 0.99,
        "GRAPH_ANCHOR_THRESHOLD": 0.08, "GRAPH_ANCHOR_MAX_STARTS": 15,
        "PROPAGATION_DECAY": 0.88, "PROPAGATION_HOP_THRESHOLD": 0.20,
        "PROPAGATION_MAX_DEPTH": 5, "PROPAGATION_MAX_NODES": 60,
        "STUB_PENALTY": 0.50, "GRAPHRAG_SEED_THRESHOLD": 0.10,
    },
    {
        "name": "combo_slow_decay_shallow",
        "description": "Very slow decay but limited depth — weight nearby nodes heavily",
        "START_THRESHOLD": 0.50, "CONFIDENT_THRESHOLD": 0.92,
        "GRAPH_ANCHOR_THRESHOLD": 0.30, "GRAPH_ANCHOR_MAX_STARTS": 4,
        "PROPAGATION_DECAY": 0.99, "PROPAGATION_HOP_THRESHOLD": 0.30,
        "PROPAGATION_MAX_DEPTH": 3, "PROPAGATION_MAX_NODES": 20,
        "STUB_PENALTY": 0.50, "GRAPHRAG_SEED_THRESHOLD": 0.30,
    },
    {
        "name": "combo_fast_decay_deep",
        "description": "Steep decay but many hops — only high-intrinsic deep nodes survive",
        "START_THRESHOLD": 0.50, "CONFIDENT_THRESHOLD": 0.92,
        "GRAPH_ANCHOR_THRESHOLD": 0.30, "GRAPH_ANCHOR_MAX_STARTS": 4,
        "PROPAGATION_DECAY": 0.50, "PROPAGATION_HOP_THRESHOLD": 0.05,
        "PROPAGATION_MAX_DEPTH": 15, "PROPAGATION_MAX_NODES": 75,
        "STUB_PENALTY": 0.50, "GRAPHRAG_SEED_THRESHOLD": 0.30,
    },
    {
        "name": "combo_strict_stubs_wide",
        "description": "Severe stub penalty + wide graph — force high-quality paths only",
        "START_THRESHOLD": 0.35, "CONFIDENT_THRESHOLD": 0.99,
        "GRAPH_ANCHOR_THRESHOLD": 0.15, "GRAPH_ANCHOR_MAX_STARTS": 8,
        "PROPAGATION_DECAY": 0.94, "PROPAGATION_HOP_THRESHOLD": 0.12,
        "PROPAGATION_MAX_DEPTH": 8, "PROPAGATION_MAX_NODES": 60,
        "STUB_PENALTY": 0.01, "GRAPHRAG_SEED_THRESHOLD": 0.15,
    },
    {
        "name": "combo_seed_heavy",
        "description": "Very low seed threshold — maximise graph-RAG seed coverage",
        "START_THRESHOLD": 0.30, "CONFIDENT_THRESHOLD": 0.99,
        "GRAPH_ANCHOR_THRESHOLD": 0.10, "GRAPH_ANCHOR_MAX_STARTS": 10,
        "PROPAGATION_DECAY": 0.92, "PROPAGATION_HOP_THRESHOLD": 0.10,
        "PROPAGATION_MAX_DEPTH": 8, "PROPAGATION_MAX_NODES": 80,
        "STUB_PENALTY": 0.70, "GRAPHRAG_SEED_THRESHOLD": 0.05,
    },
]

TIER_RANK = {"miss": 0, "tier2": 1, "tier1": 2, "tier1+tier2": 3}
TIER_LABEL = {0: "miss", 1: "tier2", 2: "tier1", 3: "tier1+tier2"}

# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GraphMetrics:
    anchor_count: int = 0
    total_nodes: int = 0
    nodes_per_hop: dict = field(default_factory=dict)
    avg_prop_score: float = 0.0
    max_prop_score: float = 0.0
    min_prop_score: float = 0.0
    avg_intrinsic_score: float = 0.0
    stub_count: int = 0
    stub_ratio: float = 0.0
    max_hop_reached: int = 0
    entity_count: int = 0
    event_count: int = 0
    unique_names: int = 0
    error: str = ""


@dataclass
class BenchmarkResult:
    config_name: str
    description: str
    sweep_param: str
    sweep_value: Any
    config: dict[str, Any]

    hits: list[str] = field(default_factory=list)
    source: str = "miss"
    confidence: float = 0.0
    reconsolidated: bool = False
    graph: GraphMetrics = field(default_factory=GraphMetrics)
    gemini_answer: str = ""

    retrieve_s: float = 0.0
    graph_metrics_s: float = 0.0
    llm_s: float = 0.0
    error: str = ""

    hits_count: int = 0
    total_context_chars: int = 0
    chen_feng_mentions: int = 0
    answer_length: int = 0
    source_tier_rank: int = 0
    composite_score: float = 0.0

    @property
    def total_s(self) -> float:
        return round(self.retrieve_s + self.graph_metrics_s + self.llm_s, 3)


# ─────────────────────────────────────────────────────────────────────────────
# Config generation
# ─────────────────────────────────────────────────────────────────────────────

def build_configs() -> list[dict[str, Any]]:
    configs: list[dict[str, Any]] = []

    # Baseline first
    configs.append({
        "name": "baseline",
        "description": "All parameters at .env defaults",
        "sweep_param": "baseline",
        "sweep_value": None,
        **BASELINE,
    })

    # OAT sweep
    for param, values in PARAM_SWEEP.items():
        for val in values:
            cfg = {**BASELINE, param: val}
            cfg["name"] = f"oat_{param.lower()}_{val}"
            cfg["description"] = f"OAT: {param}={val}  (baseline={BASELINE[param]})"
            cfg["sweep_param"] = param
            cfg["sweep_value"] = val
            configs.append(cfg)

    # Combination configs
    for combo in COMBO_CONFIGS:
        c = {**BASELINE, **combo}
        c["sweep_param"] = "combination"
        c["sweep_value"] = combo["name"]
        configs.append(c)

    return configs


# ─────────────────────────────────────────────────────────────────────────────
# Module patching
# ─────────────────────────────────────────────────────────────────────────────

_PATCHABLE = {
    "START_THRESHOLD", "CONFIDENT_THRESHOLD",
    "GRAPH_ANCHOR_THRESHOLD", "GRAPH_ANCHOR_MAX_STARTS",
    "PROPAGATION_DECAY", "PROPAGATION_HOP_THRESHOLD",
    "PROPAGATION_MAX_DEPTH", "PROPAGATION_MAX_NODES",
    "STUB_PENALTY", "GRAPHRAG_SEED_THRESHOLD",
}


def _apply_config(cfg: dict[str, Any]) -> None:
    import src.config as conf
    import src.retrieval.chain as chain
    import src.retrieval.propagator as prop
    for k, v in cfg.items():
        if k not in _PATCHABLE:
            continue
        for mod in (conf, chain, prop):
            if hasattr(mod, k):
                setattr(mod, k, v)


def _force_gemini() -> None:
    import src.core.llm as llm_mod
    llm_mod.LLM_PROVIDER = "gemini"


# ─────────────────────────────────────────────────────────────────────────────
# Graph metrics
# ─────────────────────────────────────────────────────────────────────────────

def _collect_graph_metrics(query: str) -> GraphMetrics:
    gm = GraphMetrics()
    try:
        from src.storage.graph import is_initialized
        from src.retrieval.propagator import find_start_nodes, propagate

        if not is_initialized():
            return gm

        starts = find_start_nodes(query)
        gm.anchor_count = len(starts)
        if not starts:
            return gm

        seen: dict[str, Any] = {}
        for start in starts:
            result = propagate(
                query,
                start["node_id"], start["node_type"],
                start["name"], start["content"],
                start_is_stub=start.get("is_stub", False),
            )
            for node in result.nodes:
                if node.node_id not in seen or node.prop_score > seen[node.node_id].prop_score:
                    seen[node.node_id] = node

        nodes = list(seen.values())
        if not nodes:
            return gm

        scores_prop = [n.prop_score for n in nodes]
        scores_intr = [n.intrinsic_score for n in nodes]
        hops: dict[int, int] = {}
        for n in nodes:
            hops[n.hop] = hops.get(n.hop, 0) + 1

        gm.total_nodes = len(nodes)
        gm.avg_prop_score = round(sum(scores_prop) / len(scores_prop), 4)
        gm.max_prop_score = round(max(scores_prop), 4)
        gm.min_prop_score = round(min(scores_prop), 4)
        gm.avg_intrinsic_score = round(sum(scores_intr) / len(scores_intr), 4)
        gm.stub_count = sum(1 for n in nodes if n.is_stub)
        gm.stub_ratio = round(gm.stub_count / len(nodes), 3) if nodes else 0.0
        gm.max_hop_reached = max(n.hop for n in nodes)
        gm.entity_count = sum(1 for n in nodes if n.node_type == "entity")
        gm.event_count = sum(1 for n in nodes if n.node_type == "event")
        gm.unique_names = len(set(n.name for n in nodes))
        gm.nodes_per_hop = {str(k): v for k, v in sorted(hops.items())}

    except Exception as exc:
        gm.error = str(exc)
    return gm


# ─────────────────────────────────────────────────────────────────────────────
# Scoring
# ─────────────────────────────────────────────────────────────────────────────

def _composite_score(r: BenchmarkResult) -> float:
    """
    Composite context-quality score (0–100):
      source_tier_rank × 15  → 0–45  (tier quality)
      confidence × 20        → 0–20  (semantic match)
      min(hits,5) × 3        → 0–15  (breadth)
      min(chars/100, 15)     → 0–15  (volume)
      min(mentions,5) × 2    → 0–10  (direct entity coverage)
    """
    return round(
        min(
            r.source_tier_rank * 15
            + r.confidence * 20
            + min(r.hits_count, 5) * 3
            + min(r.total_context_chars / 100, 15)
            + min(r.chen_feng_mentions, 5) * 2,
            100.0,
        ),
        2,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark runner
# ─────────────────────────────────────────────────────────────────────────────

def run_benchmark(configs: list[dict], dry_run: bool = False) -> list[BenchmarkResult]:
    from src.storage.memory import init_memory
    from src.storage.graph import init_graph
    from src.retrieval.chain import retrieve, format_context
    from src.core.llm import call as llm_call

    _force_gemini()
    print("Initialising stores (Qdrant + Kuzu)...")
    init_memory()
    init_graph()

    results: list[BenchmarkResult] = []
    total = len(configs)

    for idx, cfg in enumerate(configs, 1):
        name = cfg["name"]
        print(f"\n[{idx:>3}/{total}] {name}")
        _apply_config(cfg)

        br = BenchmarkResult(
            config_name=name,
            description=cfg.get("description", ""),
            sweep_param=cfg.get("sweep_param", ""),
            sweep_value=cfg.get("sweep_value", ""),
            config={k: v for k, v in cfg.items()
                    if k not in ("name", "description", "sweep_param", "sweep_value")},
        )

        # Retrieval
        t0 = time.perf_counter()
        ret = None
        try:
            ret = retrieve(QUERY)
            br.hits = ret.hits
            br.source = ret.source
            br.confidence = ret.confidence
            br.reconsolidated = ret.reconsolidated
        except Exception as exc:
            br.error = f"retrieve: {exc}"
            print(f"  RETRIEVE ERROR: {exc}")
        br.retrieve_s = round(time.perf_counter() - t0, 3)

        # Deep graph metrics
        t1 = time.perf_counter()
        br.graph = _collect_graph_metrics(QUERY)
        br.graph_metrics_s = round(time.perf_counter() - t1, 3)

        # Gemini
        if not dry_run and not br.error and ret is not None:
            t2 = time.perf_counter()
            try:
                context_str = format_context(ret)
                prompt = (
                    f"Context:\n{context_str}\n\n"
                    f"Question: {QUERY}\n\n"
                    "Answer concisely and precisely based only on the context above. "
                    "If the context does not contain enough information, "
                    "say exactly what is missing."
                )
                br.gemini_answer = llm_call(prompt, temperature=0.1)
            except Exception as exc:
                br.gemini_answer = f"[LLM ERROR: {exc}]"
            br.llm_s = round(time.perf_counter() - t2, 3)
        elif dry_run:
            br.gemini_answer = "[dry-run — Gemini skipped]"

        # Derived metrics
        full_ctx = " ".join(br.hits)
        br.hits_count = len(br.hits)
        br.total_context_chars = len(full_ctx)
        br.chen_feng_mentions = full_ctx.lower().count("chen feng")
        br.answer_length = len(br.gemini_answer)
        br.source_tier_rank = TIER_RANK.get(br.source, 0)
        br.composite_score = _composite_score(br)
        results.append(br)

        print(
            f"  tier={br.source:<12} conf={br.confidence:.3f}  "
            f"hits={br.hits_count}  chars={br.total_context_chars}  "
            f"nodes={br.graph.total_nodes}  score={br.composite_score:.1f}  "
            f"({br.total_s:.1f}s)"
        )

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Report helpers
# ─────────────────────────────────────────────────────────────────────────────

def _bar(score: float, max_val: float = 100.0, width: int = 24) -> str:
    filled = int(round(max(0.0, min(score, max_val)) / max_val * width))
    return "█" * filled + "░" * (width - filled)


def _hbar(count: int, max_count: int, width: int = 20) -> str:
    filled = int(round(count / max_count * width)) if max_count else 0
    return "▓" * filled + "·" * (width - filled)


def _hop_dist(nodes_per_hop: dict) -> str:
    if not nodes_per_hop:
        return "—"
    return "  ".join(
        f"h{k}:{v}" for k, v in sorted(nodes_per_hop.items(), key=lambda x: int(x[0]))
    )


def _pct(n: int, total: int) -> str:
    return f"{n / total * 100:.1f}%" if total else "—"


def _score_bucket(score: float) -> str:
    b = int(score // 10) * 10
    return f"{b}–{b+9}"


# ─────────────────────────────────────────────────────────────────────────────
# Report writer
# ─────────────────────────────────────────────────────────────────────────────

def write_report(results: list[BenchmarkResult], out_path: Path) -> None:  # noqa: C901
    ranked = sorted(results, key=lambda r: r.composite_score, reverse=True)
    oat_results = [r for r in results if r.sweep_param not in ("baseline", "combination")]
    combo_results = [r for r in results if r.sweep_param == "combination"]
    baseline_r = next((r for r in results if r.sweep_param == "baseline"), None)
    baseline_score = baseline_r.composite_score if baseline_r else 0.0
    total = len(results)

    lines: list[str] = []

    # ── Cover ─────────────────────────────────────────────────────────────────
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines += [
        "# Hippo-Cortex Deep Retrieval Metric Benchmark",
        "",
        f"| Field | Value |",
        f"|-------|-------|",
        f"| Query | `{QUERY}` |",
        f"| Generated | {now_str} |",
        f"| Strategy | OAT parameter sweep + combination configs |",
        f"| Total configurations | {total} |",
        f"| — OAT sweep | {len(oat_results)} runs across {len(PARAM_SWEEP)} parameters |",
        f"| — Combinations | {len(combo_results)} |",
        f"| — Baseline | 1 |",
        f"| LLM | Gemini (`{os.getenv('GEMINI_MODEL', 'gemini-2.5-flash-lite')}`) |",
        "",
        "---",
        "",
    ]

    # ── Table of Contents ─────────────────────────────────────────────────────
    lines += [
        "## Table of Contents",
        "",
        "1. [Scoring Formula](#scoring-formula)",
        "2. [Baseline Reference](#baseline-reference)",
        "3. [Executive Summary](#executive-summary)",
        "   - Top 5 Configurations",
        "   - Bottom 5 Configurations",
        "   - Score Distribution Histogram",
        "   - Tier Distribution",
        "4. [Cross-Parameter Impact Analysis](#cross-parameter-impact-analysis)",
        "5. [Parameter Sensitivity Analysis](#parameter-sensitivity-analysis)",
        "6. [Timing Analysis](#timing-analysis)",
        "7. [Graph Coverage Statistics](#graph-coverage-statistics)",
        "8. [Combination Configuration Results](#combination-configuration-results)",
        "9. [Full Ranked Results](#full-ranked-results)",
        "10. [Optimal Configuration Recommendation](#optimal-configuration-recommendation)",
        "11. [Appendix — All Configurations](#appendix--all-configurations)",
        "",
        "---",
        "",
    ]

    # ── Scoring Formula ───────────────────────────────────────────────────────
    lines += [
        "## Scoring Formula",
        "",
        "The **composite context-quality score** (0–100) measures how much relevant",
        "information was retrieved, not answer quality directly.",
        "",
        "| Component | Formula | Max pts | Rationale |",
        "|-----------|---------|---------|-----------|",
        "| Source tier | `tier_rank × 15` | 45 | miss=0, tier2=15, tier1=30, tier1+tier2=45 |",
        "| Retrieval confidence | `confidence × 20` | 20 | Qdrant semantic match quality |",
        "| Hits breadth | `min(hits, 5) × 3` | 15 | More distinct snippets = more coverage |",
        "| Context volume | `min(chars / 100, 15)` | 15 | Longer context = more information |",
        "| Entity coverage | `min(mentions, 5) × 2` | 10 | Direct 'Chen Feng' references |",
        "| **Total** | | **100** | Capped at 100 |",
        "",
        "---",
        "",
    ]

    # ── Baseline Reference ────────────────────────────────────────────────────
    if baseline_r:
        g = baseline_r.graph
        lines += [
            "## Baseline Reference",
            "",
            "All OAT runs hold every other parameter at these values.",
            "",
            "**Retrieval Metrics**",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Source Tier | `{baseline_r.source}` |",
            f"| Confidence | `{baseline_r.confidence:.4f}` |",
            f"| Hits | `{baseline_r.hits_count}` |",
            f"| Context Chars | `{baseline_r.total_context_chars}` |",
            f"| 'Chen Feng' Mentions | `{baseline_r.chen_feng_mentions}` |",
            f"| Composite Score | **`{baseline_r.composite_score:.1f} / 100`** |",
            f"| Score Bar | `{_bar(baseline_r.composite_score)}` |",
            "",
            "**Baseline Parameters**",
            "",
            "| Parameter | Value | Role |",
            "|-----------|-------|------|",
        ]
        for k, v in BASELINE.items():
            lines.append(f"| `{k}` | `{v}` | {PARAM_ROLES.get(k, '—')} |")

        lines += [
            "",
            "**Graph Metrics at Baseline**",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Anchor Nodes | `{g.anchor_count}` |",
            f"| Total Nodes Collected | `{g.total_nodes}` |",
            f"| Max Hop Reached | `{g.max_hop_reached}` |",
            f"| Avg Prop Score | `{g.avg_prop_score:.4f}` |",
            f"| Stub Count / Ratio | `{g.stub_count}` / `{g.stub_ratio:.1%}` |",
            f"| Entity Nodes | `{g.entity_count}` |",
            f"| Event Nodes | `{g.event_count}` |",
            "",
            "---",
            "",
        ]

    # ── Executive Summary ─────────────────────────────────────────────────────
    lines += ["## Executive Summary", ""]

    # Key findings
    scores = [r.composite_score for r in results]
    avg_score = sum(scores) / len(scores) if scores else 0
    best_overall = ranked[0] if ranked else None
    worst_overall = ranked[-1] if ranked else None
    tier_counts = Counter(r.source for r in results)

    # Param with widest spread
    spreads = {}
    for param in PARAM_SWEEP:
        pr = [r for r in results if r.sweep_param == param]
        if pr:
            spreads[param] = max(r.composite_score for r in pr) - min(r.composite_score for r in pr)
    most_impactful = max(spreads, key=spreads.get) if spreads else "—"

    lines += [
        "### Key Findings",
        "",
        f"- **{total} configurations** tested in total",
        f"- Average composite score: **{avg_score:.1f} / 100**",
        f"- Best score: **{best_overall.composite_score:.1f}** (`{best_overall.config_name}`, tier=`{best_overall.source}`)" if best_overall else "",
        f"- Worst score: **{worst_overall.composite_score:.1f}** (`{worst_overall.config_name}`, tier=`{worst_overall.source}`)" if worst_overall else "",
        f"- Most impactful parameter: **`{most_impactful}`** (score spread = {spreads.get(most_impactful, 0):.1f} pts)",
        f"- Tier distribution: " + " | ".join(f"`{t}`: {c}" for t, c in sorted(tier_counts.items())),
        "",
    ]

    # Top 5
    lines += [
        "### Top 5 Configurations",
        "",
        "| Rank | Config | Swept Param | Value | Tier | Confidence | Score | Bar |",
        "|------|--------|-------------|-------|------|-----------|-------|-----|",
    ]
    for rank, r in enumerate(ranked[:5], 1):
        val_str = str(r.sweep_value) if r.sweep_value is not None else "—"
        lines.append(
            f"| {rank} | `{r.config_name}` | `{r.sweep_param}` | `{val_str}` "
            f"| `{r.source}` | `{r.confidence:.3f}` | **{r.composite_score:.1f}** "
            f"| `{_bar(r.composite_score)}` |"
        )

    # Bottom 5
    lines += [
        "",
        "### Bottom 5 Configurations",
        "",
        "| Rank | Config | Swept Param | Value | Tier | Confidence | Score |",
        "|------|--------|-------------|-------|------|-----------|-------|",
    ]
    for rank, r in enumerate(ranked[-5:][::-1], len(ranked) - 4):
        val_str = str(r.sweep_value) if r.sweep_value is not None else "—"
        lines.append(
            f"| {rank} | `{r.config_name}` | `{r.sweep_param}` | `{val_str}` "
            f"| `{r.source}` | `{r.confidence:.3f}` | `{r.composite_score:.1f}` |"
        )

    # Score histogram
    lines += ["", "### Score Distribution Histogram", ""]
    buckets: dict[str, int] = {}
    for r in results:
        b = _score_bucket(r.composite_score)
        buckets[b] = buckets.get(b, 0) + 1
    max_bucket = max(buckets.values()) if buckets else 1
    lines += ["| Score Range | Count | Bar |", "|-------------|-------|-----|"]
    for label in [f"{i}–{i+9}" for i in range(0, 101, 10)]:
        count = buckets.get(label, 0)
        lines.append(f"| {label} | {count} | `{_hbar(count, max_bucket)}` |")

    # Tier distribution
    lines += ["", "### Tier Distribution", ""]
    lines += ["| Source Tier | Count | % of Total | Bar |", "|------------|-------|------------|-----|"]
    tier_max = max(tier_counts.values()) if tier_counts else 1
    for tier in ("miss", "tier2", "tier1", "tier1+tier2"):
        count = tier_counts.get(tier, 0)
        lines.append(
            f"| `{tier}` | {count} | {_pct(count, total)} "
            f"| `{_hbar(count, tier_max)}` |"
        )

    lines += ["", "---", ""]

    # ── Cross-Parameter Impact Analysis ───────────────────────────────────────
    lines += [
        "## Cross-Parameter Impact Analysis",
        "",
        "Parameters ranked by **score spread** (best – worst value within the OAT sweep).",
        "A high spread means this parameter strongly affects retrieval quality for this query.",
        "",
        "| Rank | Parameter | Min Score | Max Score | Spread | Best Value | Worst Value |",
        "|------|-----------|-----------|-----------|--------|-----------|------------|",
    ]
    spread_rows = []
    for param in PARAM_SWEEP:
        pr = sorted([r for r in results if r.sweep_param == param], key=lambda r: r.composite_score)
        if not pr:
            continue
        lo, hi = pr[0], pr[-1]
        spread_rows.append((param, lo, hi, hi.composite_score - lo.composite_score))
    spread_rows.sort(key=lambda x: x[3], reverse=True)

    for rank, (param, lo, hi, spread) in enumerate(spread_rows, 1):
        lines.append(
            f"| {rank} | `{param}` | `{lo.composite_score:.1f}` (`{lo.sweep_value}`) "
            f"| `{hi.composite_score:.1f}` (`{hi.sweep_value}`) | **{spread:.1f}** "
            f"| `{hi.sweep_value}` | `{lo.sweep_value}` |"
        )

    lines += ["", "---", ""]

    # ── Parameter Sensitivity Analysis ───────────────────────────────────────
    lines += [
        "## Parameter Sensitivity Analysis",
        "",
        "For each parameter: every OAT value tested, full metrics, and Δ vs baseline.",
        "",
    ]

    for param in PARAM_SWEEP:
        param_runs = sorted(
            [r for r in results if r.sweep_param == param],
            key=lambda r: r.sweep_value,
        )
        if not param_runs:
            continue

        best_p = max(param_runs, key=lambda r: r.composite_score)
        worst_p = min(param_runs, key=lambda r: r.composite_score)
        spread_p = best_p.composite_score - worst_p.composite_score

        lines += [
            f"### `{param}`",
            "",
            f"> **Role:** {PARAM_ROLES.get(param, '—')}  ",
            f"> **Baseline value:** `{BASELINE[param]}`  ",
            f"> **Sweep range:** `{param_runs[0].sweep_value}` → `{param_runs[-1].sweep_value}` "
            f"({len(param_runs)} values)  ",
            f"> **Score spread:** `{spread_p:.1f}` pts  ",
            f"> **Best value:** `{best_p.sweep_value}` (score `{best_p.composite_score:.1f}`)  ",
            f"> **Worst value:** `{worst_p.sweep_value}` (score `{worst_p.composite_score:.1f}`)  ",
            "",
            "| Value | Tier | Conf | Hits | Chars | Graph Nodes | Anchors | CF Mentions | Score | Δ | Bar |",
            "|-------|------|------|------|-------|------------|---------|------------|-------|---|-----|",
        ]
        for r in param_runs:
            delta = r.composite_score - baseline_score
            delta_str = f"+{delta:.1f}" if delta >= 0 else f"{delta:.1f}"
            is_base = r.sweep_value == BASELINE[param]
            marker = " **\\***" if is_base else ""
            lines.append(
                f"| `{r.sweep_value}`{marker} | `{r.source}` | `{r.confidence:.3f}` "
                f"| {r.hits_count} | {r.total_context_chars} | {r.graph.total_nodes} "
                f"| {r.graph.anchor_count} | {r.chen_feng_mentions} "
                f"| **{r.composite_score:.1f}** | {delta_str} "
                f"| `{_bar(r.composite_score, width=16)}` |"
            )

        lines += [
            "",
            "> \\* = baseline value",
            "",
        ]

    lines += ["---", ""]

    # ── Timing Analysis ───────────────────────────────────────────────────────
    lines += ["## Timing Analysis", ""]

    total_times = [r.total_s for r in results]
    avg_t = sum(total_times) / len(total_times) if total_times else 0
    fastest = min(results, key=lambda r: r.total_s)
    slowest = max(results, key=lambda r: r.total_s)

    lines += [
        "| Metric | Value |",
        "|--------|-------|",
        f"| Total wall-clock time | `{sum(total_times):.1f}s` |",
        f"| Average per config | `{avg_t:.2f}s` |",
        f"| Fastest config | `{fastest.config_name}` (`{fastest.total_s}s`) |",
        f"| Slowest config | `{slowest.config_name}` (`{slowest.total_s}s`) |",
        "",
        "**Average time breakdown by tier:**",
        "",
        "| Tier | Avg Retrieve (s) | Avg Graph Metrics (s) | Avg LLM (s) | Avg Total (s) | Count |",
        "|------|-----------------|----------------------|------------|--------------|-------|",
    ]
    for tier in ("miss", "tier2", "tier1", "tier1+tier2"):
        tr = [r for r in results if r.source == tier]
        if not tr:
            continue
        ar = round(sum(r.retrieve_s for r in tr) / len(tr), 3)
        ag = round(sum(r.graph_metrics_s for r in tr) / len(tr), 3)
        al = round(sum(r.llm_s for r in tr) / len(tr), 3)
        at = round(sum(r.total_s for r in tr) / len(tr), 3)
        lines.append(f"| `{tier}` | `{ar}` | `{ag}` | `{al}` | `{at}` | {len(tr)} |")

    lines += ["", "**Top 5 slowest configs:**", ""]
    slowest5 = sorted(results, key=lambda r: r.total_s, reverse=True)[:5]
    lines += ["| Config | Retrieve | Graph | LLM | Total |", "|--------|---------|-------|-----|-------|"]
    for r in slowest5:
        lines.append(f"| `{r.config_name}` | `{r.retrieve_s}s` | `{r.graph_metrics_s}s` | `{r.llm_s}s` | `{r.total_s}s` |")

    lines += ["", "---", ""]

    # ── Graph Coverage Statistics ─────────────────────────────────────────────
    lines += ["## Graph Coverage Statistics", ""]

    graph_runs = [r for r in results if r.graph.total_nodes > 0]
    no_graph = [r for r in results if r.graph.total_nodes == 0]

    lines += [
        f"- Configs that reached graph propagation: **{len(graph_runs)}** / {total}",
        f"- Configs with no graph nodes: **{len(no_graph)}** / {total}",
        "",
        "| Statistic | Value |",
        "|-----------|-------|",
    ]
    if graph_runs:
        all_nodes = [r.graph.total_nodes for r in graph_runs]
        all_anchors = [r.graph.anchor_count for r in graph_runs]
        all_avg_prop = [r.graph.avg_prop_score for r in graph_runs]
        lines += [
            f"| Avg nodes per graph run | `{sum(all_nodes)/len(all_nodes):.1f}` |",
            f"| Max nodes in one run | `{max(all_nodes)}` |",
            f"| Min nodes in one run | `{min(all_nodes)}` |",
            f"| Avg anchor count | `{sum(all_anchors)/len(all_anchors):.1f}` |",
            f"| Avg prop score (across graph runs) | `{sum(all_avg_prop)/len(all_avg_prop):.4f}` |",
            f"| Runs with stubs present | `{sum(1 for r in graph_runs if r.graph.stub_count > 0)}` |",
        ]

    lines += ["", "---", ""]

    # ── Combination Configs ───────────────────────────────────────────────────
    if combo_results:
        combo_ranked = sorted(combo_results, key=lambda r: r.composite_score, reverse=True)
        lines += [
            "## Combination Configuration Results",
            "",
            "Multi-parameter combos. Each deviates from baseline on all parameters simultaneously.",
            "",
            "| Rank | Config | Tier | Conf | Hits | Chars | Nodes | Score | Δ vs baseline | Bar |",
            "|------|--------|------|------|------|-------|-------|-------|---------------|-----|",
        ]
        for rank, r in enumerate(combo_ranked, 1):
            delta = r.composite_score - baseline_score
            delta_str = f"+{delta:.1f}" if delta >= 0 else f"{delta:.1f}"
            lines.append(
                f"| {rank} | `{r.config_name}` | `{r.source}` | `{r.confidence:.3f}` "
                f"| {r.hits_count} | {r.total_context_chars} | {r.graph.total_nodes} "
                f"| **{r.composite_score:.1f}** | {delta_str} | `{_bar(r.composite_score)}` |"
            )

        lines += ["", "**Combination parameter details:**", ""]
        for r in combo_ranked:
            lines += [
                f"**`{r.config_name}`** — {r.description}  ",
                "",
                "| Parameter | Value | Baseline | Δ |",
                "|-----------|-------|----------|---|",
            ]
            for k in sorted(BASELINE.keys()):
                v = r.config.get(k, BASELINE[k])
                bv = BASELINE[k]
                try:
                    d = f"{v - bv:+.3g}" if v != bv else "—"
                except TypeError:
                    d = "—"
                lines.append(f"| `{k}` | `{v}` | `{bv}` | {d} |")
            lines += [""]

        lines += ["---", ""]

    # ── Full Ranked Results ───────────────────────────────────────────────────
    lines += ["## Full Ranked Results", ""]

    for rank, r in enumerate(ranked, 1):
        lines += [
            f"### #{rank} — `{r.config_name}`",
            "",
            f"> {r.description}",
            "",
        ]

        if r.error:
            lines += [f"**ERROR:** `{r.error}`", ""]

        # Retrieval metrics
        lines += [
            "#### Retrieval Metrics",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Swept Parameter | `{r.sweep_param}` |",
            f"| Swept Value | `{r.sweep_value}` |",
            f"| Source Tier | `{r.source}` |",
            f"| Tier Rank | `{r.source_tier_rank}` / 3 |",
            f"| Confidence Score | `{r.confidence:.4f}` |",
            f"| Hits Retrieved | `{r.hits_count}` |",
            f"| Total Context Chars | `{r.total_context_chars}` |",
            f"| 'Chen Feng' Mentions | `{r.chen_feng_mentions}` |",
            f"| Reconsolidated | `{r.reconsolidated}` |",
            f"| Retrieve Time | `{r.retrieve_s}s` |",
        ]

        # Graph metrics
        g = r.graph
        lines += [
            "",
            "#### Graph Propagation Metrics",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Anchor Nodes Found | `{g.anchor_count}` |",
            f"| Total Nodes Collected | `{g.total_nodes}` |",
            f"| Max Hop Reached | `{g.max_hop_reached}` |",
            f"| Node Distribution | `{_hop_dist(g.nodes_per_hop)}` |",
            f"| Entity Nodes | `{g.entity_count}` |",
            f"| Event Nodes | `{g.event_count}` |",
            f"| Stub Nodes | `{g.stub_count}` ({g.stub_ratio:.1%}) |",
            f"| Unique Node Names | `{g.unique_names}` |",
            f"| Max Prop Score | `{g.max_prop_score:.4f}` |",
            f"| Avg Prop Score | `{g.avg_prop_score:.4f}` |",
            f"| Min Prop Score | `{g.min_prop_score:.4f}` |",
            f"| Avg Intrinsic Score | `{g.avg_intrinsic_score:.4f}` |",
            f"| Graph Metrics Time | `{r.graph_metrics_s}s` |",
        ]
        if g.error:
            lines.append(f"| Graph Error | `{g.error}` |")

        # Config parameters
        lines += [
            "",
            "#### Configuration Parameters",
            "",
            "| Parameter | Value | Baseline | Δ |",
            "|-----------|-------|----------|---|",
        ]
        for k in sorted(r.config.keys()):
            v = r.config[k]
            bv = BASELINE.get(k, "—")
            try:
                d = f"{v - bv:+.3g}" if v != bv else "—"
            except TypeError:
                d = "—"
            lines.append(f"| `{k}` | `{v}` | `{bv}` | {d} |")

        # Score breakdown
        lines += [
            "",
            "#### Composite Score Breakdown",
            "",
            "| Component | Calculation | Points |",
            "|-----------|-------------|--------|",
            f"| Source tier | `{r.source_tier_rank} × 15` | `{r.source_tier_rank * 15:.1f}` |",
            f"| Confidence | `{r.confidence:.3f} × 20` | `{r.confidence * 20:.1f}` |",
            f"| Hits breadth | `min({r.hits_count}, 5) × 3` | `{min(r.hits_count, 5) * 3:.1f}` |",
            f"| Context volume | `min({r.total_context_chars}/100, 15)` | `{min(r.total_context_chars/100, 15):.1f}` |",
            f"| Entity coverage | `min({r.chen_feng_mentions}, 5) × 2` | `{min(r.chen_feng_mentions, 5) * 2:.1f}` |",
            f"| **Total** | | **`{r.composite_score:.1f} / 100`** |",
            f"| Score bar | `{_bar(r.composite_score)}` | |",
        ]

        # Retrieved context
        lines += ["", "#### Retrieved Context", ""]
        if r.hits:
            lines.append("```")
            for j, hit in enumerate(r.hits, 1):
                lines.append(f"=== Hit {j}/{r.hits_count} ===")
                snippet = hit if len(hit) <= 1500 else hit[:1500] + "\n... [truncated]"
                lines.append(snippet)
                lines.append("")
            lines.append("```")
        else:
            lines.append("*No context retrieved — all tiers missed.*")

        # Gemini answer
        lines += ["", "#### Gemini Answer", ""]
        if r.gemini_answer:
            for line in r.gemini_answer.strip().splitlines():
                lines.append(f"> {line}" if line.strip() else ">")
        else:
            lines.append("> *(no answer)*")

        lines += ["", f"*LLM call: {r.llm_s}s*", "", "---", ""]

    # ── Optimal Configuration Recommendation ──────────────────────────────────
    lines += [
        "## Optimal Configuration Recommendation",
        "",
        "Based on the highest composite context-quality score across all runs.",
        "",
    ]
    if ranked:
        opt = ranked[0]
        lines += [
            f"**Best config:** `{opt.config_name}`  ",
            f"**Score:** `{opt.composite_score:.1f} / 100`  ",
            f"**Tier:** `{opt.source}`  ",
            f"**Confidence:** `{opt.confidence:.4f}`  ",
            "",
            "Copy these values into your `.env` file:",
            "",
            "```ini",
        ]
        for k, v in sorted(opt.config.items()):
            lines.append(f"{k}={v}")
        lines += ["```", ""]

        delta = opt.composite_score - baseline_score
        delta_str = f"+{delta:.1f}" if delta >= 0 else f"{delta:.1f}"
        lines += [
            f"This improves composite score by **{delta_str} pts** over baseline (`{baseline_score:.1f}`).",
            "",
        ]

        # Compare to baseline
        lines += [
            "**Parameter delta vs baseline:**",
            "",
            "| Parameter | Optimal | Baseline | Δ |",
            "|-----------|---------|----------|---|",
        ]
        for k in sorted(BASELINE.keys()):
            ov = opt.config.get(k, BASELINE[k])
            bv = BASELINE[k]
            try:
                d = f"{ov - bv:+.3g}" if ov != bv else "— (unchanged)"
            except TypeError:
                d = "—"
            lines.append(f"| `{k}` | `{ov}` | `{bv}` | {d} |")

    lines += ["", "---", ""]

    # ── Appendix ──────────────────────────────────────────────────────────────
    lines += [
        "## Appendix — All Configurations",
        "",
        "Full parameter table sorted by composite score (descending).",
        "",
        "| # | Config | START | CONF | ANCHOR_T | MAX_ST | DECAY | HOP_T | DEPTH | NODES | STUB | SEED_T | Score |",
        "|---|--------|-------|------|---------|--------|-------|-------|-------|-------|------|--------|-------|",
    ]
    for rank, r in enumerate(ranked, 1):
        c = r.config
        lines.append(
            f"| {rank} | `{r.config_name}` "
            f"| {c.get('START_THRESHOLD')} "
            f"| {c.get('CONFIDENT_THRESHOLD')} "
            f"| {c.get('GRAPH_ANCHOR_THRESHOLD')} "
            f"| {c.get('GRAPH_ANCHOR_MAX_STARTS')} "
            f"| {c.get('PROPAGATION_DECAY')} "
            f"| {c.get('PROPAGATION_HOP_THRESHOLD')} "
            f"| {c.get('PROPAGATION_MAX_DEPTH')} "
            f"| {c.get('PROPAGATION_MAX_NODES')} "
            f"| {c.get('STUB_PENALTY')} "
            f"| {c.get('GRAPHRAG_SEED_THRESHOLD')} "
            f"| {r.composite_score:.1f} |"
        )

    lines += [""]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Hippo-Cortex deep retrieval metric benchmark")
    parser.add_argument("--dry-run", action="store_true", help="Skip Gemini calls")
    parser.add_argument("--max-configs", type=int, default=0,
                        help="Cap number of configs (0 = no cap)")
    args = parser.parse_args()

    all_configs = build_configs()
    if args.max_configs and args.max_configs > 0:
        all_configs = all_configs[: args.max_configs]

    oat_n = sum(1 for c in all_configs if c.get("sweep_param") not in ("baseline", "combination"))
    combo_n = sum(1 for c in all_configs if c.get("sweep_param") == "combination")

    sweep_counts = {p: len(v) for p, v in PARAM_SWEEP.items()}
    print("=" * 72)
    print("Hippo-Cortex Deep Retrieval Metric Benchmark")
    print(f"Query         : {QUERY}")
    print(f"Total runs    : {len(all_configs)}  (OAT={oat_n}, combos={combo_n}, baseline=1)")
    print(f"Dry-run       : {args.dry_run}")
    print(f"Params swept  : {len(PARAM_SWEEP)}")
    print(f"Values/param  : {min(sweep_counts.values())}–{max(sweep_counts.values())} "
          f"(avg {sum(sweep_counts.values())/len(sweep_counts.values()):.1f})")
    print("=" * 72)

    results = run_benchmark(all_configs, dry_run=args.dry_run)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = ROOT / "docs" / "sessions" / f"metric_benchmark_{timestamp}.md"
    write_report(results, out_path)

    ranked = sorted(results, key=lambda r: r.composite_score, reverse=True)
    print("\nTop 5:")
    for i, r in enumerate(ranked[:5], 1):
        print(f"  #{i}  {r.config_name:<45}  score={r.composite_score:.1f}  tier={r.source}")


if __name__ == "__main__":
    main()
