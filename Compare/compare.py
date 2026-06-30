#!/usr/bin/env python3
"""
compare.py — SOTA Retrieval Methods Comparison

Runs all implemented retrieval strategies against samples/Qs/Qs.json and
produces a side-by-side metrics table plus an optional markdown report.

Methods compared:
  dense       — Qdrant dense semantic search (Tier 1, memory collection)
  rag_chunks  — Standard RAG over sliding-window chunks (tsg_rag collection)
  graph_prop  — Kuzu graph BFS propagation with decay scoring (Tier 2)
  graph_rag   — Entity seed + BFS over graph.json (Graph-RAG variant)
  filestore   — Keyword grep over markdown knowledge-store files (Tier 3)
  cascade     — Full three-tier pipeline (Tier 1 -> Tier 2 -> Tier 3)

Metrics per method (per question, then averaged):
  answer_hit  — Token recall ≥ 0.6 of expected_answer tokens in context
  anchor_hit  — retrieval_anchor substring found verbatim in context
  token_f1    — Token-level F1 between context and expected_answer
  context_len — Total chars retrieved
  latency_ms  — Wall-clock ms per query

Composite score (0–100):
  40 × answer_hit_rate + 30 × anchor_hit_rate + 20 × avg_token_f1
  + 10 × (1 − error_rate)

Usage:
  python compare.py                              # all methods, all 51 questions
  python compare.py --methods dense cascade      # specific methods only
  python compare.py --limit 10                   # first N questions (quick test)
  python compare.py --out docs/sessions/         # also save markdown report
  python compare.py --no-init                    # skip storage init (faster if already running)
  python compare.py --top-k 3                    # retrieve fewer results per query
"""

import argparse
import json
import logging
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

# Allow `src.*` imports when run as `python Compare/compare.py` from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Suppress noisy retrieval/embedding logs during benchmarking
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

QS_PATH = Path("samples/Qs/Qs.json")

# Stop words excluded when computing answer_hit token recall
_STOP: frozenset[str] = frozenset({
    "the", "a", "an", "is", "was", "were", "are", "and", "or", "to", "of",
    "in", "it", "he", "she", "they", "his", "her", "their", "this", "that",
    "with", "for", "on", "at", "by", "from", "as", "had", "have", "has",
    "be", "been", "being", "what", "who", "how", "when", "where", "which",
    "s", "not", "but", "so", "if", "then", "than", "into",
})


# --- Data types ---------------------------------------------------------------

@dataclass
class QueryResult:
    method: str
    question: str
    expected_answer: str
    retrieval_anchor: str
    question_type: str
    context: str
    latency_ms: float
    error: str = ""

    @property
    def answer_hit(self) -> bool:
        """True if ≥60% of key answer tokens appear in context."""
        return _token_recall(self.context, self.expected_answer) >= 0.6

    @property
    def anchor_hit(self) -> bool:
        """Exact substring match for the ground-truth retrieval anchor."""
        if not self.retrieval_anchor:
            return False
        return self.retrieval_anchor.lower() in self.context.lower()

    @property
    def token_f1(self) -> float:
        return _token_f1(self.context, self.expected_answer)

    @property
    def context_len(self) -> int:
        return len(self.context)


# --- Metric helpers -----------------------------------------------------------

def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"\b\w+\b", text.lower()))


def _token_recall(prediction: str, ground_truth: str) -> float:
    pred = _tokenize(prediction)
    gt = _tokenize(ground_truth)
    gt_key = gt - _STOP or gt          # fall back to full set if all stop words
    if not gt_key:
        return 0.0
    return len(pred & gt_key) / len(gt_key)


def _token_f1(prediction: str, ground_truth: str) -> float:
    pred = _tokenize(prediction)
    gt = _tokenize(ground_truth)
    if not pred or not gt:
        return 0.0
    common = pred & gt
    precision = len(common) / len(pred)
    recall = len(common) / len(gt)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# --- Retrieval wrappers -------------------------------------------------------
# Each function accepts (query: str, top_k: int) and returns the retrieved
# context as a plain string. Empty string = no results / method unavailable.

def _retrieve_dense(query: str, top_k: int = 5) -> str:
    from src.storage.memory import search_knowledge
    results = search_knowledge(query, top_k=top_k)
    return "\n---\n".join(r["memory"] for r in results if r.get("memory"))


def _retrieve_rag(query: str, top_k: int = 5) -> str:
    from src.retrieval.rag import retrieve_rag
    hits = retrieve_rag(query, top_k=top_k)
    return "\n---\n".join(h["text"] for h in hits if h.get("text"))


def _retrieve_graph_prop(query: str, top_k: int = 5) -> str:  # top_k unused (graph controls breadth)
    from src.storage.graph import is_initialized
    from src.retrieval.propagator import (
        PropagationResult,
        find_start_nodes,
        format_propagation_result,
        propagate,
    )

    if not is_initialized():
        return ""

    starts = find_start_nodes(query)
    if not starts:
        return ""

    seen: dict = {}
    for start in starts:
        result = propagate(
            query,
            start["node_id"],
            start["node_type"],
            start["name"],
            start["content"],
            start_is_stub=start.get("is_stub", False),
        )
        for node in result.nodes:
            prev = seen.get(node.node_id)
            if prev is None or node.prop_score > prev.prop_score:
                seen[node.node_id] = node

    if not seen:
        return ""

    merged = sorted(seen.values(), key=lambda n: n.prop_score, reverse=True)
    combined = PropagationResult(
        nodes=merged,
        start_entity=", ".join(s["name"] for s in starts),
        max_hop=max(n.hop for n in merged),
    )
    return format_propagation_result(combined)


def _retrieve_graph_rag(query: str, top_k: int = 5) -> str:
    from src.retrieval.graph_rag import retrieve_graph_rag
    result = retrieve_graph_rag(query, top_k=top_k)
    return result.get("context", "")


def _retrieve_filestore(query: str, top_k: int = 5) -> str:  # top_k unused
    from src.storage.filestore import grep_filestore
    return grep_filestore(query) or ""


def _retrieve_cascade(query: str, top_k: int = 5) -> str:  # top_k unused; chain uses its own defaults
    from src.retrieval.chain import retrieve
    result = retrieve(query)
    return "\n---\n".join(result.hits)


METHODS: dict[str, Callable[[str, int], str]] = {
    "dense":      _retrieve_dense,
    "rag_chunks": _retrieve_rag,
    "graph_prop": _retrieve_graph_prop,
    "graph_rag":  _retrieve_graph_rag,
    "filestore":  _retrieve_filestore,
    "cascade":    _retrieve_cascade,
}

METHOD_DESCRIPTIONS = {
    "dense":      "Qdrant dense semantic search (Tier 1)",
    "rag_chunks": "Standard RAG — sliding-window chunks (tsg_rag)",
    "graph_prop": "Kuzu graph BFS propagation with decay scoring (Tier 2)",
    "graph_rag":  "Entity seed + BFS over graph.json (Graph-RAG)",
    "filestore":  "Keyword grep over markdown knowledge-store (Tier 3)",
    "cascade":    "Full three-tier pipeline (Tier 1 -> Tier 2 -> Tier 3)",
}


# --- Storage init -------------------------------------------------------------

def _init_storage() -> None:
    print("Initializing storage services...")
    try:
        from src.storage.memory import init_memory
        ok = init_memory()
        if ok:
            print("  [ok] Mem0 + Qdrant (dense, rag_chunks, cascade)")
        else:
            print("  [WARN] Qdrant init failed — dense/rag_chunks/cascade will error")
    except Exception as e:
        print(f"  [WARN] Memory init error: {e}")

    try:
        from src.storage.graph import init_graph
        init_graph()
        print("  [ok] Kuzu graph (graph_prop)")
    except Exception as e:
        print(f"  [WARN] Graph init failed ({e}) — graph_prop will return empty results")

    print()


# --- Evaluation ---------------------------------------------------------------

def run_method(
    method_name: str,
    fn: Callable[[str, int], str],
    questions: list[dict],
    top_k: int,
) -> list[QueryResult]:
    results = []
    for q in questions:
        t0 = time.perf_counter()
        error = ""
        try:
            context = fn(q["question"], top_k)
        except Exception as exc:
            context = ""
            error = str(exc)[:200]
        latency_ms = (time.perf_counter() - t0) * 1000

        results.append(QueryResult(
            method=method_name,
            question=q["question"],
            expected_answer=q["expected_answer"],
            retrieval_anchor=q["retrieval_anchor"],
            question_type=q["question_type"],
            context=context,
            latency_ms=round(latency_ms, 2),
            error=error,
        ))
    return results


def aggregate(results: list[QueryResult]) -> dict:
    if not results:
        return {}
    n = len(results)
    answer_hits = sum(r.answer_hit for r in results)
    anchor_hits = sum(r.anchor_hit for r in results)
    avg_f1 = sum(r.token_f1 for r in results) / n
    avg_ctx = sum(r.context_len for r in results) / n
    avg_lat = sum(r.latency_ms for r in results) / n
    error_rate = sum(bool(r.error) for r in results) / n

    composite = (
        40 * answer_hits / n
        + 30 * anchor_hits / n
        + 20 * avg_f1
        + 10 * (1 - error_rate)
    )

    return {
        "n": n,
        "answer_hits": answer_hits,
        "answer_hit_rate": answer_hits / n,
        "anchor_hits": anchor_hits,
        "anchor_hit_rate": anchor_hits / n,
        "avg_token_f1": avg_f1,
        "avg_context_len": avg_ctx,
        "avg_latency_ms": avg_lat,
        "error_rate": error_rate,
        "composite": round(composite, 2),
    }


def aggregate_by_type(results: list[QueryResult]) -> dict[str, dict]:
    by_type: dict[str, list] = {}
    for r in results:
        by_type.setdefault(r.question_type, []).append(r)
    return {qt: aggregate(rs) for qt, rs in sorted(by_type.items())}


# --- Console output -----------------------------------------------------------

def _pct(v: float) -> str:
    return f"{v:.1%}"


def print_summary_table(all_results: dict[str, list[QueryResult]]) -> None:
    header = (
        f"{'Method':<14} {'Score':>6} {'AnsHit':>7} {'AncHit':>7} "
        f"{'TokF1':>6} {'CtxLen':>7} {'Lat(ms)':>8} {'Err':>5}"
    )
    sep = "-" * len(header)
    print(header)
    print(sep)

    rows = []
    for method, results in all_results.items():
        s = aggregate(results)
        if not s:
            continue
        rows.append((s["composite"], method, s))

    for _, method, s in sorted(rows, reverse=True):
        print(
            f"{method:<14} {s['composite']:>6.1f} "
            f"{_pct(s['answer_hit_rate']):>7} {_pct(s['anchor_hit_rate']):>7} "
            f"{s['avg_token_f1']:>6.3f} {s['avg_context_len']:>7.0f} "
            f"{s['avg_latency_ms']:>8.1f} {_pct(s['error_rate']):>5}"
        )

    print()
    print("  Score = 40×AnsHit + 30×AncHit + 20×TokenF1 + 10×(1-ErrRate)  (0–100)")
    print()


def print_qtype_table(all_results: dict[str, list[QueryResult]]) -> None:
    all_types = sorted({r.question_type for results in all_results.values() for r in results})

    # Abbreviate long type names for the table
    abbrev = {qt: qt[:16] for qt in all_types}

    col = 16
    type_header = " ".join(f"{abbrev[qt]:>{col}}" for qt in all_types)
    header = f"{'Method':<14}  {type_header}"
    print(header)
    print("-" * len(header))

    for method, results in all_results.items():
        by_type = aggregate_by_type(results)
        cells = " ".join(
            f"{_pct(by_type.get(qt, {}).get('answer_hit_rate', 0)):>{col}}"
            for qt in all_types
        )
        print(f"{method:<14}  {cells}")

    print()


# --- Markdown report ----------------------------------------------------------

def build_markdown_report(
    all_results: dict[str, list[QueryResult]],
    questions: list[dict],
    methods_run: list[str],
    top_k: int,
) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        f"# Retrieval Methods Comparison — {now}",
        "",
        f"**Questions:** {len(questions)}  |  **Methods:** {', '.join(methods_run)}  |  **top_k:** {top_k}",
        "",
        "## Method Descriptions",
        "",
    ]
    for m in methods_run:
        lines.append(f"- **{m}**: {METHOD_DESCRIPTIONS.get(m, '')}")

    lines += [
        "",
        "## Summary Metrics",
        "",
        "Sorted by composite score: `40×AnsHit + 30×AncHit + 20×TokenF1 + 10×(1−ErrRate)`",
        "",
        "| Method | Score | AnsHit | AncHit | Token F1 | Avg Ctx | Lat (ms) | Errors |",
        "|--------|------:|:------:|:------:|:--------:|--------:|---------:|:------:|",
    ]

    rows = []
    for method in methods_run:
        s = aggregate(all_results[method])
        if s:
            rows.append((s["composite"], method, s))

    for _, method, s in sorted(rows, reverse=True):
        lines.append(
            f"| {method} | {s['composite']:.1f} | {_pct(s['answer_hit_rate'])} "
            f"| {_pct(s['anchor_hit_rate'])} | {s['avg_token_f1']:.3f} "
            f"| {s['avg_context_len']:.0f} | {s['avg_latency_ms']:.1f} "
            f"| {_pct(s['error_rate'])} |"
        )

    # Per question type
    all_types = sorted({r.question_type for results in all_results.values() for r in results})
    lines += ["", "## Answer Hit Rate by Question Type", ""]
    lines.append("| Method | " + " | ".join(f"`{t}`" for t in all_types) + " |")
    lines.append("|--------|" + "|".join(":------:" for _ in all_types) + "|")

    for method in methods_run:
        by_type = aggregate_by_type(all_results[method])
        cells = " | ".join(
            _pct(by_type.get(qt, {}).get("answer_hit_rate", 0)) for qt in all_types
        )
        lines.append(f"| {method} | {cells} |")

    # Per-query detail
    lines += ["", "## Per-Query Results", ""]
    lines.append(
        "| # | Type | Question | " +
        " | ".join(f"AHit({m})" for m in methods_run) + " |"
    )
    lines.append(
        "|---|------|----------|" + "|".join(":---:" for _ in methods_run) + "|"
    )

    for i, q in enumerate(questions):
        qtext = q["question"][:70].replace("|", "\\|")
        hits = [
            ("✓" if all_results[m][i].answer_hit else "✗") for m in methods_run
        ]
        lines.append(
            f"| {i+1} | {q['question_type'][:20]} | {qtext} | " +
            " | ".join(hits) + " |"
        )

    # Errors
    errors_found = [
        (m, i, r.error)
        for m in methods_run
        for i, r in enumerate(all_results[m])
        if r.error
    ]
    if errors_found:
        lines += ["", "## Errors", ""]
        for method, idx, err in errors_found:
            lines.append(f"- **{method}** Q{idx+1}: `{err}`")

    return "\n".join(lines)


# --- Main ---------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare SOTA retrieval methods on samples/Qs/Qs.json"
    )
    parser.add_argument(
        "--methods", nargs="+", choices=list(METHODS.keys()),
        default=list(METHODS.keys()),
        help="Methods to evaluate (default: all)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Evaluate only the first N questions",
    )
    parser.add_argument(
        "--top-k", type=int, default=5, dest="top_k",
        help="Number of results to retrieve per query (default: 5)",
    )
    parser.add_argument(
        "--out", type=str, default=None,
        help="Directory to write a markdown report (e.g. docs/sessions/)",
    )
    parser.add_argument(
        "--no-init", action="store_true",
        help="Skip storage initialization (use when services are already running)",
    )
    args = parser.parse_args()

    if not QS_PATH.exists():
        print(f"ERROR: Question file not found: {QS_PATH}", file=sys.stderr)
        sys.exit(1)

    questions: list[dict] = json.loads(QS_PATH.read_text(encoding="utf-8"))
    if args.limit:
        questions = questions[: args.limit]

    print(f"Questions : {len(questions)} (from {QS_PATH})")
    print(f"Methods   : {', '.join(args.methods)}")
    print(f"top_k     : {args.top_k}")
    print()

    if not args.no_init:
        _init_storage()

    all_results: dict[str, list[QueryResult]] = {}

    for method in args.methods:
        fn = METHODS[method]
        print(f"  {method:<14}", end=" ", flush=True)
        t_start = time.perf_counter()

        results = run_method(method, fn, questions, args.top_k)

        elapsed = time.perf_counter() - t_start
        s = aggregate(results)
        errors = sum(bool(r.error) for r in results)

        print(
            f"[{elapsed:5.1f}s]  "
            f"AnsHit={_pct(s['answer_hit_rate'])}  "
            f"AncHit={_pct(s['anchor_hit_rate'])}  "
            f"F1={s['avg_token_f1']:.3f}  "
            f"Score={s['composite']:.1f}"
            + (f"  ERRORS={errors}" if errors else "")
        )
        all_results[method] = results

    print()
    print("=" * 80)
    print("SUMMARY (sorted by composite score)")
    print("=" * 80)
    print()
    print_summary_table(all_results)

    print("-- Answer Hit Rate by Question Type ----------------------------------------")
    print()
    print_qtype_table(all_results)

    if args.out:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = out_dir / f"compare_{ts}.md"
        report = build_markdown_report(all_results, questions, args.methods, args.top_k)
        report_path.write_text(report, encoding="utf-8")
        print(f"Report saved -> {report_path}")


if __name__ == "__main__":
    main()
