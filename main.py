"""
Smoke test and interactive CLI for the hippo-cortex retrieval MVP.

Usage:
    python main.py              # run smoke test, then drop into REPL
    python main.py --smoke      # smoke test only
    python main.py --repl       # REPL only (skip smoke test)
    python main.py --ingest     # demo full pipeline ingest (requires LLM)
"""

import argparse
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)-8s | %(name)-12s | %(message)s",
)

from src.storage.filestore import write_entity
from src.storage.graph import init_graph
from src.storage.memory import init_memory, store_fact
from src.retrieval.chain import RetrievalResult, format_context, retrieve


def run_smoke_test(run_ingest: bool = False) -> bool:
    print("\n=== Smoke Test ===\n")

    print("1. Initializing memory layer (Mem0 + Qdrant)...")
    ok = init_memory()
    if not ok:
        print("  FAIL — memory init failed. Check Qdrant + Ollama are running.")
        print("         See README.md section 13 (Startup Checklist).")
        return False
    print("  OK\n")

    print("2. Initializing graph layer (Kuzu)...")
    ok = init_graph()
    if not ok:
        print("  FAIL — graph init failed. Check data/graph/ is writable.")
        return False
    print("  OK\n")

    # Store facts via Mem0 (Tier 1 store)
    print("3. Storing 3 facts into Mem0...")
    facts = [
        ("Ada Lovelace is considered the first computer programmer", {"subject": "Ada Lovelace"}),
        ("Ada Lovelace wrote the first algorithm for Charles Babbage's Analytical Engine", {"subject": "Ada Lovelace"}),
        ("Charles Babbage designed the Analytical Engine in the 1830s", {"subject": "Charles Babbage"}),
    ]
    for text, meta in facts:
        store_fact(text, meta)
        print(f"  stored: {text[:60]}...")
    print("  OK\n")

    # Write entity to file store (Tier 2 file fallback)
    print("4. Writing entity files to knowledge-store...")
    write_entity("Ada Lovelace", "People", "first computer programmer, 1843")
    write_entity("Charles Babbage", "People", "designed the Analytical Engine, 1830s")
    print("  OK\n")

    # Tier 1 query
    print("5. Querying 'Who is Ada Lovelace?' (expect Tier 1 hit)...")
    result = retrieve("Who is Ada Lovelace?")
    _print_result(result)

    # Tier 1 query
    print("6. Querying 'Analytical Engine inventor' (expect Tier 1 or Tier 2)...")
    result = retrieve("Analytical Engine inventor")
    _print_result(result)

    # Optional: full pipeline ingest demo
    if run_ingest:
        print("7. Running full pipeline ingest (requires LLM)...")
        try:
            from src.ingestion.pipeline import ingest
            sample = (
                "Ada Lovelace, born in 1815, was the daughter of the poet Lord Byron. "
                "She worked with Charles Babbage on the Analytical Engine and wrote "
                "what is considered the first computer algorithm in 1843. "
                "Babbage, a Cambridge mathematician, designed the Analytical Engine "
                "in the 1830s as a general-purpose mechanical computer."
            )
            result_dict = ingest(sample, doc_id=1, title="Ada Lovelace Demo")
            print(f"  segments={result_dict['segments']} "
                  f"events={result_dict['events']} "
                  f"props={result_dict['properties']}")
            if result_dict.get("conflict"):
                print(f"  conflict detected: {result_dict['conflict']}")
            print("  OK\n")

            print("8. Re-querying after ingest (expect Tier 1 or Tier 2 graph hit)...")
            result = retrieve("Ada Lovelace algorithm Babbage")
            _print_result(result)
        except Exception as exc:
            print(f"  SKIP — ingest demo failed: {exc}")
            print("         Check Ollama is running with LIGHTWEIGHT_LLM available.\n")

    print("=== Smoke test complete ===\n")
    return True


def _print_result(result: RetrievalResult) -> None:
    tier_label = {
        "tier1": "TIER 1 (Mem0)",
        "tier1+tier2": "TIER 1+2 (Mem0 + graph)",
        "tier2": "TIER 2 (graph/file)",
        "miss": "MISS",
    }.get(result.source, result.source.upper())
    print(f"  Source : {tier_label}")
    if result.confidence:
        print(f"  Score  : {result.confidence:.3f}")
    if result.reconsolidated:
        print(f"  Action : reconsolidated to Mem0")
    for i, hit in enumerate(result.hits, 1):
        print(f"  Hit {i}  : {hit[:120]}")
    if not result.hits:
        print("  (no results)")
    print()


def run_repl() -> None:
    print("Interactive retrieval REPL. Type 'quit' to exit.\n")
    while True:
        try:
            query = input("query> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not query or query.lower() in ("quit", "exit", "q"):
            break
        result = retrieve(query)
        print(format_context(result))
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Hippo-cortex retrieval MVP")
    parser.add_argument("--smoke", action="store_true", help="Run smoke test only")
    parser.add_argument("--repl", action="store_true", help="Run REPL only")
    parser.add_argument("--ingest", action="store_true", help="Include pipeline ingest demo in smoke test")
    args = parser.parse_args()

    if args.repl:
        ok = init_memory() and init_graph()
        if not ok:
            sys.exit(1)
        run_repl()
        return

    ok = run_smoke_test(run_ingest=args.ingest)
    if not ok:
        sys.exit(1)

    if not args.smoke:
        run_repl()


if __name__ == "__main__":
    main()
