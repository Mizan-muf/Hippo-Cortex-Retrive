"""
Generate test_data/hippo_questions.json from The Strongest Gene chapters 1-50.

Reads pre-extracted XHTML files from samples/epub_extracted/OEBPS/Text/,
strips HTML, prompts the configured LLM (Ollama or Gemini) for 3 questions per
chapter, and saves them in hippo_questions format.

Usage:
    python scripts/generate_questions.py
    python scripts/generate_questions.py --chapters 1-10
    python scripts/generate_questions.py --chapters 11-25 --append
    python scripts/generate_questions.py --output path/to/custom.json
"""

import argparse
import json
import logging
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.llm import call

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

EPUB_TEXT_DIR = Path("samples/epub_extracted/OEBPS/Text")
OUTPUT_FILE = Path("test_data/hippo_questions.json")
CHAPTER_TEXT_LIMIT = 4000

VALID_TYPES = {
    "attribute_recall",
    "event_participant",
    "temporal_causal",
    "entity_relationship",
    "event_outcome",
    "event_location",
    "cross_entity",
}

_PROMPT = """\
You are building an evaluation dataset for a knowledge retrieval system.

Read the chapter text below from "The Strongest Gene" and generate exactly 3 questions.
Every question must be answerable directly and specifically from THIS chapter alone.

CHAPTER {num}:
{text}

Output a JSON array of exactly 3 objects. Each object must have these fields:
  "question"        - a specific, answerable factual question about this chapter
  "expected_answer" - a short answer (1-3 sentences) containing the key facts
  "retrieval_anchor" - a verbatim 2-6 word phrase from the chapter text that uniquely identifies the answer location
  "question_type"   - one of: attribute_recall, event_participant, temporal_causal, entity_relationship, event_outcome, event_location

Rules:
- Use a different question_type for each of the 3 questions where possible
- Questions must be specific ("What ability did Chen Feng activate?" not "What happened?")
- expected_answer must contain the actual facts, not vague paraphrases
- retrieval_anchor must appear verbatim in the chapter text
- Do NOT ask about translator or editor credits
- Return ONLY the JSON array with no explanation, no markdown fences

JSON:"""


def _strip_html(html: str) -> str:
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"\s+", " ", text).strip()
    # drop translator/editor credit line at the top
    text = re.sub(r"Chapter \d+:.*?Translator:.*?Editor:\S+\s*", "", text)
    return text


def _load_chapter(num: int) -> str | None:
    path = EPUB_TEXT_DIR / f"{num:04d}_Chapter__{num}.xhtml"
    if not path.exists():
        return None
    raw = path.read_text(encoding="utf-8")
    return _strip_html(raw)[:CHAPTER_TEXT_LIMIT]


def _parse_response(response: str, chapter_num: int) -> list[dict]:
    match = re.search(r"\[.*?\]", response, re.DOTALL)
    if not match:
        logger.warning("chapter %02d | no JSON array in response", chapter_num)
        return []
    try:
        items = json.loads(match.group())
    except json.JSONDecodeError as e:
        logger.warning("chapter %02d | JSON parse error: %s", chapter_num, e)
        return []

    required = {"question", "expected_answer", "retrieval_anchor", "question_type"}
    valid = []
    for item in items:
        if not isinstance(item, dict):
            continue
        if not required.issubset(item):
            logger.warning("chapter %02d | question missing fields, skipping", chapter_num)
            continue
        if item["question_type"] not in VALID_TYPES:
            item["question_type"] = "attribute_recall"
        item["source_doc_id"] = chapter_num - 1  # 0-indexed to match ingest pipeline
        valid.append(item)
    return valid


def generate(chapter_nums: list[int]) -> list[dict]:
    results = []
    total = len(chapter_nums)
    for i, num in enumerate(chapter_nums, 1):
        logger.info("[%d/%d] chapter %02d", i, total, num)
        text = _load_chapter(num)
        if not text:
            logger.warning("chapter %02d | file not found — skipping", num)
            continue
        prompt = _PROMPT.format(num=num, text=text)
        try:
            response = call(prompt, temperature=0.2)
            questions = _parse_response(response, num)
            logger.info("chapter %02d | %d questions generated", num, len(questions))
            results.extend(questions)
        except Exception as exc:
            logger.error("chapter %02d | LLM call failed: %s", num, exc)
    return results


def _parse_range(s: str) -> list[int]:
    m = re.fullmatch(r"(\d+)-(\d+)", s.strip())
    if not m:
        raise argparse.ArgumentTypeError(f"chapter range must be START-END, got: {s!r}")
    start, end = int(m.group(1)), int(m.group(2))
    if start > end:
        raise argparse.ArgumentTypeError("START must be <= END")
    return list(range(start, end + 1))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate hippo_questions.json from EPUB chapters"
    )
    parser.add_argument(
        "--chapters", default="1-50", type=str,
        help="Chapter range e.g. 1-50 or 10-20 (default: 1-50)",
    )
    parser.add_argument(
        "--output", default=str(OUTPUT_FILE), type=str,
        help="Output JSON file path",
    )
    parser.add_argument(
        "--append", action="store_true",
        help="Append to existing output file instead of overwriting",
    )
    args = parser.parse_args()

    try:
        chapter_nums = _parse_range(args.chapters)
    except argparse.ArgumentTypeError as e:
        logger.error("%s", e)
        sys.exit(1)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    existing: list[dict] = []
    if args.append and output_path.exists():
        try:
            existing = json.loads(output_path.read_text(encoding="utf-8"))
            logger.info("append mode | loaded %d existing questions", len(existing))
        except Exception as e:
            logger.error("failed to load existing file: %s", e)
            sys.exit(1)

    new_questions = generate(chapter_nums)
    combined = existing + new_questions

    output_path.write_text(
        json.dumps(combined, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\n--- Summary ---")
    print(f"Chapters processed : {chapter_nums[0]}–{chapter_nums[-1]}")
    print(f"Questions generated: {len(new_questions)}")
    print(f"Total in file      : {len(combined)}")
    print(f"Output             : {output_path}")
    if combined:
        print("\nBy type:")
        for qtype, count in sorted(Counter(q["question_type"] for q in combined).items()):
            print(f"  {qtype:<22} {count}")


if __name__ == "__main__":
    main()
