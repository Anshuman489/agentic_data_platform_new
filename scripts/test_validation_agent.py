"""
scripts/test_validation_agent.py — End-to-end test for the full pipeline.

Chains: IntentAgent → SqlGenerationAgent → ValidationAgent

For each question, prints:
  • The generated SQL
  • Syntax check result
  • Semantic check result (approved / feedback)
  • Query rows (if both layers passed)

Usage:
    python -m scripts.test_validation_agent --table project.dataset.table
    python -m scripts.test_validation_agent --table project.dataset.table \\
        --questions "top 5 countries by revenue" "sales by month in 2011"
"""

import argparse
import logging
import sys
from decimal import Decimal

from core.bigquery_client import BigQueryClient
from agents.schema_discovery_agent import SchemaDiscoveryAgent
from agents.sql_generation_agent import SqlGenerationAgent
from agents.validation_agent import ValidationAgent
from agents.pipeline import run_pipeline
from models.validation import ValidationResult

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s:%(name)s:%(message)s",
)

DEFAULT_QUESTIONS = [
    "What is the total fare amount collected per payment type?",
    "How many trips were taken each month in 2022?",
    "What is the average tip amount per vendor?",
    "Show me all trips where the total amount is greater than 200 dollars.",
    "Which pickup location had the highest number of trips?",
]


# ── Display ────────────────────────────────────────────────────────────────────

def _print_result(i: int, total: int, question: str, result: ValidationResult) -> None:
    width = 64
    print(f"\n{'=' * width}")
    print(f"[{i}/{total}] {question}")
    print(f"{'=' * width}")
    print(f"\nSQL:\n{result.sql}")

    print(f"\nValidation:")
    syntax_icon = "OK" if result.syntax_valid else "FAIL"
    print(f"  Syntax   [{syntax_icon}]", end="")
    if result.syntax_error:
        print(f"  {result.syntax_error[:100]}")
    else:
        print()

    if result.semantic_valid is not None:
        sem_icon = "OK" if result.semantic_valid else "FAIL"
        print(f"  Semantic [{sem_icon}]  {result.semantic_feedback or ''}")

    if result.passed and result.rows:
        print(f"\nResults ({result.total_rows} rows):")
        _print_rows(result.rows)
    elif result.passed and result.total_rows == 0:
        print("\n  (query returned 0 rows)")


def _print_rows(rows: list[dict], max_rows: int = 10) -> None:
    if not rows:
        return

    # Collect columns and compute widths
    cols = list(rows[0].keys())
    widths = {c: max(len(c), *(len(_fmt(r.get(c))) for r in rows[:max_rows])) for c in cols}

    header = "  " + "  ".join(c.ljust(widths[c]) for c in cols)
    print(header)
    print("  " + "  ".join("-" * widths[c] for c in cols))
    for row in rows[:max_rows]:
        print("  " + "  ".join(_fmt(row.get(c)).ljust(widths[c]) for c in cols))
    if len(rows) > max_rows:
        print(f"  … {len(rows) - max_rows} more rows")


def _fmt(value) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, float):
        return f"{value:,.2f}"
    if isinstance(value, Decimal):
        return f"{float(value):,.2f}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


# ── CLI ────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="test_validation_agent",
        description="Full pipeline test: Intent → SQL → Validate → Execute.",
    )
    parser.add_argument("--table", required=True, metavar="PROJECT.DATASET.TABLE")
    parser.add_argument("--questions", nargs="+", default=None, metavar="QUESTION")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    questions = args.questions or DEFAULT_QUESTIONS

    print(f"\nLoading profile for: {args.table}")
    bq = BigQueryClient()
    profile = SchemaDiscoveryAgent(bq).run(args.table)
    print(f"  {len(profile.columns)} columns — {profile.row_count:,} rows")

    sql_agent = SqlGenerationAgent()
    val_agent = ValidationAgent(bq)

    passed = 0
    for i, question in enumerate(questions, 1):
        try:
            result = run_pipeline(
                question=question,
                profile=profile,
                sql_agent=sql_agent,
                val_agent=val_agent,
            )
            _print_result(i, len(questions), question, result)
            if result.passed:
                passed += 1
        except Exception as exc:
            print(f"\n[{i}/{len(questions)}] PIPELINE ERROR: {exc}")
            logging.exception("Pipeline failed for: %s", question)

    print(f"\n\nPassed: {passed}/{len(questions)} questions through all validation layers.")


if __name__ == "__main__":
    main()
