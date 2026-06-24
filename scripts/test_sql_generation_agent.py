"""
scripts/test_sql_generation_agent.py — End-to-end test for the intent → SQL pipeline.

Chains IntentAgent → SqlGenerationAgent for each test question and prints
the generated SQL alongside the structured intent that produced it.

Usage:
    python -m scripts.test_sql_generation_agent --table project.dataset.table

    # Custom questions:
    python -m scripts.test_sql_generation_agent --table project.dataset.table \\
        --questions "top 10 customers by spend" "sales by country in 2011"
"""

import argparse
import logging
import sys

from core.bigquery_client import BigQueryClient
from agents.schema_discovery_agent import SchemaDiscoveryAgent
from agents.intent_agent import IntentAgent
from agents.sql_generation_agent import SqlGenerationAgent

logging.basicConfig(
    level=logging.WARNING,        # quiet — we want clean SQL output
    format="%(levelname)s:%(name)s:%(message)s",
)

DEFAULT_QUESTIONS = [
    "What are the top 5 countries by total quantity sold?",
    "How many distinct customers made a purchase in December 2011?",
    "Show me all transactions from the United Kingdom with quantity greater than 100.",
    "What is the total revenue (unit price × quantity) for each month in 2011?",
    "Which products were sold in Germany or France?",
]


def _print_result(i: int, total: int, question: str, sql: str) -> None:
    print(f"\n{'='*64}")
    print(f"[{i}/{total}] {question}")
    print(f"{'='*64}")
    print(sql)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="test_sql_generation_agent",
        description="Test IntentAgent → SqlGenerationAgent pipeline.",
    )
    parser.add_argument(
        "--table",
        required=True,
        metavar="PROJECT.DATASET.TABLE",
        help="Fully-qualified BigQuery table reference.",
    )
    parser.add_argument(
        "--questions",
        nargs="+",
        default=None,
        metavar="QUESTION",
        help="Questions to test (uses built-in set if omitted).",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    questions = args.questions or DEFAULT_QUESTIONS

    # ── Step 1: load profile ──────────────────────────────────────────────────
    print(f"\nLoading profile for: {args.table}")
    bq = BigQueryClient()
    profile = SchemaDiscoveryAgent(bq).run(args.table)
    print(f"  {len(profile.columns)} columns — {', '.join(c.name for c in profile.columns)}")

    intent_agent = IntentAgent()
    sql_agent = SqlGenerationAgent()

    success = 0
    for i, question in enumerate(questions, 1):
        try:
            intent = intent_agent.run(question, profile)
            sql = sql_agent.run(intent, profile)
            _print_result(i, len(questions), question, sql)
            success += 1
        except Exception as exc:
            print(f"\n[{i}/{len(questions)}] ERROR: {question}")
            print(f"  {exc}")
            logging.exception("Pipeline failed")

    print(f"\n\nDone: {success}/{len(questions)} SQL statements generated.")


if __name__ == "__main__":
    main()
