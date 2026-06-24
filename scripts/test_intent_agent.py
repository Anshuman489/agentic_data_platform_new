"""
scripts/test_intent_agent.py — Integration test for IntentAgent using Vertex AI Gemini.

Loads a cached DatasetProfile (or profiles the table live if no cache exists),
then fires several natural-language questions through IntentAgent and prints the
structured IntentResult for each one.

Usage:
    python -m scripts.test_intent_agent --table project.dataset.table

    # Specific questions:
    python -m scripts.test_intent_agent --table project.dataset.table \
        --questions "top 5 countries by revenue" "total sales in 2011"

Environment:
    Reads from .env — at minimum GCP_PROJECT, BQ_LOCATION, VERTEX_LOCATION, LLM_MODEL.
    Authentication via ADC (gcloud auth application-default login).
"""

import argparse
import json
import logging
import sys

from core.bigquery_client import BigQueryClient
from agents.schema_discovery_agent import SchemaDiscoveryAgent
from agents.intent_agent import IntentAgent
from models.intent import IntentResult

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:%(name)s:%(message)s",
)
logger = logging.getLogger(__name__)

# ── Default questions ──────────────────────────────────────────────────────────
# These cover the main intent patterns: aggregation, filtering, grouping, top-N,
# date range. Swap in questions that match your actual table.

DEFAULT_QUESTIONS = [
    "What are the top 5 countries by total quantity sold?",
    "How many distinct customers made a purchase in December 2011?",
    "What is the average unit price per product category?",
    "Show me all transactions from the United Kingdom with quantity greater than 100.",
    "What is the total revenue (unit price × quantity) for each month in 2011?",
]


# ── Display helpers ────────────────────────────────────────────────────────────

def _print_intent(question: str, intent: IntentResult) -> None:
    """Print a single IntentResult in a readable format."""
    print()
    print(f"  Question : {question}")
    print(f"  Reasoning: {intent.reasoning or '—'}")
    print()

    # Metrics
    if intent.metrics:
        metrics_str = ", ".join(
            f"{m.aggregation}({m.column})" for m in intent.metrics
        )
        print(f"  SELECT   : {metrics_str}")
    else:
        print("  SELECT   : * (no aggregation)")

    # Dimensions
    if intent.dimensions:
        print(f"  GROUP BY : {', '.join(intent.dimensions)}")

    # Filters
    if intent.filters:
        for f in intent.filters:
            val = f" {f.value}" if f.value is not None else ""
            print(f"  WHERE    : {f.column} {f.operator}{val}")

    # Time range
    if intent.time_range:
        tr = intent.time_range
        print(f"  TIME     : {tr.column} BETWEEN {tr.start or '—'} AND {tr.end or '—'}")

    # Order / limit
    if intent.order_by:
        order_str = ", ".join(f"{o.column} {o.direction}" for o in intent.order_by)
        print(f"  ORDER BY : {order_str}")
    if intent.limit is not None:
        print(f"  LIMIT    : {intent.limit}")

    print()
    print("  " + "-" * 60)

    # Raw JSON (useful for debugging)
    print("  Raw JSON:")
    raw = json.loads(intent.model_dump_json(indent=2))
    for line in json.dumps(raw, indent=2).splitlines():
        print(f"    {line}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="test_intent_agent",
        description="Test IntentAgent against a real BigQuery table.",
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
        help="Questions to ask (defaults to a built-in set if omitted).",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    questions = args.questions or DEFAULT_QUESTIONS

    # ── Step 1: get the DatasetProfile ────────────────────────────────────────
    print(f"\nLoading profile for: {args.table}")
    bq = BigQueryClient()
    discovery = SchemaDiscoveryAgent(bq)
    try:
        profile = discovery.run(args.table)
    except Exception as exc:
        logger.error("Failed to profile table '%s': %s", args.table, exc)
        sys.exit(1)

    print(f"  {len(profile.columns)} columns, {profile.row_count:,} rows")
    print(f"  Columns: {', '.join(c.name for c in profile.columns)}")

    # ── Step 2: run each question through IntentAgent ─────────────────────────
    print(f"\nRunning {len(questions)} question(s) through IntentAgent …")
    from config.settings import settings
    print(f"  Model: {settings.llm_model}  Region: {settings.vertex_location}\n")
    print("=" * 64)

    agent = IntentAgent()
    success = 0
    for i, question in enumerate(questions, 1):
        print(f"\n[{i}/{len(questions)}]")
        try:
            intent = agent.run(question, profile)
            _print_intent(question, intent)
            success += 1
        except Exception as exc:
            print(f"  ERROR: {exc}")
            logger.exception("IntentAgent failed for question: %s", question)

    print(f"\nDone: {success}/{len(questions)} questions parsed successfully.")


if __name__ == "__main__":
    main()
