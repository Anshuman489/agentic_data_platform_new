"""
main.py — CLI entry point for the Agentic Data Intelligence Platform.

Usage:
    # Profile a table (schema discovery):
    python main.py --table project.dataset.table

    # Ask a natural-language question against a table:
    python main.py --table project.dataset.table --ask "total revenue by country"

    # Show the schema profile even when asking a question:
    python main.py --table project.dataset.table --ask "..." --profile
"""

import argparse
import logging
import sys
from decimal import Decimal

from core.bigquery_client import BigQueryClient
from agents.schema_discovery_agent import SchemaDiscoveryAgent
from agents.intent_agent import IntentAgent
from agents.sql_generation_agent import SqlGenerationAgent
from agents.validation_agent import ValidationAgent
from agents.pipeline import run_pipeline
from models.dataset import DatasetProfile
from models.validation import ValidationResult

logger = logging.getLogger(__name__)


# ── Helpers ────────────────────────────────────────────────────────────────────

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


def _format_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


# ── Profile display ────────────────────────────────────────────────────────────

def _print_profile(profile: DatasetProfile) -> None:
    print()
    print(f"  Table   : {profile.table_ref}")
    if profile.table_description:
        print(f"  Desc    : {profile.table_description}")
    print(f"  Rows    : {profile.row_count:,}")
    print(f"  Size    : {_format_bytes(profile.size_bytes)}")
    print(f"  Columns : {len(profile.columns)}")
    print(f"  Profiled: {profile.profiled_at.strftime('%Y-%m-%d %H:%M UTC')}")
    print()

    print(f"  {'NAME':<24} {'TYPE':<12} {'ROLE':<12} {'CARDINALITY':>12}  {'NULLS':>6}  RANGE / SAMPLES")
    print("  " + "-" * 90)

    for col in profile.columns:
        cardinality = f"{col.cardinality:,}" if col.cardinality is not None else "-"
        nulls = f"{col.null_fraction * 100:.1f}%"

        extra = ""
        if col.numeric_min is not None and col.numeric_max is not None:
            extra = f"{col.numeric_min:g} -> {col.numeric_max:g}"
        elif col.date_min and col.date_max:
            extra = f"{col.date_min} -> {col.date_max}"
        elif col.sample_values:
            samples = [str(v) for v in col.sample_values[:3]]
            extra = ", ".join(samples)
            if len(col.sample_values) > 3:
                extra += ", ..."

        print(
            f"  {col.name:<24} {col.bq_type:<12} {col.inferred_role:<12} "
            f"{cardinality:>12}  {nulls:>6}  {extra}"
        )

    print()


# ── Query result display ───────────────────────────────────────────────────────

def _print_rows(rows: list[dict], max_rows: int = 10) -> None:
    if not rows:
        return
    cols = list(rows[0].keys())
    widths = {c: max(len(c), *(len(_fmt(r.get(c))) for r in rows[:max_rows])) for c in cols}

    print("  " + "  ".join(c.ljust(widths[c]) for c in cols))
    print("  " + "  ".join("-" * widths[c] for c in cols))
    for row in rows[:max_rows]:
        print("  " + "  ".join(_fmt(row.get(c)).ljust(widths[c]) for c in cols))
    if len(rows) > max_rows:
        print(f"  ... {len(rows) - max_rows} more rows")


def _print_query_result(question: str, result: ValidationResult) -> None:
    width = 64
    print(f"\n{'=' * width}")
    print(f"Question: {question}")
    print(f"{'=' * width}")

    print(f"\nSQL:\n{result.sql}")
    print(f"\nValidation:")

    syntax_icon = "OK" if result.syntax_valid else "FAIL"
    print(f"  Syntax   [{syntax_icon}]", end="")
    if result.syntax_error:
        print(f"  {result.syntax_error[:120]}")
    else:
        print()

    if result.semantic_valid is not None:
        sem_icon = "OK" if result.semantic_valid else "FAIL"
        print(f"  Semantic [{sem_icon}]  {result.semantic_feedback or ''}")

    if not result.passed:
        print("\nQuery did not pass validation — no results returned.")
        return

    if result.rows:
        print(f"\nResults ({result.total_rows} rows):")
        _print_rows(result.rows)
    else:
        print("\n  (query returned 0 rows)")


# ── CLI ────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Agentic Data Intelligence Platform.",
    )
    parser.add_argument(
        "--table",
        required=True,
        metavar="PROJECT.DATASET.TABLE",
        help="Fully-qualified BigQuery table reference.",
    )
    parser.add_argument(
        "--ask",
        default=None,
        metavar="QUESTION",
        help="Natural-language question to run against the table.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Print the schema profile (always shown when --ask is omitted).",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    # Suppress verbose BQ/Gemini logs when asking a question; keep INFO for profile mode.
    log_level = logging.WARNING if args.ask else logging.INFO
    logging.basicConfig(level=log_level, format="%(levelname)s:%(name)s:%(message)s")

    bq = BigQueryClient()

    print(f"\nLoading profile for: {args.table}")
    try:
        profile = SchemaDiscoveryAgent(bq).run(args.table)
    except Exception as exc:
        logger.error("Failed to profile table '%s': %s", args.table, exc)
        sys.exit(1)

    print(f"  {len(profile.columns)} columns  {profile.row_count:,} rows")

    if args.profile or not args.ask:
        _print_profile(profile)

    if args.ask:
        try:
            result = run_pipeline(
                question=args.ask,
                profile=profile,
                intent_agent=IntentAgent(),
                sql_agent=SqlGenerationAgent(),
                val_agent=ValidationAgent(bq),
            )
            _print_query_result(args.ask, result)
        except Exception as exc:
            logger.error("Pipeline failed: %s", exc)
            logging.exception("Details:")
            sys.exit(1)


if __name__ == "__main__":
    main()
