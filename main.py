"""
main.py — CLI entry point for the Agentic Data Intelligence Platform.

Usage:
    # Register a table into the catalog (profile + save):
    python main.py --register project.dataset.table

    # List all registered tables:
    python main.py --list-tables

    # Ask a question — router picks the best table automatically:
    python main.py --ask "total revenue by country"

    # Ask against a specific table (skips router):
    python main.py --table project.dataset.table --ask "total revenue by country"

    # Profile a table without asking a question:
    python main.py --table project.dataset.table

    # Show schema profile alongside query results:
    python main.py --table project.dataset.table --ask "..." --profile
"""

import argparse
import logging
import re
import sys
from decimal import Decimal

from core.bigquery_client import BigQueryClient
from agents.schema_discovery_agent import SchemaDiscoveryAgent
from agents.sql_generation_agent import SqlGenerationAgent
from agents.validation_agent import ValidationAgent
from agents.pipeline import run_pipeline
from agents.catalog_manager import CatalogManager
from agents.table_router_agent import TableRouterAgent
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


# ── Catalog display ────────────────────────────────────────────────────────────

def _print_catalog(catalog: CatalogManager) -> None:
    profiles = catalog.list()
    if not profiles:
        print("\n  No cached tables found. Run: python main.py --register --table PROJECT.DATASET.TABLE")
        return

    print(f"\n  {len(profiles)} table(s) available for routing:\n")
    print(f"  {'TABLE':<55} {'ROWS':>12}  {'COLUMNS':>8}  DATE RANGE")
    print("  " + "-" * 95)
    for p in profiles:
        date_min = next((c.date_min for c in p.columns if c.date_min), None)
        date_max = next((c.date_max for c in p.columns if c.date_max), None)
        date_range = f"{date_min} to {date_max}" if date_min else "-"
        print(
            f"  {p.table_ref:<55} {p.row_count:>12,}  {len(p.columns):>8}  {date_range}"
        )
    print()


# ── CLI ────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Agentic Data Intelligence Platform.",
    )
    parser.add_argument(
        "--table",
        default=None,
        metavar="PROJECT.DATASET.TABLE",
        help="Target table. Required for --register and --profile. Optional for --ask (router picks if omitted).",
    )
    parser.add_argument(
        "--register",
        action="store_true",
        help="Profile --table and cache it so it is available for automatic routing.",
    )
    parser.add_argument(
        "--list-tables",
        action="store_true",
        help="List all cached tables available for routing.",
    )
    parser.add_argument(
        "--ask",
        default=None,
        metavar="QUESTION",
        help="Natural-language question. Uses --table if given, else routes automatically.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Print the schema profile (always shown when --ask is omitted).",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    log_level = logging.WARNING if args.ask else logging.INFO
    logging.basicConfig(level=log_level, format="%(levelname)s:%(name)s:%(message)s")

    catalog = CatalogManager()

    # ── --list-tables ──────────────────────────────────────────────────────────
    if args.list_tables:
        _print_catalog(catalog)
        if not args.ask and not args.register:
            return

    # ── --register ─────────────────────────────────────────────────────────────
    if args.register:
        if not args.table:
            print("Error: --register requires --table PROJECT.DATASET.TABLE")
            sys.exit(1)

        bq = BigQueryClient()
        print(f"\nProfiling: {args.table}")
        try:
            profile = SchemaDiscoveryAgent(bq).run(args.table)
        except Exception as exc:
            logger.error("Failed to profile '%s': %s", args.table, exc)
            sys.exit(1)

        print(f"  Cached  : {profile.table_ref}")
        print(f"  Columns : {len(profile.columns)}")
        print(f"  Rows    : {profile.row_count:,}")
        date_min = next((c.date_min for c in profile.columns if c.date_min), None)
        date_max = next((c.date_max for c in profile.columns if c.date_max), None)
        if date_min:
            print(f"  Dates   : {date_min} to {date_max}")
        print(f"\n  {catalog.count} table(s) now available for routing.")

        if not args.ask:
            return

    # ── --ask (with or without --table) ───────────────────────────────────────
    if args.ask:
        args.ask = re.sub(r"[?!.\s]+$", "", args.ask.strip()).lower()
        bq = BigQueryClient()

        if args.table:
            # Explicit table — skip router
            print(f"\nLoading profile for: {args.table}")
            try:
                profile = SchemaDiscoveryAgent(bq).run(args.table)
            except Exception as exc:
                logger.error("Failed to profile '%s': %s", args.table, exc)
                sys.exit(1)
            print(f"  {len(profile.columns)} columns  {profile.row_count:,} rows")
        else:
            # No table specified — route from catalog
            print(f"\nRouting question to best table...")
            try:
                route = TableRouterAgent(catalog).route(args.ask)
            except Exception as exc:
                print(f"\nRouting failed: {exc}")
                sys.exit(1)

            if route.ambiguous:
                # Build ordered list: winner first, then alternatives
                candidates = [(route.table_ref, route.confidence, route.reasoning)] + [
                    (ref, conf, "") for ref, conf in route.alternatives
                    if conf >= 0.50
                ]

                print(f"\n  I found {len(candidates)} possible tables for this question:\n")
                for i, (ref, conf, reason) in enumerate(candidates, 1):
                    print(f"  [{i}] {ref}  ({conf:.0%})")
                    if reason:
                        print(f"      {reason}")

                while True:
                    try:
                        raw = input(f"\n  Which table should I query? Enter 1-{len(candidates)}: ").strip()
                        choice = int(raw)
                        if 1 <= choice <= len(candidates):
                            break
                        print(f"  Please enter a number between 1 and {len(candidates)}.")
                    except (ValueError, EOFError):
                        print("  Please enter a valid number.")

                chosen_ref = candidates[choice - 1][0]
                route = route.model_copy(update={"table_ref": chosen_ref})
                print()
            else:
                print(f"  Selected : {route.table_ref}")
                print(f"  Confidence: {route.confidence:.0%}")
                print(f"  Reason   : {route.reasoning}")

            print(f"\nLoading profile for: {route.table_ref}")
            try:
                profile = SchemaDiscoveryAgent(bq).run(route.table_ref)
            except Exception as exc:
                logger.error("Failed to profile '%s': %s", route.table_ref, exc)
                sys.exit(1)
            print(f"  {len(profile.columns)} columns  {profile.row_count:,} rows")

        if args.profile:
            _print_profile(profile)

        try:
            result = run_pipeline(
                question=args.ask,
                profile=profile,
                sql_agent=SqlGenerationAgent(),
                val_agent=ValidationAgent(bq),
            )
            _print_query_result(args.ask, result)
        except Exception as exc:
            logger.error("Pipeline failed: %s", exc)
            logging.exception("Details:")
            sys.exit(1)

        return

    # ── --table alone (profile mode) ──────────────────────────────────────────
    if args.table:
        bq = BigQueryClient()
        print(f"\nLoading profile for: {args.table}")
        try:
            profile = SchemaDiscoveryAgent(bq).run(args.table)
        except Exception as exc:
            logger.error("Failed to profile '%s': %s", args.table, exc)
            sys.exit(1)
        print(f"  {len(profile.columns)} columns  {profile.row_count:,} rows")
        _print_profile(profile)
        return

    # ── No action ─────────────────────────────────────────────────────────────
    _build_parser().print_help()


if __name__ == "__main__":
    main()
