"""
agents/sql_generation_agent.py — Translates IntentResult into BigQuery SQL.

Pure deterministic translation — no LLM is used here. Every IntentResult
field maps 1:1 to a SQL clause. The only decisions this module makes are:

  1. Whether a Metric.column is a raw expression (Quantity * UnitPrice) or a
     plain column name — which decides backtick-quoting.
  2. Whether a filter value needs string quotes or not — decided by the
     column's BQ type from the DatasetProfile.
  3. COUNT_DISTINCT → COUNT(DISTINCT ...) rewrite (BigQuery syntax).
  4. Whether a Dimension has a date_trunc granularity — which wraps the column
     in DATE_TRUNC(col, GRANULARITY) in both SELECT and GROUP BY.
"""

import logging
import re

from models.dataset import DatasetProfile
from models.intent import Dimension, Filter, IntentResult, Metric

logger = logging.getLogger(__name__)

# Type sets used for filter value quoting.
_NUMERIC_TYPES = frozenset({
    "INT64", "INT", "INTEGER", "SMALLINT", "BIGINT", "TINYINT", "BYTEINT",
    "FLOAT64", "FLOAT", "NUMERIC", "DECIMAL", "BIGNUMERIC", "BIGDECIMAL",
})
_DATE_TYPES = frozenset({"DATE", "DATETIME", "TIMESTAMP", "TIME"})


# ── Expression detection ───────────────────────────────────────────────────────

def _is_expression(column: str) -> bool:
    """
    Return True when column is a derived SQL expression, not a plain column name.

    A plain column name contains only word characters, digits, and underscores.
    Anything containing arithmetic operators or whitespace is an expression.

    Examples:
      "Quantity"              → False  (plain column)
      "Quantity * UnitPrice"  → True   (expression)
      "*"                     → False  (COUNT(*) special case — handled separately)
    """
    if column.strip() == "*":
        return False
    return bool(re.search(r"[\+\-\*/\s]", column))


# ── Per-dimension SQL builder ─────────────────────────────────────────────────

def _dim_sql(dim: Dimension) -> tuple[str, str]:
    """
    Build the SQL expression and alias for one Dimension.

    Returns:
        (sql_expression, alias) — e.g.:
          plain column → ("`Country`",              "Country")
          date_trunc   → ("DATE_TRUNC(`InvoiceDate`, MONTH)", "InvoiceDate")

    The alias is always the bare column name so ORDER BY can reference it
    whether or not the column is wrapped in DATE_TRUNC.
    """
    if dim.date_trunc:
        return f"DATE_TRUNC(`{dim.column}`, {dim.date_trunc})", dim.column
    return f"`{dim.column}`", dim.column


# ── Per-metric SQL builder ─────────────────────────────────────────────────────

def _metric_sql(metric: Metric) -> tuple[str, str]:
    """
    Build the SQL fragment and alias for one Metric.

    Returns:
        (sql_fragment, alias) — e.g. ("SUM(`Quantity`)", "Quantity")
    """
    col = metric.column
    agg = metric.aggregation

    # COUNT(*) — special case, no column quoting
    if col.strip() == "*":
        return "COUNT(*)", "count_all"

    # COUNT_DISTINCT → COUNT(DISTINCT ...) in BigQuery syntax
    if agg == "COUNT_DISTINCT":
        if _is_expression(col):
            return f"COUNT(DISTINCT {col})", "count_distinct_expr"
        return f"COUNT(DISTINCT `{col}`)", col

    # Derived expression — pass as-is, no backticks on the whole expression
    if _is_expression(col):
        return f"{agg}({col})", f"{agg.lower()}_expr"

    # Plain column
    return f"{agg}(`{col}`)", col


# ── SqlGenerationAgent ─────────────────────────────────────────────────────────

class SqlGenerationAgent:
    """
    Translates an IntentResult + DatasetProfile into a BigQuery SQL string.

    No external calls. No LLM. Every clause is built from the IntentResult
    fields using the DatasetProfile only for column type lookups.

    Usage:
        agent = SqlGenerationAgent()
        sql = agent.run(intent, profile)
    """

    def run(self, intent: IntentResult, profile: DatasetProfile) -> str:
        """
        Build a BigQuery-compatible SELECT statement.

        Args:
            intent:  Structured query parameters produced by IntentAgent.
            profile: DatasetProfile of the target table — used for column
                     types (to decide quoting) and the table reference.

        Returns:
            A formatted BigQuery SQL string, ready to dry-run or execute.
        """
        col_types: dict[str, str] = {
            c.name: c.bq_type.upper() for c in profile.columns
        }

        clauses: list[str] = []

        # ── SELECT ────────────────────────────────────────────────────────────
        select_exprs, metric_aliases = self._build_select(intent)
        clauses.append("SELECT\n  " + ",\n  ".join(select_exprs))

        # ── FROM ──────────────────────────────────────────────────────────────
        clauses.append(f"FROM `{profile.table_ref}`")

        # ── WHERE ─────────────────────────────────────────────────────────────
        where_parts = self._build_where(intent, col_types)
        if where_parts:
            clauses.append("WHERE\n  " + "\n  AND ".join(where_parts))

        # ── GROUP BY ──────────────────────────────────────────────────────────
        # Only emit GROUP BY when we have aggregation metrics + grouping dimensions.
        # Use the same expression as SELECT (DATE_TRUNC when set, plain column otherwise).
        if intent.metrics and intent.dimensions:
            dims = [_dim_sql(d)[0] for d in intent.dimensions]
            clauses.append("GROUP BY\n  " + ",\n  ".join(dims))

        # ── ORDER BY ──────────────────────────────────────────────────────────
        if intent.order_by:
            order_parts = self._build_order_by(intent, metric_aliases)
            clauses.append("ORDER BY\n  " + ",\n  ".join(order_parts))

        # ── LIMIT ─────────────────────────────────────────────────────────────
        if intent.limit is not None:
            clauses.append(f"LIMIT {intent.limit}")

        sql = "\n".join(clauses)
        logger.debug("Generated SQL:\n%s", sql)
        return sql

    # ── Private: SELECT ────────────────────────────────────────────────────────

    def _build_select(
        self, intent: IntentResult
    ) -> tuple[list[str], dict[str, str]]:
        """
        Build the SELECT expression list and a metric alias map.

        metric_aliases: Metric.column → alias used in SELECT so ORDER BY can
        reference the alias instead of repeating the aggregation expression.

        Layout:
          • No metrics → SELECT *
          • Metrics only (no dimensions) → SELECT AGG(col) AS alias, ...
          • Metrics + dimensions → SELECT `dim1`, `dim2`, AGG(col) AS alias, ...
        """
        if not intent.metrics:
            return ["*"], {}

        exprs: list[str] = []
        aliases: dict[str, str] = {}

        # Dimensions come first in SELECT (mirrors GROUP BY order).
        # DATE_TRUNC dimensions get an alias so ORDER BY can reference them.
        for dim in intent.dimensions:
            expr, alias = _dim_sql(dim)
            if dim.date_trunc:
                exprs.append(f"{expr} AS `{alias}`")
            else:
                exprs.append(expr)

        for metric in intent.metrics:
            frag, alias = _metric_sql(metric)
            exprs.append(f"{frag} AS `{alias}`")
            aliases[metric.column] = alias  # "* " → "count_all"
            aliases[frag] = alias           # "COUNT(*)" → "count_all"
            aliases[alias] = alias          # "count_all" → "count_all"

        return exprs, aliases

    # ── Private: WHERE ─────────────────────────────────────────────────────────

    def _build_where(
        self,
        intent: IntentResult,
        col_types: dict[str, str],
    ) -> list[str]:
        parts: list[str] = []

        for f in intent.filters:
            parts.append(self._format_filter(f, col_types.get(f.column)))

        if intent.time_range:
            tr = intent.time_range
            col_type = col_types.get(tr.column, "")

            if tr.start and tr.end:
                if col_type in _DATE_TYPES:
                    # Wrap in DATE() so DATETIME/TIMESTAMP columns compare cleanly
                    parts.append(
                        f"DATE(`{tr.column}`) BETWEEN "
                        f"DATE '{tr.start}' AND DATE '{tr.end}'"
                    )
                else:
                    parts.append(
                        f"`{tr.column}` BETWEEN '{tr.start}' AND '{tr.end}'"
                    )
            elif tr.start:
                parts.append(f"`{tr.column}` >= '{tr.start}'")
            elif tr.end:
                parts.append(f"`{tr.column}` <= '{tr.end}'")

        return parts

    def _format_filter(self, f: Filter, bq_type: str | None) -> str:
        """Render one Filter as a SQL WHERE condition."""
        if f.operator in ("IS NULL", "IS NOT NULL"):
            return f"`{f.column}` {f.operator}"

        if f.operator in ("IN", "NOT IN") and f.value:
            items = [v.strip() for v in f.value.split(",")]
            if bq_type in _NUMERIC_TYPES:
                joined = ", ".join(items)
            else:
                joined = ", ".join(f"'{v}'" for v in items)
            return f"`{f.column}` {f.operator} ({joined})"

        quoted = self._quote_scalar(f.value or "", bq_type)
        return f"`{f.column}` {f.operator} {quoted}"

    def _quote_scalar(self, value: str, bq_type: str | None) -> str:
        """Quote a scalar filter value based on column type."""
        if bq_type in _NUMERIC_TYPES:
            return value  # no quotes — BigQuery infers the numeric type
        return f"'{value}'"

    # ── Private: ORDER BY ──────────────────────────────────────────────────────

    def _build_order_by(
        self,
        intent: IntentResult,
        metric_aliases: dict[str, str],
    ) -> list[str]:
        """
        Build ORDER BY clauses.

        If the sort column matches an aggregated metric, use the alias defined
        in SELECT (e.g. ORDER BY `Quantity` instead of ORDER BY SUM(`Quantity`)).
        Otherwise backtick-quote the column name directly.
        """
        parts: list[str] = []
        for o in intent.order_by:
            if o.column in metric_aliases:
                parts.append(f"`{metric_aliases[o.column]}` {o.direction}")
            else:
                parts.append(f"`{o.column}` {o.direction}")
        return parts
