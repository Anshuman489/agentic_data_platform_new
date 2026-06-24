"""
core/dataset_profiler.py — Profiles an arbitrary BigQuery table.

Single responsibility: run the minimal queries needed to populate every field
of ColumnProfile and extract sample rows. No caching, no DatasetProfile assembly,
no LLM usage — those belong to the agent above this layer.
"""

import logging
import re
from typing import Any

from google.cloud import bigquery

from config.settings import settings
from core.bigquery_client import BigQueryClient
from models.dataset import ColumnProfile, InferredRole

logger = logging.getLogger(__name__)

# ── Type classification ────────────────────────────────────────────────────────
# Controls which aggregation expressions are valid for each column type.

_NUMERIC_TYPES = frozenset({
    "INT64", "INT", "INTEGER", "SMALLINT", "BIGINT", "TINYINT", "BYTEINT",
    "FLOAT64", "FLOAT",
    "NUMERIC", "DECIMAL",
    "BIGNUMERIC", "BIGDECIMAL",
})

_DATE_TYPES = frozenset({"DATE", "DATETIME", "TIMESTAMP", "TIME"})

_STRING_TYPES = frozenset({"STRING", "BYTES"})

# Last-token words that strongly suggest a column is a row or foreign-key identifier.
# Checked against the final token after splitting on underscores and camelCase boundaries,
# so both snake_case (customer_id) and camelCase (CustomerID, InvoiceNo) are caught.
_IDENTIFIER_TOKENS = frozenset({"id", "key", "uuid", "no", "num", "number", "code"})


def _tokenize_name(name: str) -> list[str]:
    """
    Split a column name into lowercase tokens on underscores and camelCase boundaries.

    Examples:
      customer_id  → ["customer", "id"]
      CustomerID   → ["customer", "id"]
      InvoiceNo    → ["invoice", "no"]
      StockCode    → ["stock", "code"]
      country      → ["country"]
    """
    # Insert separator before a lowercase→uppercase transition (e.g. Invoice→No)
    s = re.sub(r"([a-z])([A-Z])", r"\1_\2", name)
    # Insert separator before an uppercase run followed by a capital+lowercase (e.g. XMLParser)
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", s)
    return [t.lower() for t in s.split("_") if t]


def _looks_like_identifier(name: str) -> bool:
    """Return True when the last token of a column name is a known identifier word."""
    tokens = _tokenize_name(name)
    return bool(tokens) and tokens[-1] in _IDENTIFIER_TOKENS


# ── DatasetProfiler ────────────────────────────────────────────────────────────

class DatasetProfiler:
    """
    Profiles an arbitrary BigQuery table using exactly two queries plus one
    free metadata call. Downstream agents receive clean ColumnProfile objects
    without depending on the google-cloud-bigquery library directly.
    """

    def __init__(self, bq: BigQueryClient) -> None:
        """
        Args:
            bq: An initialised BigQueryClient. Injected so callers (and tests)
                control the connection — this class never constructs one itself.
        """
        self._bq = bq

    # ── Public API ─────────────────────────────────────────────────────────────

    def profile(
        self,
        table_ref: str,
    ) -> tuple[list[ColumnProfile], list[dict[str, Any]]]:
        """
        Profile the given table and return per-column metadata and sample rows.

        Three calls to BigQuery are made:
          0. get_table()      — free metadata (schema, num_rows, num_bytes)
          1. Aggregation SQL  — null fractions, cardinalities, min/max — one scan
          2. SELECT * LIMIT N — representative sample rows

        Args:
            table_ref: Fully-qualified table reference — "project.dataset.table".

        Returns:
            A tuple of:
              - list[ColumnProfile]: one entry per column, in schema order,
                                     with all stats and inferred_role populated.
              - list[dict]:          up to bq_sample_row_limit raw rows.
        """
        logger.info("Profiling table: %s", table_ref)

        # ── Step 0: schema and metadata (free, no rows scanned) ───────────────
        table = self._bq.get_table(table_ref)
        schema: list[bigquery.SchemaField] = list(table.schema)

        columns: list[ColumnProfile] = [
            ColumnProfile(
                name=field.name,
                bq_type=field.field_type,
                mode=field.mode,
                description=field.description or None,
            )
            for field in schema
        ]

        # ── Step 1: single aggregation query ──────────────────────────────────
        if columns:
            agg_sql = self._build_stats_query(table_ref, schema)
            if agg_sql:  # empty when all columns are REPEATED or RECORD
                logger.debug("Running aggregation stats query for %s", table_ref)
                agg_rows = self._bq.run_query(agg_sql)
                if agg_rows:
                    # Aggregation always returns exactly one row.
                    self._apply_stats(columns, agg_rows[0])

        # ── Step 2: sample rows ────────────────────────────────────────────────
        logger.debug(
            "Fetching %d sample rows from %s",
            settings.bq_sample_row_limit,
            table_ref,
        )
        sample_rows = self._bq.run_query(
            f"SELECT * FROM `{table_ref}` LIMIT {settings.bq_sample_row_limit}"
        )

        # ── Step 3: derive sample_values and inferred_role (pure Python) ──────
        self._apply_sample_values(columns, sample_rows)
        for col in columns:
            col.inferred_role = self._infer_role(col)

        logger.info(
            "Profiled %s: %d columns, %d sample rows",
            table_ref,
            len(columns),
            len(sample_rows),
        )
        return columns, sample_rows

    # ── Private: query builder ─────────────────────────────────────────────────

    def _build_stats_query(
        self,
        table_ref: str,
        schema: list[bigquery.SchemaField],
    ) -> str:
        """
        Build a single SELECT that computes all column statistics in one table scan.

        Column aliases use positional indices (c{i}__metric) rather than column
        names to avoid any conflict with special characters, reserved words, or
        columns whose names clash after sanitisation.

        Returns an empty string when no valid aggregation expressions exist
        (e.g. the table has only REPEATED or RECORD columns).
        """
        parts: list[str] = []

        for i, field in enumerate(schema):
            bq_type = field.field_type.upper()

            # APPROX_COUNT_DISTINCT and MIN/MAX are not valid on REPEATED (array)
            # columns in BigQuery Standard SQL.
            if field.mode == "REPEATED":
                continue

            # RECORD/STRUCT: aggregate the sub-fields, not the parent. Skip.
            if bq_type in ("RECORD", "STRUCT"):
                continue

            # Backtick-quote the column name to handle reserved words and spaces.
            col = f"`{field.name}`"

            parts.append(
                f"COUNTIF({col} IS NULL) / NULLIF(COUNT(*), 0) AS c{i}__nf"
            )
            parts.append(f"APPROX_COUNT_DISTINCT({col}) AS c{i}__cd")

            if bq_type in _NUMERIC_TYPES:
                parts.append(f"MIN({col}) AS c{i}__nmin")
                parts.append(f"MAX({col}) AS c{i}__nmax")
            elif bq_type in _DATE_TYPES:
                # CAST to STRING for portable ISO 8601 storage in ColumnProfile.
                parts.append(f"CAST(MIN({col}) AS STRING) AS c{i}__dmin")
                parts.append(f"CAST(MAX({col}) AS STRING) AS c{i}__dmax")

        if not parts:
            return ""

        return "SELECT\n  " + ",\n  ".join(parts) + f"\nFROM `{table_ref}`"

    # ── Private: stat application ──────────────────────────────────────────────

    def _apply_stats(
        self,
        columns: list[ColumnProfile],
        row: dict[str, Any],
    ) -> None:
        """
        Write the aggregation row's values back into the ColumnProfile list.

        Uses the same positional c{i}__metric aliases as _build_stats_query.
        Columns that were skipped in the query (REPEATED, RECORD) are left at
        their model defaults (null_fraction=0.0, cardinality=None).
        """
        for i, col in enumerate(columns):
            bq_type = col.bq_type.upper()

            if col.mode == "REPEATED" or bq_type in ("RECORD", "STRUCT"):
                continue

            col.null_fraction = float(row.get(f"c{i}__nf") or 0.0)
            col.cardinality = row.get(f"c{i}__cd")

            if bq_type in _NUMERIC_TYPES:
                nmin = row.get(f"c{i}__nmin")
                nmax = row.get(f"c{i}__nmax")
                col.numeric_min = float(nmin) if nmin is not None else None
                col.numeric_max = float(nmax) if nmax is not None else None
            elif bq_type in _DATE_TYPES:
                col.date_min = row.get(f"c{i}__dmin")
                col.date_max = row.get(f"c{i}__dmax")

    # ── Private: sample values ─────────────────────────────────────────────────

    def _apply_sample_values(
        self,
        columns: list[ColumnProfile],
        sample_rows: list[dict[str, Any]],
    ) -> None:
        """
        Populate sample_values on each ColumnProfile from the fetched sample rows.

        NULLs are skipped so every entry in sample_values is a real, non-null
        value. Duplicates are deduplicated (first-seen preserved) to maximise
        the variety within the row limit.
        """
        for col in columns:
            # REPEATED columns return lists from SELECT *; skip rather than
            # storing nested lists in sample_values.
            if col.mode == "REPEATED":
                continue

            seen: set[str] = set()
            values: list[Any] = []

            for sample_row in sample_rows:
                v = sample_row.get(col.name)
                if v is None:
                    continue
                # repr() gives a type-aware hashable key: "1" vs 1 vs 1.0
                key = repr(v)
                if key not in seen:
                    seen.add(key)
                    values.append(v)

            col.sample_values = values

    # ── Private: role inference ────────────────────────────────────────────────

    def _infer_role(self, col: ColumnProfile) -> InferredRole:
        """
        Classify a column's semantic role from its type, cardinality, and name.

        Rules are evaluated top-to-bottom; the first match wins.

        1. REPEATED mode   → UNKNOWN   (arrays need special handling, not scalar)
        2. Date/time type  → DATE
        3. Boolean type    → DIMENSION  (only two distinct values; always categorical)
        4. Numeric type:
             name looks like an identifier → IDENTIFIER
             cardinality ≤ measure_cardinality_threshold → DIMENSION
             otherwise → MEASURE
        5. String/Bytes type:
             cardinality ≤ dimension_cardinality_threshold → DIMENSION
             name looks like an identifier → IDENTIFIER
             otherwise → FREE_TEXT
        6. Anything else (RECORD, STRUCT, unknown) → UNKNOWN

        Cardinality thresholds are read from Settings so they can be tuned
        without modifying this file.
        """
        bq_type = col.bq_type.upper()

        if col.mode == "REPEATED":
            return InferredRole.UNKNOWN

        if bq_type in _DATE_TYPES:
            return InferredRole.DATE

        if bq_type in ("BOOL", "BOOLEAN"):
            return InferredRole.DIMENSION

        cardinality = col.cardinality or 0

        if bq_type in _NUMERIC_TYPES:
            # Name check takes priority: user_id INT64 with 50 rows is still an
            # identifier, not a dimension — cardinality alone cannot distinguish.
            if _looks_like_identifier(col.name):
                return InferredRole.IDENTIFIER
            if cardinality <= settings.measure_cardinality_threshold:
                return InferredRole.DIMENSION
            return InferredRole.MEASURE

        if bq_type in _STRING_TYPES:
            # Low-cardinality strings (e.g. status, country) → categorical.
            if cardinality <= settings.dimension_cardinality_threshold:
                return InferredRole.DIMENSION
            # High-cardinality: name decides between IDENTIFIER and FREE_TEXT.
            if _looks_like_identifier(col.name):
                return InferredRole.IDENTIFIER
            return InferredRole.FREE_TEXT

        return InferredRole.UNKNOWN
