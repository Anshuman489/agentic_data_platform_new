from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Optional

from pydantic import BaseModel, Field


# ── InferredRole ──────────────────────────────────────────────────────────────

class InferredRole(StrEnum):
    """
    The platform's classification of what a column represents in the dataset.

    This is inferred automatically from the column's BigQuery type and cardinality
    (number of distinct values). It is never hardcoded per-dataset — the same
    heuristics apply to any table.

    StrEnum (Python 3.11+) means each value IS a string:
    InferredRole.DIMENSION == "DIMENSION"  →  True
    This makes JSON output human-readable without any special serialization config.
    """

    # Categorical column: used for grouping and filtering.
    # Examples: country, status, product_category, gender
    DIMENSION = "DIMENSION"

    # Quantitative column: used for aggregation (SUM, AVG, COUNT).
    # Examples: revenue, quantity, price, duration_seconds
    MEASURE = "MEASURE"

    # Temporal column: used for date filtering, trending, and time-series analysis.
    # Examples: created_at, order_date, event_timestamp
    DATE = "DATE"

    # A column that uniquely identifies a row or links to another table.
    # Examples: user_id, order_id, transaction_uuid, customer_key
    IDENTIFIER = "IDENTIFIER"

    # A high-cardinality string column that is unlikely to be useful for grouping.
    # Examples: email_body, description, notes, raw_json
    FREE_TEXT = "FREE_TEXT"

    # Fallback — used when heuristics cannot determine the role.
    # Examples: ARRAY columns (REPEATED mode), STRUCT columns, ambiguous types
    UNKNOWN = "UNKNOWN"


# ── ColumnProfile ─────────────────────────────────────────────────────────────

class ColumnProfile(BaseModel):
    """
    Everything the platform knows about a single column in a BigQuery table.

    Populated entirely by schema_inspector.py — no manual input required.
    All fields have defaults so a ColumnProfile can be constructed incrementally.
    """

    # The exact column name as it appears in BigQuery.
    name: str

    # The BigQuery data type as a string, e.g. "STRING", "INT64", "FLOAT64",
    # "DATE", "TIMESTAMP", "BOOL", "NUMERIC", "BYTES".
    # Stored as a raw string rather than an enum to avoid breaking on custom
    # or future BQ types we haven't accounted for.
    bq_type: str

    # BigQuery field mode — controls whether the column can be null or is an array.
    # "NULLABLE"  → column can contain NULL values (most common)
    # "REQUIRED"  → column never contains NULL (enforced by BQ schema)
    # "REPEATED"  → column is an array (list of values); cardinality cannot be computed
    mode: str

    # The column description from the BigQuery table schema, if one was set.
    # Often empty on raw tables, but useful when present.
    description: Optional[str] = None

    # The platform's classification of this column's purpose.
    # Starts as UNKNOWN and is set by the role inference logic in schema_inspector.
    inferred_role: InferredRole = InferredRole.UNKNOWN

    # Fraction of rows where this column is NULL, as a value between 0.0 and 1.0.
    # 0.0 = no nulls, 1.0 = entirely null.
    # Computed as: COUNT(NULL rows) / total row count
    null_fraction: float = 0.0

    # Approximate number of distinct non-null values in this column.
    # Computed using BigQuery's APPROX_COUNT_DISTINCT function, which is fast and
    # cheap but has a small margin of error (~1%) on very large tables.
    # None for REPEATED (array) columns — APPROX_COUNT_DISTINCT does not work on arrays.
    cardinality: Optional[int] = None

    # A small list of real, non-null values from this column, taken from sample rows.
    # Used by the SQL Generation Agent to understand what values look like.
    # Examples for a "status" column: ["active", "inactive", "pending"]
    # Any type because BQ column values can be strings, ints, floats, dates, etc.
    sample_values: list[Any] = []

    # Min and max for numeric columns (INT64, FLOAT64, NUMERIC, etc.).
    # Both are stored as float for consistency — BQ integers fit safely in float64.
    # None for non-numeric columns.
    numeric_min: Optional[float] = None
    numeric_max: Optional[float] = None

    # Min and max for date/time columns (DATE, DATETIME, TIMESTAMP).
    # Stored as ISO 8601 strings (e.g. "2023-01-01") rather than Python date objects
    # to avoid timezone and serialization complexity in V1.
    # None for non-temporal columns.
    date_min: Optional[str] = None
    date_max: Optional[str] = None


# ── DatasetProfile ────────────────────────────────────────────────────────────

class DatasetProfile(BaseModel):
    """
    A complete snapshot of what the platform knows about a BigQuery table.

    This is the central data contract for the entire platform.
    It is produced once by SchemaDiscoveryAgent, cached to disk as JSON,
    and consumed by every downstream agent (Intent, SQL Generation, Validation).

    No agent modifies this object after it is created — it is read-only context.
    """

    # The fully qualified BigQuery table reference: "project.dataset.table"
    # This is the primary key used to look up or invalidate the cache.
    table_ref: str

    # The three components of table_ref, stored separately for convenient access.
    # Set explicitly by SchemaDiscoveryAgent after parsing table_ref — not derived
    # here in the model to keep the model simple and free of parsing logic.
    project: str
    dataset_id: str
    table_id: str

    # Total number of rows in the table at the time of profiling.
    # Comes from BigQuery table metadata — no full scan required.
    row_count: int

    # Size of the table in bytes at the time of profiling.
    # Comes from BigQuery table metadata — useful for estimating query cost.
    size_bytes: int

    # The table-level description from BigQuery, if one was set.
    table_description: Optional[str] = None

    # One ColumnProfile for every column in the table.
    # Populated by schema_inspector.py — order matches the BigQuery schema order.
    columns: list[ColumnProfile] = []

    # A small number of raw rows from the table (controlled by bq_sample_row_limit).
    # Each row is a dict of {column_name: value}.
    # Used to give the SQL Generation Agent a concrete sense of what the data looks like.
    sample_rows: list[dict] = []

    # UTC timestamp of when this profile was generated.
    # Used by SchemaDiscoveryAgent to decide whether the cached profile is still fresh.
    # default_factory means each new DatasetProfile gets the current time automatically.
    profiled_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
