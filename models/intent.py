"""
models/intent.py — Pydantic models for structured query intent.

IntentAgent populates an IntentResult from a natural-language question.
SqlGenerationAgent turns the IntentResult into a BigQuery SQL statement.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Metric(BaseModel):
    """A single aggregated expression in the SELECT clause."""

    column: str = Field(
        description="Column to aggregate. Use '*' for COUNT(*)."
    )
    aggregation: Literal["COUNT", "COUNT_DISTINCT", "SUM", "AVG", "MIN", "MAX"] = Field(
        description="SQL aggregation function to apply."
    )


class Filter(BaseModel):
    """A single WHERE clause condition."""

    column: str = Field(description="Exact column name from the table schema.")
    operator: Literal[
        "=", "!=", ">", "<", ">=", "<=", "IN", "NOT IN", "LIKE", "IS NULL", "IS NOT NULL"
    ] = Field(description="SQL comparison operator.")
    value: str | None = Field(
        default=None,
        description=(
            "Filter value as a string. "
            "For IN / NOT IN operators separate multiple values with commas, e.g. 'UK,Germany,France'. "
            "Omit for IS NULL / IS NOT NULL."
        ),
    )


class TimeRange(BaseModel):
    """A date or timestamp window applied as an additional WHERE condition."""

    column: str = Field(description="Date or timestamp column to filter on.")
    start: str | None = Field(
        default=None,
        description="ISO 8601 start date/datetime, inclusive. Null means open-ended.",
    )
    end: str | None = Field(
        default=None,
        description="ISO 8601 end date/datetime, inclusive. Null means open-ended.",
    )


class OrderBy(BaseModel):
    """A single ORDER BY expression."""

    column: str = Field(description="Column to sort by.")
    direction: Literal["ASC", "DESC"] = Field(default="DESC")


class Dimension(BaseModel):
    """
    A GROUP BY column with optional date extraction or truncation.

    Use date_trunc for calendar-period grouping (per month, per quarter, etc.).
    Use extract for sub-day or ordinal grouping (hour of day, day of week, etc.).
    Only one of date_trunc / extract should be set at a time.

    Examples:
      "sales by country"            → Dimension(column="Country")
      "revenue per month"           → Dimension(column="InvoiceDate", date_trunc="MONTH")
      "trips by hour of day"        → Dimension(column="pickup_datetime", extract="HOUR")
      "trips by day of week"        → Dimension(column="pickup_datetime", extract="DOW")
    """

    column: str = Field(description="Exact column name from the table schema.")
    date_trunc: Literal["DAY", "WEEK", "MONTH", "QUARTER", "YEAR"] | None = Field(
        default=None,
        description=(
            "Truncation granularity for date/timestamp columns. "
            "Set when the user asks for 'per day', 'per week', 'per month', "
            "'per quarter', or 'per year' grouping. Null for non-date dimensions "
            "or when exact date-level grouping is intended."
        ),
    )
    extract: Literal["HOUR", "MINUTE", "DOW", "DOY", "MONTH", "YEAR"] | None = Field(
        default=None,
        description=(
            "Date part to extract from a date/timestamp column using EXTRACT(). "
            "Use for sub-day or ordinal grouping: "
            "HOUR = hour of day (0-23), "
            "MINUTE = minute of hour (0-59), "
            "DOW = day of week (0=Sunday, 6=Saturday), "
            "DOY = day of year (1-366), "
            "MONTH = month number (1-12), "
            "YEAR = four-digit year. "
            "Prefer date_trunc for calendar-period grouping (per month, per quarter). "
            "Use extract when the user asks for 'by hour', 'by day of week', etc."
        ),
    )


class IntentResult(BaseModel):
    """
    Structured representation of what a natural-language question is asking for.

    Every field maps directly to a clause in the generated BigQuery SQL:
      metrics     → SELECT aggregation expressions
      dimensions  → GROUP BY columns (with optional date truncation)
      filters     → WHERE conditions
      time_range  → additional date/time WHERE condition
      order_by    → ORDER BY clause
      limit       → LIMIT clause
    """

    metrics: list[Metric] = Field(
        default_factory=list,
        description=(
            "Columns to aggregate (the SELECT expressions). "
            "Empty means SELECT * with no aggregation."
        ),
    )
    dimensions: list[Dimension] = Field(
        default_factory=list,
        description=(
            "Columns to GROUP BY. Each entry may carry a date_trunc granularity "
            "for date/timestamp columns. Empty means no grouping."
        ),
    )
    filters: list[Filter] = Field(
        default_factory=list,
        description="WHERE clause conditions derived from the question.",
    )
    time_range: TimeRange | None = Field(
        default=None,
        description="Optional date/time filter extracted from the question.",
    )
    limit: int | None = Field(
        default=None,
        description="Row limit (TOP N questions). Null means no limit.",
    )
    order_by: list[OrderBy] = Field(
        default_factory=list,
        description="ORDER BY expressions.",
    )
    reasoning: str = Field(
        default="",
        description=(
            "Brief explanation of how you mapped the question to these query components. "
            "One to three sentences."
        ),
    )
