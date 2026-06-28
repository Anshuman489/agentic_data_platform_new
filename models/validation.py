"""
models/validation.py — Result model for ValidationAgent.

Holds the outcome of both validation layers (syntax + semantic) and the
query rows if execution was reached.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, computed_field


class ValidationResult(BaseModel):
    """
    Complete record of what ValidationAgent decided and produced.

    Lifecycle:
      syntax_valid=False              → stopped after BQ dry-run (SQL is broken)
      syntax_valid=True,
        semantic_valid=False          → stopped after Gemini review (SQL is wrong)
      syntax_valid=True,
        semantic_valid=True           → executed; rows contains results
    """

    sql: str = Field(description="The SQL that was validated.")

    # ── Layer 1: BQ dry-run ───────────────────────────────────────────────────
    syntax_valid: bool
    syntax_error: str | None = Field(
        default=None,
        description="BigQuery error message when syntax check fails.",
    )

    # ── Layer 2: Gemini semantic check ────────────────────────────────────────
    semantic_valid: bool | None = Field(
        default=None,
        description="None when syntax check failed (semantic check was skipped).",
    )
    semantic_feedback: str | None = Field(
        default=None,
        description="Gemini's explanation — present whether approved or not.",
    )

    # ── Execution results ─────────────────────────────────────────────────────
    rows: list[dict[str, Any]] = Field(default_factory=list)
    total_rows: int = 0

    # ── NL answer template (from SQL generation) ──────────────────────────────
    answer_template: str = ""

    # ── Derived ───────────────────────────────────────────────────────────────
    @computed_field
    @property
    def passed(self) -> bool:
        """True only when both validation layers passed and results were returned."""
        return self.syntax_valid and self.semantic_valid is True
