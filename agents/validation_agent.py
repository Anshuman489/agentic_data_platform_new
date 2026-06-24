"""
agents/validation_agent.py — Two-layer SQL validation and execution.

Layer 1 — Syntax (BigQuery dry-run):
    Submits the SQL to BigQuery with dry_run=True. This validates the full
    query — column names, types, table references — at zero cost. Any syntax
    or schema error raises BadRequest before a single byte is scanned.

Layer 2 — Semantic (Gemini reflection):
    Asks Gemini "does this SQL actually answer the user's question?". Catches
    logical errors the syntax check cannot see: wrong aggregation level,
    missing filters, incorrect metric formula, etc.

If both layers pass, the query is executed and rows are returned.
"""

import concurrent.futures
import logging

from google import genai
from google.api_core.exceptions import BadRequest
from google.genai import types
from pydantic import BaseModel

from config.settings import settings
from core.bigquery_client import BigQueryClient
from models.dataset import DatasetProfile
from models.validation import ValidationResult

logger = logging.getLogger(__name__)

_RESULT_ROW_LIMIT = 500  # safety cap on execution results


# ── Internal model for Gemini structured output ────────────────────────────────

class _SemanticCheck(BaseModel):
    """Gemini's verdict on whether the SQL correctly answers the question."""
    approved: bool
    feedback: str


# ── ValidationAgent ────────────────────────────────────────────────────────────

class ValidationAgent:
    """
    Validates a generated SQL query against the original user question.

    Requires a BigQueryClient (for dry-run and execution) and a Gemini client
    (for semantic reflection). Both use ADC — no extra credentials needed.

    Usage:
        agent = ValidationAgent(bq)
        result = agent.run(question, sql, profile)
        if result.passed:
            print(result.rows)
    """

    def __init__(self, bq: BigQueryClient) -> None:
        self._bq = bq
        self._gemini = genai.Client(
            vertexai=True,
            project=settings.gcp_project,
            location=settings.vertex_location,
        )
        self._model = settings.llm_model

    # ── Public API ─────────────────────────────────────────────────────────────

    def run(
        self,
        question: str,
        sql: str,
        profile: DatasetProfile,
    ) -> ValidationResult:
        """
        Run both validation layers, then execute if both pass.

        Args:
            question: The original natural-language question from the user.
            sql:      BigQuery SQL produced by SqlGenerationAgent.
            profile:  DatasetProfile of the target table.

        Returns:
            A ValidationResult with the outcome of each layer and query rows
            if execution was reached.
        """
        # ── Layer 1: syntax ────────────────────────────────────────────────────
        syntax_error = self._syntax_check(sql)
        if syntax_error:
            logger.warning("Syntax check FAILED: %s", syntax_error[:120])
            return ValidationResult(
                sql=sql,
                syntax_valid=False,
                syntax_error=syntax_error,
            )
        logger.info("Syntax check PASSED")

        # ── Layer 2: semantic ──────────────────────────────────────────────────
        check = self._semantic_check(question, sql, profile)
        logger.info(
            "Semantic check %s — %s",
            "PASSED" if check.approved else "FAILED",
            check.feedback[:120],
        )

        if not check.approved:
            return ValidationResult(
                sql=sql,
                syntax_valid=True,
                semantic_valid=False,
                semantic_feedback=check.feedback,
            )

        # ── Execute ────────────────────────────────────────────────────────────
        rows = self._execute(sql)
        logger.info("Query executed — %d rows returned", len(rows))

        return ValidationResult(
            sql=sql,
            syntax_valid=True,
            semantic_valid=True,
            semantic_feedback=check.feedback,
            rows=rows,
            total_rows=len(rows),
        )

    # ── Private: Layer 1 ───────────────────────────────────────────────────────

    def _syntax_check(self, sql: str) -> str | None:
        """
        BigQuery dry-run. Returns None when SQL is valid, error string otherwise.

        dry_run=True validates the full query — columns, types, functions,
        table existence — without executing it or scanning any data.
        """
        try:
            self._bq.run_query(sql, dry_run=True)
            return None
        except BadRequest as exc:
            # exc.message contains BQ's structured error (location, reason, etc.)
            return exc.message
        except Exception as exc:
            return str(exc)

    # ── Private: Layer 2 ───────────────────────────────────────────────────────

    def _semantic_check(
        self,
        question: str,
        sql: str,
        profile: DatasetProfile,
    ) -> _SemanticCheck:
        """
        Ask Gemini whether the SQL correctly answers the original question.

        Provides the schema so Gemini can spot missing columns, wrong granularity,
        or incorrect aggregations that the dry-run cannot detect.
        """
        schema_summary = "\n".join(
            f"  - {c.name} ({c.bq_type}, role={c.inferred_role})"
            for c in profile.columns
        )

        prompt = (
            f'User question: "{question}"\n\n'
            f"Generated SQL:\n{sql}\n\n"
            f"Available columns in {profile.table_ref}:\n{schema_summary}\n\n"
            "Does this SQL correctly answer the user's question?\n"
            "Check: correct columns, right aggregation, proper filters, "
            "correct grouping granularity (e.g. per-month vs per-row)."
        )

        config = types.GenerateContentConfig(
            system_instruction=(
                "You are a SQL reviewer for a data analytics platform. "
                "Evaluate whether the generated BigQuery SQL correctly answers "
                "the user's analytical question. Approve only if the SQL would "
                "return exactly what the user asked for. Be concise in feedback."
            ),
            response_mime_type="application/json",
            response_schema=_SemanticCheck,
            temperature=0.0,
        )

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                self._gemini.models.generate_content,
                model=self._model,
                contents=prompt,
                config=config,
            )
            try:
                response = future.result(timeout=45)
            except concurrent.futures.TimeoutError:
                raise TimeoutError("Gemini did not respond within 45 seconds — try again or switch to a faster model")

        return _SemanticCheck.model_validate_json(response.text)

    # ── Private: execution ─────────────────────────────────────────────────────

    def _execute(self, sql: str) -> list[dict]:
        """
        Run the validated SQL and return at most _RESULT_ROW_LIMIT rows.

        A safety LIMIT is appended when the SQL does not already have one,
        so a runaway SELECT * never returns millions of rows to the caller.
        """
        upper = sql.upper()
        if "LIMIT" not in upper:
            sql = f"{sql}\nLIMIT {_RESULT_ROW_LIMIT}"

        return self._bq.run_query(sql)
