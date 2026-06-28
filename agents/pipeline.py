"""
agents/pipeline.py — End-to-end query pipeline with retry.

Chains SqlGenerationAgent -> ValidationAgent.
On failure the validator's feedback is passed back to SqlGenerationAgent so
the second attempt can correct both syntax errors and semantic mismatches.
"""

import logging

from agents.sql_generation_agent import SqlGenerationAgent
from agents.validation_agent import ValidationAgent
from models.dataset import DatasetProfile
from models.validation import ValidationResult

logger = logging.getLogger(__name__)

_MAX_ATTEMPTS = 2

_UNSUPPORTED_PATTERNS = (
    "subquery", "sub-query", "nested query", "cte", "with clause",
    "window function", "partition by", "two-level", "two level",
    "aggregate of aggregate", "group by invoiceno", "group by order",
    "per order", "per invoice",
)


def _is_structural_failure(feedback: str) -> bool:
    lower = feedback.lower()
    return any(p in lower for p in _UNSUPPORTED_PATTERNS)


def run_pipeline(
    question: str,
    profile: DatasetProfile,
    sql_agent: SqlGenerationAgent,
    val_agent: ValidationAgent,
) -> ValidationResult:
    """
    Run the full query pipeline with up to _MAX_ATTEMPTS attempts.

    On syntax failure the BQ error is fed back as feedback so Gemini can
    correct column names or function calls. On semantic failure the validator's
    feedback is fed back so Gemini can fix wrong granularity, missing filters, etc.

    Args:
        question:  Natural-language question from the user.
        profile:   DatasetProfile of the target table.
        sql_agent: SqlGenerationAgent instance.
        val_agent: ValidationAgent instance (holds BigQueryClient).

    Returns:
        The ValidationResult from the final attempt.
    """
    feedback: str | None = None

    for attempt in range(1, _MAX_ATTEMPTS + 1):
        logger.info("Pipeline attempt %d/%d for: %s", attempt, _MAX_ATTEMPTS, question[:80])

        sql, answer_template = sql_agent.run(question, profile, feedback=feedback)
        result = val_agent.run(question, sql, profile)
        result = result.model_copy(update={"answer_template": answer_template})

        if result.passed:
            if attempt > 1:
                logger.info("Pipeline succeeded on attempt %d", attempt)
            return result

        if attempt == _MAX_ATTEMPTS:
            break

        if not result.syntax_valid:
            logger.info(
                "Syntax check failed (attempt %d) — retrying with error: %s",
                attempt, result.syntax_error,
            )
            feedback = f"Syntax error from BigQuery: {result.syntax_error}"
            continue

        if not result.semantic_valid:
            if _is_structural_failure(result.semantic_feedback or ""):
                logger.warning(
                    "Semantic failure requires unsupported SQL feature — skipping retry: %s",
                    result.semantic_feedback,
                )
                return ValidationResult(
                    sql=result.sql,
                    syntax_valid=True,
                    semantic_valid=False,
                    semantic_feedback=(
                        "This question requires a multi-level aggregation or subquery "
                        "which is not yet supported. Try rephrasing — for example, "
                        "'total revenue per customer' instead of 'average order value'."
                    ),
                )
            logger.info(
                "Semantic check failed (attempt %d) — retrying with feedback: %s",
                attempt, result.semantic_feedback,
            )
            feedback = result.semantic_feedback

    return result
