"""
agents/pipeline.py — End-to-end query pipeline with semantic retry.

Chains IntentAgent -> SqlGenerationAgent -> ValidationAgent.
If the first attempt fails the semantic check, the validator's feedback is
passed back to IntentAgent so it can generate a corrected IntentResult.
Syntax failures are not retried — they indicate a schema mismatch that
re-running intent extraction cannot fix.
"""

import logging

from agents.intent_agent import IntentAgent
from agents.sql_generation_agent import SqlGenerationAgent
from agents.validation_agent import ValidationAgent
from models.dataset import DatasetProfile
from models.validation import ValidationResult

logger = logging.getLogger(__name__)

_MAX_ATTEMPTS = 2

# Feedback keywords that signal a structural SQL limitation.
# These require subqueries, CTEs, or window functions — none of which the
# deterministic SQL generator supports. Retrying with feedback won't help.
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
    intent_agent: IntentAgent,
    sql_agent: SqlGenerationAgent,
    val_agent: ValidationAgent,
) -> ValidationResult:
    """
    Run the full query pipeline with up to _MAX_ATTEMPTS attempts.

    On a semantic failure the validator's feedback is fed back into IntentAgent
    so the second attempt corrects the specific issue (e.g. wrong granularity,
    missing filter, incorrect aggregation).

    Args:
        question:     Natural-language question from the user.
        profile:      DatasetProfile of the target table.
        intent_agent: Shared IntentAgent instance.
        sql_agent:    Shared SqlGenerationAgent instance.
        val_agent:    Shared ValidationAgent instance (holds BigQueryClient).

    Returns:
        The ValidationResult from the final attempt. Callers should check
        result.passed to determine whether execution results are available.
    """
    feedback: str | None = None

    for attempt in range(1, _MAX_ATTEMPTS + 1):
        logger.info("Pipeline attempt %d/%d for: %s", attempt, _MAX_ATTEMPTS, question[:80])

        intent = intent_agent.run(question, profile, feedback=feedback)
        sql = sql_agent.run(intent, profile)
        result = val_agent.run(question, sql, profile)

        if result.passed:
            if attempt > 1:
                logger.info("Pipeline succeeded on attempt %d", attempt)
            return result

        if not result.syntax_valid:
            # Syntax errors mean the SQL references something BQ doesn't recognise.
            # IntentAgent retry won't help — return immediately.
            logger.warning("Syntax check failed — not retrying: %s", result.syntax_error)
            return result

        if attempt < _MAX_ATTEMPTS and not result.semantic_valid:
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
                attempt,
                result.semantic_feedback,
            )
            feedback = result.semantic_feedback

    return result
