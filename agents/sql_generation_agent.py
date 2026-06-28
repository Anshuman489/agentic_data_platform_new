"""
agents/sql_generation_agent.py — Generates BigQuery SQL from a natural language question.

Uses Vertex AI Gemini to write SQL directly from the question and DatasetProfile.
The profile provides rich context (column types, cardinalities, roles, sample values)
so Gemini can make accurate column choices and handle complex SQL patterns like
subqueries, window functions, CTEs, and multi-condition filters.
"""

import concurrent.futures
import logging

from google import genai
from google.genai import types
from pydantic import BaseModel

from config.settings import settings
from models.dataset import DatasetProfile

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a BigQuery SQL expert. Given a natural language question and a table schema, "
    "write a single valid BigQuery SQL query that correctly answers the question.\n\n"
    "The question may contain spelling mistakes — interpret them based on semantic intent "
    "and the available column names in the schema. 'quntity' means 'Quantity', "
    "'prise' means 'UnitPrice', 'contry' means 'Country', etc.\n\n"
    "Rules:\n"
    "- Use only columns that exist in the schema — never invent column names.\n"
    "- Always backtick-quote column names and the fully-qualified table name.\n"
    "- Always use a consistent table alias (t) for the main table.\n"
    "- Use standard BigQuery SQL syntax (not legacy SQL).\n"
    "- For date/time grouping use DATE_TRUNC or EXTRACT as appropriate.\n"
    "- For ranking or percentages use window functions when needed.\n"
    "- Add ORDER BY and LIMIT where the question implies them.\n"
    "- Always give every SELECT expression a meaningful alias using AS "
    "(e.g. AVG(order_total) AS avg_order_value, COUNT(*) AS trip_count).\n"
    "- Always use the column alias name in GROUP BY, not a positional index "
    "(e.g. GROUP BY country, not GROUP BY 1).\n"
    "- For numeric measure columns (prices, quantities, amounts), always filter out "
    "zero and negative values in WHERE unless the question explicitly asks about "
    "returns, refunds, cancellations, or negative balances.\n"
    "- When listing items that match a filter condition (e.g. 'which products have X'), "
    "always include COUNT(*) AS occurrence_count (never COUNT(column)) and the SUM of "
    "the filtered numeric column (e.g. SUM(Quantity) AS total_quantity). "
    "Order by occurrence_count DESC so the most frequent items appear first.\n\n"
    "Also return an answer_template: a 1-2 sentence plain English answer using "
    "{column_alias} placeholders that exactly match the column aliases in your SQL. "
    "The template will be filled with actual values from the first result row.\n"
    "Example — SQL has: SELECT Country, SUM(...) AS total_revenue ...\n"
    "Template: '{Country} generated the highest revenue at {total_revenue}.'\n"
    "For list queries (top N): '{first_col} leads with {second_col}.'\n"
    "Use only aliases that appear in your SELECT — never invent new ones."
)


class _SqlOutput(BaseModel):
    sql: str
    answer_template: str


class SqlGenerationAgent:
    """
    Generates a BigQuery SQL query from a natural language question.

    Uses Gemini on Vertex AI with structured output to guarantee a clean SQL
    string is returned. The DatasetProfile is formatted as a schema table so
    Gemini has full context — column types, roles, cardinalities, and sample
    values — to make accurate decisions without hallucinating column names.

    Usage:
        agent = SqlGenerationAgent()
        sql = agent.run("top 5 pickup locations by trip count", profile)
    """

    def __init__(self) -> None:
        self._client = genai.Client(
            vertexai=True,
            project=settings.gcp_project,
            location=settings.vertex_location,
        )
        self._model = settings.llm_model

    def run(
        self,
        question: str,
        profile: DatasetProfile,
        feedback: str | None = None,
    ) -> tuple[str, str]:  # (sql, answer_template)
        """
        Generate a BigQuery SQL query for the given question.

        Args:
            question: Natural-language question from the user.
            profile:  DatasetProfile of the target table.
            feedback: Optional feedback from a failed validation attempt.
                      When provided, Gemini is told what was wrong so it can
                      generate a corrected query on retry.

        Returns:
            A BigQuery SQL string ready for dry-run or execution.
        """
        prompt = self._build_prompt(question, profile, feedback)

        logger.info(
            "SqlGenerationAgent.run: model=%s table=%s question='%s'%s",
            self._model,
            profile.table_ref,
            question[:80],
            " [retry]" if feedback else "",
        )

        config = types.GenerateContentConfig(
            system_instruction=_SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=_SqlOutput,
            temperature=0.0,
            seed=42,
        )

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                self._client.models.generate_content,
                model=self._model,
                contents=prompt,
                config=config,
            )
            try:
                response = future.result(timeout=45)
            except concurrent.futures.TimeoutError:
                raise TimeoutError(
                    "Gemini did not respond within 45 seconds — try again or switch to a faster model"
                )

        output = _SqlOutput.model_validate_json(response.text)
        logger.debug("Generated SQL:\n%s", output.sql)
        return output.sql, output.answer_template

    def _build_prompt(
        self, question: str, profile: DatasetProfile, feedback: str | None
    ) -> str:
        schema = self._format_profile(profile)
        prompt = f"Question: {question}\n\n{schema}"

        if feedback:
            prompt += (
                f"\n\nIMPORTANT — a previous attempt was rejected:\n"
                f"{feedback}\n\n"
                f"Generate a corrected SQL query that fixes this issue."
            )

        return prompt

    def _format_profile(self, profile: DatasetProfile) -> str:
        lines = [
            f"Table: `{profile.table_ref}` — {profile.row_count:,} rows",
            "",
            "| Column | Type | Role | Cardinality | Sample / Range |",
            "| ------ | ---- | ---- | ----------- | -------------- |",
        ]

        for col in profile.columns:
            cardinality = f"{col.cardinality:,}" if col.cardinality is not None else "-"

            extra = ""
            if col.numeric_min is not None and col.numeric_max is not None:
                extra = f"{col.numeric_min:g} - {col.numeric_max:g}"
            elif col.date_min and col.date_max:
                extra = f"{col.date_min} - {col.date_max}"
            elif col.sample_values:
                extra = ", ".join(str(v) for v in col.sample_values[:4])

            lines.append(
                f"| {col.name} | {col.bq_type} | {col.inferred_role} "
                f"| {cardinality} | {extra} |"
            )

        return "\n".join(lines)
