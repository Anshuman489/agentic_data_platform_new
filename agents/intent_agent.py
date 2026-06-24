"""
agents/intent_agent.py — Translates a natural-language question into a structured query intent.

Uses Gemini on Vertex AI with structured output (response_schema) to extract
IntentResult fields from a free-text question, grounded in the table's DatasetProfile.

Authentication is via Application Default Credentials (ADC) — the same credentials
already used for BigQuery. No extra API key is required.
"""

import concurrent.futures
import logging

from google import genai
from google.genai import types

from config.settings import settings
from models.dataset import DatasetProfile
from models.intent import IntentResult

logger = logging.getLogger(__name__)


class IntentAgent:
    """
    Converts a natural-language question into a validated IntentResult.

    Mechanism: sends the question + a compact table schema to Gemini via Vertex AI
    and forces a JSON response that matches the IntentResult schema exactly.
    Gemini's structured-output mode (`response_mime_type="application/json"` +
    `response_schema=IntentResult`) guarantees the reply is always valid — no
    free-text parsing needed.

    Usage:
        agent = IntentAgent()
        intent = agent.run("Top 5 countries by total quantity sold?", profile)
    """

    def __init__(self) -> None:
        self._client = genai.Client(
            vertexai=True,
            project=settings.gcp_project,
            location=settings.vertex_location,
        )
        self._model = settings.llm_model

    # ── Public API ─────────────────────────────────────────────────────────────

    def run(
        self,
        question: str,
        profile: DatasetProfile,
        feedback: str | None = None,
    ) -> IntentResult:
        """
        Parse a natural-language question against a DatasetProfile.

        Makes a single Vertex AI call. The model is constrained to return JSON
        matching IntentResult's schema — no tool-use loop, no post-hoc parsing.

        Args:
            question: Free-text analytical question about the table.
            profile:  DatasetProfile describing columns, types, and statistics.
            feedback: Optional semantic feedback from a prior validation attempt.
                      When provided, Gemini is told what was wrong so it can
                      generate a corrected IntentResult on retry.

        Returns:
            A validated IntentResult instance.

        Raises:
            google.api_core.exceptions.GoogleAPIError: if the Vertex AI call fails.
            pydantic.ValidationError: if the model returns structurally invalid JSON.
        """
        system_prompt = self._build_system_prompt(profile)
        contents = self._build_contents(question, feedback)

        logger.info(
            "IntentAgent.run: model=%s table=%s question='%s'%s",
            self._model,
            profile.table_ref,
            question[:100],
            " [retry with feedback]" if feedback else "",
        )

        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            response_mime_type="application/json",
            response_schema=IntentResult,
            temperature=0.0,
        )

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                self._client.models.generate_content,
                model=self._model,
                contents=contents,
                config=config,
            )
            try:
                response = future.result(timeout=45)
            except concurrent.futures.TimeoutError:
                raise TimeoutError("Gemini did not respond within 45 seconds — try again or switch to a faster model")

        raw_json = response.text
        logger.debug("IntentAgent raw response:\n%s", raw_json)

        result = IntentResult.model_validate_json(raw_json)
        return result

    # ── Private: prompt construction ───────────────────────────────────────────

    def _build_contents(self, question: str, feedback: str | None) -> str:
        """
        Build the user-turn content sent to Gemini.

        On first attempt this is just the question. On retry the semantic
        feedback from the validator is appended so Gemini knows what to fix.
        """
        if not feedback:
            return question
        return (
            f"{question}\n\n"
            f"IMPORTANT — your previous attempt was rejected by the SQL validator "
            f"for this reason:\n\n"
            f"  {feedback}\n\n"
            f"Generate a corrected IntentResult that fixes this issue."
        )

    def _build_system_prompt(self, profile: DatasetProfile) -> str:
        """Build the system instruction that provides table context to Gemini."""
        return (
            "You are a data analytics assistant. Your task is to translate a "
            "natural-language question into structured BigQuery query parameters "
            "for the table described below.\n\n"
            "Rules:\n"
            "- Use only column names that appear in the schema below — do not invent columns.\n"
            "- Match column names exactly (case-sensitive).\n"
            "- You MUST respond with valid JSON matching the IntentResult schema.\n\n"
            "## Table Schema\n\n"
            f"{self._format_profile(profile)}"
        )

    def _format_profile(self, profile: DatasetProfile) -> str:
        """
        Render the DatasetProfile as a compact markdown table for the prompt.

        Only includes what the model needs for SQL generation: column name, type,
        inferred role, cardinality, and a representative sample or numeric range.
        Keeps token count low — the full DatasetProfile JSON is never sent.
        """
        lines: list[str] = [
            f"**{profile.table_ref}** — {profile.row_count:,} rows",
            "",
            "| Column | Type | Role | Cardinality | Sample / Range |",
            "| ------ | ---- | ---- | ----------- | -------------- |",
        ]

        for col in profile.columns:
            cardinality = f"{col.cardinality:,}" if col.cardinality is not None else "—"

            extra = ""
            if col.numeric_min is not None and col.numeric_max is not None:
                extra = f"{col.numeric_min:g} – {col.numeric_max:g}"
            elif col.date_min and col.date_max:
                extra = f"{col.date_min} – {col.date_max}"
            elif col.sample_values:
                samples = [str(v) for v in col.sample_values[:4]]
                extra = ", ".join(samples)

            lines.append(
                f"| {col.name} | {col.bq_type} | {col.inferred_role} "
                f"| {cardinality} | {extra} |"
            )

        return "\n".join(lines)
