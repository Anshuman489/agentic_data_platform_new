"""
agents/table_router_agent.py — Routes a natural-language question to the best
cached table using Gemini.

For each cached DatasetProfile Gemini assigns a confidence score (0.0–1.0) and
a one-sentence reason. The agent picks the winner and flags ambiguity when two
tables score within 15 points of each other at >= 50% confidence.
"""

import concurrent.futures
import logging

from google import genai
from google.genai import types
from pydantic import BaseModel

from agents.catalog_manager import CatalogManager
from config.settings import settings
from models.catalog import RouteResult
from models.dataset import DatasetProfile

logger = logging.getLogger(__name__)

_MIN_CONFIDENCE = 0.50
_AMBIGUITY_GAP = 0.15

_SYSTEM_PROMPT = (
    "You are a data routing expert. Given a natural language question and a list of "
    "BigQuery tables (each with column names, types, roles, and sample values), decide "
    "which table is best suited to answer the question.\n\n"
    "Score each table's ability to answer the question from 0.0 to 1.0:\n"
    "  1.0 — perfect: the table has exactly the right columns and domain\n"
    "  0.7 — good: most relevant columns are present\n"
    "  0.4 — partial: some overlap but missing key columns or wrong domain\n"
    "  0.1 — poor: almost no relevant columns\n"
    "  0.0 — none: completely different domain\n\n"
    "Be discriminating. If two tables both have revenue columns, use domain context "
    "and sample values to distinguish them — do not score both at 0.9.\n"
    "Return a score and one-sentence reasoning for EVERY table."
)


class _TableScore(BaseModel):
    table_ref: str
    confidence: float
    reasoning: str


class _RouteOutput(BaseModel):
    scores: list[_TableScore]


class TableRouterAgent:
    """
    Scores every cached table against the question and picks the best match.

    Usage:
        router = TableRouterAgent(CatalogManager())
        result = router.route("total revenue by country")
        print(result.table_ref, result.confidence, result.reasoning)
    """

    def __init__(self, catalog: CatalogManager) -> None:
        self._catalog = catalog
        self._client = genai.Client(
            vertexai=True,
            project=settings.gcp_project,
            location=settings.vertex_location,
        )
        self._model = settings.llm_model

    def route(self, question: str) -> RouteResult:
        profiles = self._catalog.list()

        if not profiles:
            raise ValueError(
                "No profiled tables found in cache. "
                "Run: python main.py --register --table PROJECT.DATASET.TABLE"
            )

        if len(profiles) == 1:
            return RouteResult(
                table_ref=profiles[0].table_ref,
                confidence=1.0,
                reasoning="Only one table in cache — selected automatically.",
            )

        prompt = self._build_prompt(question, profiles)
        config = types.GenerateContentConfig(
            system_instruction=_SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=_RouteOutput,
            temperature=0.0,
        )

        logger.info("TableRouterAgent: scoring %d tables for question", len(profiles))

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
                raise TimeoutError("Table router did not respond within 45 seconds.")

        output = _RouteOutput.model_validate_json(response.text)
        return self._resolve(output)

    def _resolve(self, output: _RouteOutput) -> RouteResult:
        if not output.scores:
            raise ValueError("Router returned no scores.")

        scores = sorted(output.scores, key=lambda s: s.confidence, reverse=True)
        best = scores[0]

        if best.confidence < _MIN_CONFIDENCE:
            raise ValueError(
                f"No table is confident enough to answer this question "
                f"(best: '{best.table_ref}' at {best.confidence:.0%}). "
                f"Try rephrasing or profile a more relevant table first."
            )

        ambiguous = (
            len(scores) > 1
            and scores[1].confidence >= _MIN_CONFIDENCE
            and (best.confidence - scores[1].confidence) < _AMBIGUITY_GAP
        )

        return RouteResult(
            table_ref=best.table_ref,
            confidence=best.confidence,
            reasoning=best.reasoning,
            ambiguous=ambiguous,
            alternatives=[(s.table_ref, s.confidence, s.reasoning) for s in scores[1:]],
        )

    def _build_prompt(self, question: str, profiles: list[DatasetProfile]) -> str:
        lines = [f"Question: {question}\n", "Available tables:"]

        for profile in profiles:
            lines.append(f"\n--- {profile.table_ref} ---")
            if profile.table_description:
                lines.append(f"Description: {profile.table_description}")
            lines.append(f"Rows: {profile.row_count:,}")

            for col in profile.columns:
                date_range = ""
                if col.date_min and col.date_max:
                    date_range = f", range: {col.date_min} to {col.date_max}"
                samples = (
                    ", ".join(str(v) for v in col.sample_values[:3])
                    if col.sample_values
                    else "-"
                )
                lines.append(
                    f"  {col.name} ({col.bq_type}, {col.inferred_role}{date_range}) — e.g. {samples}"
                )

        return "\n".join(lines)
