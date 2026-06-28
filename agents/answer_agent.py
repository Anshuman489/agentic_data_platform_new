"""
agents/answer_agent.py — Narrates query results as a plain English sentence.

Takes the user's original question and the first N rows of results, then asks
Gemini to write a 1-2 sentence answer that includes the actual numbers.
"""

import concurrent.futures
import logging
from typing import Any

from google import genai
from google.genai import types
from pydantic import BaseModel

from config.settings import settings

logger = logging.getLogger(__name__)


class _NLAnswer(BaseModel):
    answer: str


class AnswerAgent:
    """
    Generates a plain-English answer from query results.

    Usage:
        agent = AnswerAgent()
        answer = agent.run("Which country has most revenue?", rows, total_rows)
        # "United Kingdom generated the highest revenue at £9,025,222, representing..."
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
        rows: list[dict[str, Any]],
        total_rows: int,
    ) -> str:
        if not rows:
            return "The query returned no results."

        sample = rows[:10]
        rows_text = "\n".join(
            ", ".join(f"{k}: {v}" for k, v in row.items())
            for row in sample
        )

        prompt = (
            f'User asked: "{question}"\n\n'
            f"Query returned {total_rows} row(s). Sample:\n{rows_text}\n\n"
            "Write 1-2 sentences that directly answer the question. "
            "Include the key number or finding. Be specific — mention actual values from the data."
        )

        config = types.GenerateContentConfig(
            system_instruction=(
                "You are a data analyst narrating query results in plain English. "
                "Give a concise, specific answer that always includes actual numbers from the data. "
                "Never say 'the data shows' — just state the finding directly."
            ),
            response_mime_type="application/json",
            response_schema=_NLAnswer,
            temperature=0.0,
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
                logger.warning("AnswerAgent timed out — skipping NL answer")
                return ""

        return _NLAnswer.model_validate_json(response.text).answer
