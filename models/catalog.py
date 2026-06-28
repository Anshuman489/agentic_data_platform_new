from pydantic import BaseModel


class RouteResult(BaseModel):
    table_ref: str
    confidence: float
    reasoning: str
    ambiguous: bool = False
    alternatives: list[tuple[str, float, str]] = []  # (table_ref, confidence, reasoning)
