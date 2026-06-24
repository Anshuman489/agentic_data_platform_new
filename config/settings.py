from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Central configuration for the Agentic Data Intelligence Platform.

    Values are loaded in this priority order (highest to lowest):
      1. Actual environment variables set in the shell
      2. Variables defined in a .env file in the project root
      3. Default values defined below

    pydantic-settings handles all of this automatically.
    Adding a new config value means adding one field here — nothing else.
    """

    # Tell pydantic-settings to look for a .env file in the current working directory.
    # extra="ignore" means unknown variables in .env are silently skipped rather than
    # raising a validation error — useful when .env contains comments or extra vars.
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # ── Google Cloud ──────────────────────────────────────────────────────────

    # GCP project ID — required, no default. The app will not start without this.
    gcp_project: str

    # BigQuery region. Must match the region of your dataset.
    bq_location: str = "US"

    # ── Schema Profiling ──────────────────────────────────────────────────────

    # How many rows to pull when sampling a table for example values.
    bq_sample_row_limit: int = 10

    # String column cardinality at or below this value → classified as DIMENSION.
    # Above this value → FREE_TEXT or IDENTIFIER.
    dimension_cardinality_threshold: int = 100

    # Numeric column cardinality at or below this value → classified as DIMENSION.
    # Above this value → classified as MEASURE.
    measure_cardinality_threshold: int = 50

    # ── LLM (Vertex AI — Gemini) ──────────────────────────────────────────────

    # Vertex AI region for Gemini calls. "us-central1" is cheapest and has the
    # broadest model availability. You can also use "us" (multi-region) or
    # "global" for maximum availability.
    vertex_location: str = "us-central1"

    # Gemini model ID on Vertex AI.
    # Recommended defaults (cheapest to most capable):
    #   gemini-2.0-flash-lite-001   — fastest, lowest cost
    #   gemini-2.0-flash-001        — balanced (default)
    #   gemini-2.5-flash            — best quality/cost ratio
    #   gemini-2.5-pro              — highest capability
    llm_model: str = "gemini-2.0-flash-001"

    # ── Cache ─────────────────────────────────────────────────────────────────

    # Where DatasetProfile JSON files are stored on disk.
    # Path type means pydantic will automatically convert the string from .env
    # into a proper Path object — so callers get Path, never a raw string.
    cache_dir: Path = Path("./cache")

    # A cached profile older than this many hours will be re-profiled from BigQuery.
    # Set to 0 during development to always re-profile.
    cache_max_age_hours: int = 24


# ── Singleton ─────────────────────────────────────────────────────────────────
# Instantiated once at import time. Every module in the project imports this
# single instance rather than creating its own. This means .env is read exactly
# once, and all components share the same configuration values.
#
# Usage in any other file:
#   from config.settings import settings
#   print(settings.gcp_project)
settings = Settings()
