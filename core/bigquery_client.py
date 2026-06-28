"""
core/bigquery_client.py — Thin wrapper around google-cloud-bigquery.

Single responsibility: execute BigQuery API calls and return plain Python types.
No schema analysis, no LLM calls, no business logic — those live in the agents
above this layer.
"""

import logging
from typing import Any

from google.api_core.exceptions import GoogleAPIError, NotFound
from google.cloud import bigquery

from config.settings import settings

logger = logging.getLogger(__name__)


class BigQueryClient:
    """
    Wraps google.cloud.bigquery.Client with project-level defaults from Settings.

    One instance per process is sufficient — the underlying BQ client is
    thread-safe and maintains its own connection pool internally.
    """

    def __init__(self) -> None:
        """
        Instantiate the underlying BQ client using Application Default Credentials (ADC).

        Run `gcloud auth application-default login` once before using this class.
        The project and location are pulled from Settings so callers never
        hard-code a project ID or region.
        """
        # Pass project explicitly so every API call targets the correct GCP project
        # even if the ADC credentials belong to a different project.
        self._client = bigquery.Client(project=settings.gcp_project)

        # Location is stored separately because it is supplied per-job (QueryJobConfig),
        # not on the client constructor. BigQuery rejects cross-region queries.
        self._location = settings.bq_location

    # ── Public API ─────────────────────────────────────────────────────────────

    def get_table(self, table_ref: str) -> bigquery.Table:
        """
        Fetch BigQuery table metadata without scanning any rows.

        This is a single lightweight metadata API call — not a query. It is
        how the platform reads the schema and row/byte counts of a table.

        Args:
            table_ref: Fully-qualified table reference in the form
                       "project.dataset.table" or "dataset.table".
                       When the project is omitted, settings.gcp_project is used.

        Returns:
            A google.cloud.bigquery.Table object with:
            - .schema      → list[SchemaField]  column names, types, modes, descriptions
            - .num_rows    → int   approximate row count (from table metadata, no scan)
            - .num_bytes   → int   table size in bytes
            - .description → str | None  table-level description set in BQ

        Raises:
            google.api_core.exceptions.NotFound:      Table does not exist in BigQuery.
            google.api_core.exceptions.GoogleAPIError: Any other GCP API failure
                                                       (auth, quota, network).
        """
        try:
            return self._client.get_table(table_ref)
        except NotFound:
            logger.error("Table not found in BigQuery: %s", table_ref)
            raise
        except GoogleAPIError:
            logger.exception("GCP API error while fetching table '%s'", table_ref)
            raise

    def run_query(
        self,
        sql: str,
        dry_run: bool = False,
        location: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Execute a Standard SQL query and return results as plain Python dicts.

        Args:
            sql:      Standard SQL query string. Legacy SQL is not supported.
            dry_run:  If True, validates the query and logs the estimated bytes
                      that would be processed, but does not execute it.
                      Returns an empty list when dry_run=True.

        Returns:
            For dry_run=False: a list of row dicts, e.g.
                [{"word": "hamlet", "word_count": 100}, ...]
            For dry_run=True:  [] — BigQuery does not return rows for dry-run jobs.

        Raises:
            google.api_core.exceptions.BadRequest:     SQL is syntactically invalid.
            google.api_core.exceptions.GoogleAPIError: Any other GCP API failure.
        """
        job_config = bigquery.QueryJobConfig(
            dry_run=dry_run,
            use_legacy_sql=False,
        )

        try:
            query_job = self._client.query(
                sql,
                job_config=job_config,
                location=location or self._location,
            )
        except GoogleAPIError:
            logger.exception("Failed to submit query to BigQuery")
            raise

        if dry_run:
            # BQ populates total_bytes_processed on the job object immediately
            # for dry-run jobs — no data is scanned, but the planner runs.
            # Use this to surface cost estimates to callers via the log.
            estimated_mb = (query_job.total_bytes_processed or 0) / 1_048_576
            logger.info(
                "dry_run: query would process ~%.2f MB (%s bytes)",
                estimated_mb,
                f"{query_job.total_bytes_processed:,}",
            )
            return []

        try:
            # .result() blocks until the job completes and raises on failure.
            results = query_job.result()
        except GoogleAPIError:
            logger.exception("Query execution failed (job_id=%s)", query_job.job_id)
            raise

        # Convert BQ Row objects → plain dicts so callers have no BQ dependency.
        # dict(row) uses column names as keys, preserving BQ's schema ordering.
        return [dict(row) for row in results]
