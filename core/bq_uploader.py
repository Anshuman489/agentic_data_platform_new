"""
core/bq_uploader.py — Upload a CSV or Excel file directly to BigQuery.

Converts Excel to CSV in-memory if needed, then uses BigQuery's
load_table_from_file with autodetect=True so the user does not need
to define a schema manually.
"""

import io
import logging
from pathlib import Path

from google.cloud import bigquery

from config.settings import settings

logger = logging.getLogger(__name__)


def upload_file_to_bigquery(
    file_bytes: bytes,
    filename: str,
    table_ref: str,
    location: str = "",
) -> int:
    """
    Upload a CSV or Excel file to a BigQuery table, replacing it if it exists.

    Args:
        file_bytes: Raw bytes of the uploaded file.
        filename:   Original filename — used to detect CSV vs Excel.
        table_ref:  Fully-qualified BQ table reference: "project.dataset.table".
        location:   BQ region, e.g. "US" or "us-central1". Falls back to settings.

    Returns:
        Number of rows loaded.

    Raises:
        ValueError: Unsupported file type.
        google.api_core.exceptions.GoogleAPIError: BQ load job failure.
    """
    suffix = Path(filename).suffix.lower()

    if suffix in (".xlsx", ".xls"):
        import pandas as pd
        df = pd.read_excel(io.BytesIO(file_bytes))
        buf = io.BytesIO()
        df.to_csv(buf, index=False)
        buf.seek(0)
        data: io.IOBase = buf
    elif suffix == ".csv":
        data = io.BytesIO(file_bytes)
    else:
        raise ValueError(
            f"Unsupported file type '{suffix}'. Please upload a CSV or Excel file."
        )

    client = bigquery.Client(project=settings.gcp_project)
    job_config = bigquery.LoadJobConfig(
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
        source_format=bigquery.SourceFormat.CSV,
        skip_leading_rows=1,
        autodetect=True,
    )

    job = client.load_table_from_file(
        data,
        table_ref,
        job_config=job_config,
        location=location or settings.bq_location,
    )
    job.result()

    table = client.get_table(table_ref)
    rows = table.num_rows or 0
    logger.info("Uploaded %d rows to %s", rows, table_ref)
    return rows
