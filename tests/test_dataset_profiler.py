from unittest.mock import MagicMock
from google.cloud.bigquery import SchemaField, Table
from core.dataset_profiler import DatasetProfiler


def _make_schema(*fields):
    """Helper: build a list of SchemaField objects."""
    return fields


def _make_mock_bq(schema_fields, agg_row, sample_rows):
    """Returns a mock BigQueryClient with controlled responses."""
    mock_table = MagicMock()
    mock_table.schema = schema_fields

    mock_bq = MagicMock()
    mock_bq.get_table.return_value = mock_table
    mock_bq.run_query.side_effect = [
        [agg_row],   # first call → aggregation query
        sample_rows, # second call → SELECT * LIMIT N
    ]
    return mock_bq


def test_string_dimension():
    schema = [SchemaField("country", "STRING", mode="NULLABLE")]
    agg_row = {"c0__nf": 0.0, "c0__cd": 5}
    sample_rows = [{"country": "US"}, {"country": "UK"}]

    bq = _make_mock_bq(schema, agg_row, sample_rows)
    profiler = DatasetProfiler(bq)
    columns, rows = profiler.profile("p.d.t")

    assert columns[0].inferred_role == "DIMENSION"
    assert columns[0].cardinality == 5
    assert "US" in columns[0].sample_values


def test_numeric_identifier():
    schema = [SchemaField("user_id", "INT64", mode="NULLABLE")]
    agg_row = {"c0__nf": 0.0, "c0__cd": 1_000_000}

    bq = _make_mock_bq(schema, agg_row, [])
    profiler = DatasetProfiler(bq)
    columns, _ = profiler.profile("p.d.t")

    assert columns[0].inferred_role == "IDENTIFIER"


def test_date_column():
    schema = [SchemaField("created_at", "TIMESTAMP", mode="NULLABLE")]
    agg_row = {"c0__nf": 0.02, "c0__cd": 500,
               "c0__dmin": "2023-01-01", "c0__dmax": "2024-01-01"}

    bq = _make_mock_bq(schema, agg_row, [])
    profiler = DatasetProfiler(bq)
    columns, _ = profiler.profile("p.d.t")

    assert columns[0].inferred_role == "DATE"
    assert columns[0].date_min == "2023-01-01"
    assert columns[0].date_max == "2024-01-01"


def test_repeated_column_skipped():
    schema = [SchemaField("tags", "STRING", mode="REPEATED")]
    # agg_row is empty because REPEATED columns are skipped in the query
    bq = _make_mock_bq(schema, {}, [])
    profiler = DatasetProfiler(bq)
    columns, _ = profiler.profile("p.d.t")

    assert columns[0].inferred_role == "UNKNOWN"
    assert columns[0].cardinality is None