from unittest.mock import MagicMock, patch

def test_get_table_returns_table():
    mock_table = MagicMock()
    mock_table.num_rows = 1000

    with patch("core.bigquery_client.bigquery.Client") as MockClient:
        MockClient.return_value.get_table.return_value = mock_table

        from core.bigquery_client import BigQueryClient
        client = BigQueryClient()
        result = client.get_table("project.dataset.table")

    assert result.num_rows == 1000

def test_run_query_returns_dicts():
    mock_row = {"col_a": 1, "col_b": "hello"}

    with patch("core.bigquery_client.bigquery.Client") as MockClient:
        mock_job = MagicMock()
        mock_job.result.return_value = [mock_row]
        MockClient.return_value.query.return_value = mock_job

        from core.bigquery_client import BigQueryClient
        client = BigQueryClient()
        rows = client.run_query("SELECT 1")

    assert rows == [{"col_a": 1, "col_b": "hello"}]

