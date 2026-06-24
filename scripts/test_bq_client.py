import logging
logging.basicConfig(level=logging.INFO)

from core.bigquery_client import BigQueryClient

client = BigQueryClient()

# 1. get_table — uses a public BQ dataset, no billing required for metadata
table = client.get_table("bigquery-public-data.samples.shakespeare")
print(f"Table:   {table.table_id}")
print(f"Rows:    {table.num_rows:,}")
print(f"Bytes:   {table.num_bytes:,}")
print(f"Columns: {[f.name for f in table.schema]}")

# 2. dry_run — validates SQL and logs estimated cost, returns []
result = client.run_query(
    "SELECT word, word_count FROM `bigquery-public-data.samples.shakespeare` LIMIT 5",
    dry_run=True,
)
print(f"dry_run result: {result}")  # []

# 3. run_query — actual execution, 5 rows billed
rows = client.run_query(
    "SELECT word, word_count FROM `bigquery-public-data.samples.shakespeare` LIMIT 5"
)
for row in rows:
    print(row)