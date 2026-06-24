import logging
logging.basicConfig(level=logging.INFO)

from core.bigquery_client import BigQueryClient
from agents.schema_discovery_agent import SchemaDiscoveryAgent

bq = BigQueryClient()
agent = SchemaDiscoveryAgent(bq)

# First run — cache miss, hits BigQuery
profile = agent.run("bigquery-public-data.new_york_taxi_trips.tlc_yellow_trips_2022")
print(f"Columns: {len(profile.columns)}")
print(f"Rows: {profile.row_count:,}")
print(f"Cached at: {profile.profiled_at}")

# Second run — should say "Cache hit", no BQ queries
profile2 = agent.run("bigquery-public-data.new_york_taxi_trips.tlc_yellow_trips_2022")
print(f"Same profile: {profile.profiled_at == profile2.profiled_at}")