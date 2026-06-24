import logging
logging.basicConfig(level=logging.INFO)

from core.bigquery_client import BigQueryClient
from core.dataset_profiler import DatasetProfiler

bq = BigQueryClient()
profiler = DatasetProfiler(bq)

columns, sample_rows = profiler.profile(
    "bigquery-public-data.new_york_taxi_trips.tlc_yellow_trips_2022"
)

for col in columns:
    print(f"{col.name:20s} {col.bq_type:12s} {col.inferred_role:12s} "
          f"cardinality={col.cardinality}  null={col.null_fraction:.2f}")

print(f"\nSample rows ({len(sample_rows)}):")
for row in sample_rows[:3]:
    print(row)