# Agentic Data Intelligence Platform

Ask questions about your data in plain English. The platform finds the right table, writes the SQL, validates it, and explains the answer.

---

## What it does

- **Natural language → SQL** — type a question, get an answer
- **Automatic table routing** — picks the right dataset from your catalog
- **Self-correcting** — validates its own SQL and retries on failure
- **Dataset upload** — upload CSV or Excel directly into BigQuery from the UI
- **Multi-table** — works across any number of registered BigQuery tables

---

## Prerequisites

- Python 3.11 or higher → https://www.python.org/downloads
  - During install on Windows: tick **"Add Python to PATH"**
- Git → https://git-scm.com/downloads
- A `gcp-key.json` file (provided separately by the project owner)

No Google Cloud account or gcloud CLI required.

---

## Setup

### 1. Clone the repository

```bash
git clone <repo-url>
cd agentic_data_platform_new
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it:

```bash
# Windows
.venv\Scripts\activate

# Mac / Linux
source .venv/bin/activate
```

You should see `(.venv)` in your terminal prompt.

### 3. Install dependencies

```bash
pip install -e .
```

### 4. Add your credentials

Place the `gcp-key.json` file (provided by the project owner) in the project root folder.

Create a `.env` file in the project root with the following content:

```
GCP_PROJECT=agentic-data-intelligence-poc
GOOGLE_APPLICATION_CREDENTIALS=gcp-key.json
```

---

## Register your first table

Before you can ask questions, you need to profile at least one BigQuery table. This runs once per table and saves a local cache file.

```bash
python main.py --register --table PROJECT.DATASET.TABLE
```

Example — register the online retail table:

```bash
python main.py --register --table agentic-data-intelligence-poc.agentic_analytics.online_retail_full
```

Example — register the credit card transactions table:

```bash
python main.py --register --table agentic-data-intelligence-poc.agentic_analytics.credit_card_transactions
```

Each registration takes 15–30 seconds. It reads the schema, profiles the columns, and generates a description using Gemini. The result is saved to `cache/` and used for all future queries.

List all registered tables:

```bash
python main.py --list-tables
```

---

## Run the app

```bash
streamlit run app.py
```

Your browser opens automatically at `http://localhost:8501`.

---

## Using the app

### Ask a question

Type a question in plain English and click **Ask**:

```
Which country generates the most revenue?
Show me the top 10 products by quantity sold
What percentage of transactions are fraudulent?
Show me monthly revenue trend for 2011
Which merchant category has the highest fraud rate?
```

### What you see

- **Green banner** — plain English answer with the result
- **SQL expander** — the exact query that was run
- **Results table** — up to 500 rows with currency formatting where applicable
- **Validation pills** — confirms syntax and semantic correctness

### Upload a dataset

Use the sidebar → **Upload Dataset** to upload a CSV or Excel file directly into BigQuery. The table is profiled automatically after upload and becomes immediately available to query.

---

## CLI usage

```bash
# Register a table
python main.py --register --table PROJECT.DATASET.TABLE

# Ask a question (auto-routes to best table)
python main.py --ask "which country generates most revenue"

# Ask against a specific table
python main.py --table PROJECT.DATASET.TABLE --ask "total revenue by country"

# List all registered tables
python main.py --list-tables
```

---

## Project structure

```
agents/          Routing, SQL generation, validation, pipeline
core/            BigQuery client, profiler, uploader
models/          Data models (profile, validation result, route result)
config/          Settings loaded from .env
cache/           Auto-generated table profiles (created after --register)
app.py           Streamlit web UI
main.py          CLI entry point
```

---

## Configuration

All settings are controlled via `.env`. Defaults work out of the box — only `GCP_PROJECT` and `GOOGLE_APPLICATION_CREDENTIALS` are required.

| Variable | Default | Description |
|---|---|---|
| `GCP_PROJECT` | required | GCP project ID |
| `GOOGLE_APPLICATION_CREDENTIALS` | required | Path to service account key file |
| `BQ_LOCATION` | `US` | BigQuery dataset region |
| `CACHE_MAX_AGE_HOURS` | `24` | Hours before a table profile is refreshed |
| `LLM_MODEL` | `gemini-2.0-flash-001` | Gemini model used for all AI calls |
| `VERTEX_LOCATION` | `us-central1` | Vertex AI region |

---

## Troubleshooting

**"No tables found"**
Run `python main.py --register --table ...` to profile at least one table first.

**"403 Access Denied" on BigQuery**
The service account key may not have the right permissions. Contact the project owner.

**"No table is confident enough"**
The question doesn't match any registered table closely enough. Try rephrasing or register a more relevant table.

**App shows old results after asking a new question**
This is fixed in the latest version. Make sure you're running the latest code.

**Stale bytecode errors after updates**
```bash
find . -name "*.pyc" -delete
find . -name "__pycache__" -exec rm -rf {} +
```
Then restart the app.
