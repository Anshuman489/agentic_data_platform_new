"""
app.py — Streamlit UI for the Agentic Data Intelligence Platform.

Run with:
    streamlit run app.py
"""

import logging
from decimal import Decimal
from typing import Any

import pandas as pd
import streamlit as st

from agents.answer_agent import AnswerAgent
from agents.catalog_manager import CatalogManager
from agents.pipeline import run_pipeline
from agents.schema_discovery_agent import SchemaDiscoveryAgent
from agents.sql_generation_agent import SqlGenerationAgent
from agents.table_router_agent import TableRouterAgent
from agents.validation_agent import ValidationAgent
from config.settings import settings
from core.bigquery_client import BigQueryClient
from core.bq_uploader import upload_file_to_bigquery

logging.basicConfig(level=logging.WARNING)

# ── Page config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Data Intelligence Platform",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state defaults ─────────────────────────────────────────────────────

_DEFAULTS: dict[str, Any] = {
    "stage": "idle",        # idle | ambiguous | result | error
    "question": "",
    "candidates": [],       # [(table_ref, confidence, reasoning), ...]
    "selected_table": None,
    "route_info": None,
    "result": None,
    "nl_answer": None,
    "error": None,
}

for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

if "upload_key" not in st.session_state:
    st.session_state.upload_key = 0


# ── Cached resources ───────────────────────────────────────────────────────────

@st.cache_resource
def get_bq() -> BigQueryClient:
    return BigQueryClient()


@st.cache_resource
def get_sql_agent() -> SqlGenerationAgent:
    return SqlGenerationAgent()


@st.cache_resource
def get_answer_agent() -> AnswerAgent:
    return AnswerAgent()


# ── Helpers ────────────────────────────────────────────────────────────────────

def _format_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def _reset(question: str = "") -> None:
    for k, v in _DEFAULTS.items():
        st.session_state[k] = v
    st.session_state.question = question


def _rows_to_df(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    for col in df.columns:
        if df[col].dtype == "object":
            try:
                df[col] = df[col].apply(
                    lambda x: float(x) if isinstance(x, Decimal) else x
                )
            except Exception:
                pass
    return df


# ── Pipeline runner ────────────────────────────────────────────────────────────

def _run_pipeline_for(question: str, table_ref: str) -> None:
    """Run full pipeline for a chosen table and store result in session_state."""
    bq = get_bq()

    with st.spinner(f"Generating and validating SQL..."):
        try:
            profile = SchemaDiscoveryAgent(bq).run(table_ref)
            result = run_pipeline(
                question=question,
                profile=profile,
                sql_agent=get_sql_agent(),
                val_agent=ValidationAgent(bq),
            )
        except Exception as exc:
            st.session_state.error = str(exc)
            st.session_state.stage = "error"
            return

    nl_answer = ""
    if result.passed and result.rows:
        with st.spinner("Summarising results..."):
            try:
                nl_answer = get_answer_agent().run(question, result.rows, result.total_rows)
            except Exception:
                pass

    st.session_state.result = result
    st.session_state.nl_answer = nl_answer
    st.session_state.stage = "result"


# ── Sidebar ────────────────────────────────────────────────────────────────────

def render_sidebar() -> None:
    st.sidebar.title("Data Intelligence")

    # ── Available tables ───────────────────────────────────────────────────────
    st.sidebar.header("Available Tables")

    catalog = CatalogManager()
    profiles = catalog.list()

    if not profiles:
        st.sidebar.info("No tables available. Upload a dataset below to get started.")
    else:
        for p in profiles:
            date_min = next((c.date_min for c in p.columns if c.date_min), None)
            date_max = next((c.date_max for c in p.columns if c.date_max), None)
            with st.sidebar.expander(f"**{p.table_id}**", expanded=False):
                st.write(f"**Project:** `{p.project}`")
                st.write(f"**Dataset:** `{p.dataset_id}`")
                st.write(f"**Rows:** {p.row_count:,}")
                st.write(f"**Columns:** {len(p.columns)}")
                st.write(f"**Size:** {_format_bytes(p.size_bytes)}")
                st.write(f"**Location:** {p.location or settings.bq_location}")
                if date_min:
                    st.write(f"**Date range:** {str(date_min)[:10]} → {str(date_max)[:10]}")

    st.sidebar.divider()

    # ── Upload dataset ─────────────────────────────────────────────────────────
    st.sidebar.header("Upload Dataset")

    uploaded_file = st.sidebar.file_uploader(
        "CSV or Excel file",
        type=["csv", "xlsx", "xls"],
        label_visibility="collapsed",
        key=f"file_uploader_{st.session_state.upload_key}",
    )

    if uploaded_file:
        default_table = (
            uploaded_file.name.rsplit(".", 1)[0]
            .lower()
            .replace(" ", "_")
            .replace("-", "_")
        )

        col1, col2 = st.sidebar.columns(2)
        with col1:
            dataset_id = st.text_input("Dataset ID", value="agentic_analytics")
        with col2:
            table_name = st.text_input("Table name", value=default_table)

        if st.sidebar.button("Upload to BigQuery", type="primary", use_container_width=True):
            table_ref = f"{settings.gcp_project}.{dataset_id}.{table_name}"
            with st.sidebar.status("Uploading...") as status:
                try:
                    n_rows = upload_file_to_bigquery(
                        uploaded_file.getvalue(),
                        uploaded_file.name,
                        table_ref,
                        location=settings.bq_location,
                    )
                    status.update(
                        label=f"Uploaded {n_rows:,} rows. Profiling table...",
                        state="running",
                    )
                    SchemaDiscoveryAgent(get_bq()).run(table_ref)
                    status.update(
                        label=f"Done! **{table_name}** is ready to query.",
                        state="complete",
                    )
                    st.session_state.upload_key += 1
                    st.rerun()
                except Exception as exc:
                    status.update(label=f"Failed: {exc}", state="error")


# ── Main area ──────────────────────────────────────────────────────────────────

def render_main() -> None:
    st.title("Ask Your Data")
    st.caption(
        "Type a question in plain English — the platform finds the right table, "
        "writes the SQL, validates it, and explains the answer."
    )

    st.divider()

    # ── Question input ─────────────────────────────────────────────────────────
    col1, col2 = st.columns([5, 1])
    with col1:
        question = st.text_input(
            "question",
            placeholder="e.g. Which country generates the most revenue?",
            label_visibility="collapsed",
        )
    with col2:
        ask_clicked = st.button("Ask", type="primary", use_container_width=True)

    # ── Handle Ask — reset immediately then rerun for clean slate ────────────────
    if ask_clicked and question.strip():
        _reset(question.strip())
        st.session_state.stage = "routing"
        st.rerun()

    # ── Routing stage ──────────────────────────────────────────────────────────
    if st.session_state.stage == "routing":
        with st.spinner("Finding the best table for your question..."):
            try:
                catalog = CatalogManager()
                route = TableRouterAgent(catalog).route(st.session_state.question)
            except Exception as exc:
                st.session_state.error = str(exc)
                st.session_state.stage = "error"
                st.rerun()
                return

        st.session_state.route_info = route

        if route.ambiguous:
            candidates = [(route.table_ref, route.confidence, route.reasoning)]
            for alt in route.alternatives:
                ref, conf = alt[0], alt[1]
                reason = alt[2] if len(alt) > 2 else ""
                if conf >= 0.50:
                    candidates.append((ref, conf, reason))
            st.session_state.candidates = candidates
            st.session_state.stage = "ambiguous"
        else:
            st.session_state.selected_table = route.table_ref
            _run_pipeline_for(st.session_state.question, route.table_ref)

    # ── Ambiguous: table selection ─────────────────────────────────────────────
    if st.session_state.stage == "ambiguous":
        st.warning("I found multiple tables that could answer this. Please choose one:")
        st.write("")

        candidates = st.session_state.candidates
        cols = st.columns(len(candidates))

        for i, (ref, conf, reason) in enumerate(candidates):
            table_id = ref.split(".")[-1]
            with cols[i]:
                st.metric(label=table_id, value=f"{conf:.0%} match")
                st.caption(reason if reason else "No additional reasoning available.")
                if st.button(
                    f"Query {table_id}",
                    key=f"pick_{i}",
                    use_container_width=True,
                ):
                    st.session_state.selected_table = ref
                    _run_pipeline_for(st.session_state.question, ref)
                    st.rerun()

    # ── Result ─────────────────────────────────────────────────────────────────
    if st.session_state.stage == "result" and st.session_state.result:
        result = st.session_state.result
        route = st.session_state.route_info

        # Route badge
        if route:
            table_id = st.session_state.selected_table.split(".")[-1]
            st.caption(
                f"Queried: **{table_id}** — {route.confidence:.0%} confidence  |  "
                f"{route.reasoning}"
            )

        st.write("")

        # NL answer
        if st.session_state.nl_answer:
            st.success(st.session_state.nl_answer)

        # Validation pills
        vcol1, vcol2 = st.columns(2)
        with vcol1:
            if result.syntax_valid:
                st.success("Syntax valid")
            else:
                st.error(f"Syntax error: {result.syntax_error}")
        with vcol2:
            if result.semantic_valid:
                st.success("Semantically correct")
            elif result.semantic_valid is False:
                st.warning(result.semantic_feedback or "Semantic check failed")

        # SQL expander
        with st.expander("View generated SQL", expanded=False):
            st.code(result.sql, language="sql")

        # Results table
        if result.passed and result.rows:
            st.subheader(f"Results — {result.total_rows} row(s)")
            df = _rows_to_df(result.rows)
            st.dataframe(df, use_container_width=True, hide_index=True)
        elif result.passed and result.total_rows == 0:
            st.info("Query returned 0 rows.")
        elif not result.passed:
            st.error("Query did not pass validation.")
            if result.semantic_feedback:
                st.write(result.semantic_feedback)

    # ── Error ──────────────────────────────────────────────────────────────────
    if st.session_state.stage == "error" and st.session_state.error:
        st.error(f"Something went wrong: {st.session_state.error}")


# ── Entry point ────────────────────────────────────────────────────────────────

render_sidebar()
render_main()
