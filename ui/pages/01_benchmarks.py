"""Streamlit benchmark dashboard.

Reads runs from the SQLite results store written by the benchmark runner and
lets the user select 1–N runs to compare on latency, accuracy, and memory.
"""

from __future__ import annotations

from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from app.benchmarking.results_store import get_rows, list_runs
from app.config import get_settings

LATENCY_METRICS = ("p50_latency_ms", "p95_latency_ms", "mean_latency_ms")
ACCURACY_METRICS = ("schema_valid_rate", "field_accuracy_mean", "retry_rate")


def _run_label(row: pd.Series) -> str:
    pieces = [
        str(row.get("backend") or "?"),
        str(row.get("model") or "?"),
        str(row.get("quantization") or "?"),
        str(row.get("timestamp") or "")[:19],
    ]
    return " · ".join(pieces)


def main() -> None:
    st.set_page_config(page_title="Benchmark Dashboard", layout="wide")
    st.title("Benchmark Dashboard")

    settings = get_settings()
    default_db = str(settings.results_db_path)
    with st.sidebar:
        st.header("Results database")
        db_path_str = st.text_input("Path", value=default_db, key="results_db_path")
        db_path = Path(db_path_str)

    runs = list_runs(db_path)
    if not runs:
        st.info(
            f"No benchmark runs found at `{db_path}`. "
            "Run `python scripts/benchmark.py ...` or `POST /benchmarks/run` to populate."
        )
        return

    runs_df = pd.DataFrame(runs)
    runs_df["label"] = runs_df.apply(_run_label, axis=1)

    with st.sidebar:
        backends = sorted(runs_df["backend"].dropna().unique().tolist())
        selected_backends = st.multiselect("Backends", backends, default=backends)
        datasets = sorted(runs_df["dataset"].dropna().unique().tolist())
        selected_datasets = st.multiselect("Datasets", datasets, default=datasets)

    filtered = runs_df[
        runs_df["backend"].isin(selected_backends) & runs_df["dataset"].isin(selected_datasets)
    ].copy()

    if filtered.empty:
        st.warning("No runs match the current filters.")
        return

    st.subheader("All runs")
    st.dataframe(
        filtered[
            [
                "label",
                "n",
                "p50_latency_ms",
                "p95_latency_ms",
                "schema_valid_rate",
                "field_accuracy_mean",
                "retry_rate",
                "run_id",
            ]
        ],
        hide_index=True,
        use_container_width=True,
    )

    st.subheader("Compare runs")
    selected_ids = st.multiselect(
        "Pick runs to compare",
        options=filtered["run_id"].tolist(),
        default=filtered["run_id"].head(min(2, len(filtered))).tolist(),
        format_func=lambda rid: filtered.set_index("run_id").loc[rid, "label"],
    )
    if not selected_ids:
        st.info("Select at least one run to see comparison charts.")
        return

    compare = filtered[filtered["run_id"].isin(selected_ids)].copy()

    col_l, col_a = st.columns(2)
    with col_l:
        st.markdown("**Latency (ms)** — lower is better")
        latency_long = compare.melt(
            id_vars=["label"],
            value_vars=list(LATENCY_METRICS),
            var_name="metric",
            value_name="ms",
        )
        chart = (
            alt.Chart(latency_long)
            .mark_bar()
            .encode(
                x=alt.X("metric:N", title=None),
                y=alt.Y("ms:Q", title="latency (ms)"),
                color=alt.Color("label:N", title="run"),
                column=alt.Column("label:N", title=None, header=alt.Header(labelAngle=0)),
                tooltip=["label", "metric", "ms"],
            )
            .resolve_scale(y="shared")
        )
        st.altair_chart(chart, use_container_width=False)

    with col_a:
        st.markdown("**Accuracy / reliability** — higher is better (except retry_rate)")
        acc_long = compare.melt(
            id_vars=["label"],
            value_vars=list(ACCURACY_METRICS),
            var_name="metric",
            value_name="value",
        )
        chart = (
            alt.Chart(acc_long)
            .mark_bar()
            .encode(
                x=alt.X("label:N", title=None, axis=alt.Axis(labelAngle=-30)),
                y=alt.Y("value:Q", title=None),
                color=alt.Color("label:N", title="run", legend=None),
                column=alt.Column("metric:N", title=None),
                tooltip=["label", "metric", "value"],
            )
        )
        st.altair_chart(chart, use_container_width=False)

    st.markdown("**Throughput & memory**")
    tput_long = compare.melt(
        id_vars=["label"],
        value_vars=["mean_tokens_per_second", "mean_ttft_ms"],
        var_name="metric",
        value_name="value",
    )
    chart = (
        alt.Chart(tput_long)
        .mark_bar()
        .encode(
            x=alt.X("label:N", title=None, axis=alt.Axis(labelAngle=-30)),
            y=alt.Y("value:Q", title=None),
            color=alt.Color("label:N", legend=None),
            column=alt.Column("metric:N", title=None),
            tooltip=["label", "metric", "value"],
        )
    )
    st.altair_chart(chart, use_container_width=False)

    if len(selected_ids) == 1:
        st.subheader("Per-item breakdown")
        rows = get_rows(db_path, selected_ids[0])
        if rows:
            rows_df = pd.DataFrame(rows)
            scatter = (
                alt.Chart(rows_df)
                .mark_circle(size=80, opacity=0.7)
                .encode(
                    x=alt.X("total_latency_ms:Q", title="latency (ms)"),
                    y=alt.Y(
                        "field_accuracy:Q",
                        title="field accuracy",
                        scale=alt.Scale(domain=[0, 1]),
                    ),
                    color=alt.Color("schema_valid:N", title="schema valid"),
                    tooltip=["item_id", "total_latency_ms", "field_accuracy", "retry_count"],
                )
            )
            st.altair_chart(scatter, use_container_width=True)
            st.dataframe(rows_df, hide_index=True, use_container_width=True)
        else:
            st.info("No per-item rows recorded for this run.")


main()
