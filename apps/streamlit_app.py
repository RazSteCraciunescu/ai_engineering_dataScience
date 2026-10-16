"""Optional Streamlit dashboard. Run with: streamlit run apps/streamlit_app.py"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="Feedback Intelligence", layout="wide")
    frame = pd.read_csv(ROOT / "data" / "processed" / "feedback.csv")
    metrics = json.loads((ROOT / "artifacts" / "reports" / "model-metrics.json").read_text())

    st.title("Public Feedback Intelligence")
    st.caption("Offline-first data collection, analytics, machine learning and retrieval")
    first, second, third, fourth = st.columns(4)
    first.metric("Records", f"{len(frame):,}")
    second.metric("Median resolution", f"{frame['resolution_hours'].median():.1f} h")
    third.metric("Macro-F1", f"{metrics['priority_classifier']['macro_f1']:.3f}")
    fourth.metric("Outliers", int(frame["is_outlier"].sum()))

    category = st.multiselect("Category", sorted(frame["category"].unique()))
    priority = st.multiselect("Priority", ["high", "medium", "low"])
    filtered = frame
    if category:
        filtered = filtered[filtered["category"].isin(category)]
    if priority:
        filtered = filtered[filtered["priority"].isin(priority)]

    chart, table = st.columns([1, 1.3])
    with chart:
        st.subheader("Volume by category")
        st.bar_chart(filtered["category"].value_counts())
    with table:
        st.subheader("Records")
        st.dataframe(
            filtered[["record_id", "category", "priority", "channel", "title"]],
            use_container_width=True,
            hide_index=True,
        )


if __name__ == "__main__":
    main()
