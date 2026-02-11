"""Content KPI tab renderer."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st

from utils.content_kpis import build_content_slices, compute_derived_interactions, summarize_content
from utils.trace_parsing import normalize_trace_format


def _pct_str(v: Any) -> str:
    try:
        return f"{float(v) * 100:.1f}%"
    except Exception:
        return "0.0%"


def _truncate(v: Any, n: int = 200) -> str:
    s = str(v or "")
    return s if len(s) <= n else s[: n - 1] + "…"


def render(
    public_key: str,
    secret_key: str,
    base_url: str,
    base_thread_url: str,
    gemini_api_key: str,
    use_date_filter: bool,
    start_date,
    end_date,
    envs: list[str],
    stats_page_limit: int,
    stats_max_traces: int,
) -> None:
    del public_key, secret_key, base_url, gemini_api_key, use_date_filter, start_date, end_date, envs, stats_page_limit, stats_max_traces

    st.title("🧱 Content KPIs (Deterministic)")
    st.caption("Deterministic structural quality KPIs; no LLM.")

    traces = st.session_state.get("stats_traces") or []
    if not traces:
        st.info("No traces loaded. Load traces from the sidebar/date filters first, then return to this page.")
        return

    normed = [normalize_trace_format(t) for t in traces]
    trace_ids = sorted(str(t.get("id") or "") for t in normed)
    fingerprint = hashlib.sha256("|".join(trace_ids).encode("utf-8")).hexdigest()

    if fingerprint != st.session_state.get("content_kpis_fingerprint"):
        derived = compute_derived_interactions(normed)
        summary = summarize_content(derived)
        slices = build_content_slices(derived)
        st.session_state["content_kpis_fingerprint"] = fingerprint
        st.session_state["content_kpis_derived"] = derived
        st.session_state["content_kpis_summary"] = summary
        st.session_state["content_kpis_slices"] = slices

    derived: pd.DataFrame = st.session_state.get("content_kpis_derived", pd.DataFrame())
    summary: dict[str, Any] = st.session_state.get("content_kpis_summary", {})
    slices: pd.DataFrame = st.session_state.get("content_kpis_slices", pd.DataFrame())

    if derived.empty:
        st.warning("No prompt-bearing traces found for content KPI analysis.")
        return

    kpis = summary.get("kpis", {})
    cols = st.columns(6)
    metric_keys = [
        ("Complete (scored)", "complete_answer_rate_scored_intents"),
        ("Needs input (scored)", "needs_user_input_rate_scored_intents"),
        ("Errors (scored)", "error_rate_scored_intents"),
        ("Dataset identifiable (scored)", "global_dataset_identifiable_rate_scored_intents"),
        ("Citation rate", "global_citation_rate"),
        ("Threads ending in needs-input", "threads_ended_after_needs_user_input_rate"),
    ]
    for col, (label, key) in zip(cols, metric_keys):
        col.metric(label, _pct_str(kpis.get(key, 0.0)))

    gq = summary.get("global_quality", {})
    c1, c2, c3 = st.columns(3)

    completion_df = pd.DataFrame(
        [{"completion_state": k, "count": v} for k, v in (gq.get("completion_state_counts") or {}).items()]
    )
    if not completion_df.empty:
        c1.altair_chart(
            alt.Chart(completion_df).mark_bar().encode(x="completion_state:N", y="count:Q"),
            use_container_width=True,
        )

    nui_df = pd.DataFrame(
        [{"reason": k, "count": v} for k, v in (gq.get("needs_user_input_reason_counts") or {}).items()]
    )
    if not nui_df.empty:
        c2.altair_chart(
            alt.Chart(nui_df).mark_bar().encode(x="reason:N", y="count:Q"),
            use_container_width=True,
        )

    fail_counts = (summary.get("struct_outcome_summary", {}) or {}).get("failure_reasons", {}) or {}
    fail_df = pd.DataFrame([{"reason": k, "count": v} for k, v in fail_counts.items()]).sort_values("count", ascending=False).head(10)
    if not fail_df.empty:
        c3.altair_chart(
            alt.Chart(fail_df).mark_bar().encode(x="reason:N", y="count:Q"),
            use_container_width=True,
        )

    st.subheader("Intent summary")
    intent_summary = pd.DataFrame.from_dict(summary.get("intent_summary", {}), orient="index").reset_index().rename(columns={"index": "intent_primary"})
    st.dataframe(intent_summary, use_container_width=True)

    st.subheader("Dataset family summary")
    dataset_summary = pd.DataFrame.from_dict(summary.get("dataset_family_summary", {}), orient="index").reset_index().rename(columns={"index": "dataset_family"})
    if not dataset_summary.empty and "count_data_intents" in dataset_summary.columns:
        dataset_summary = dataset_summary.sort_values("count_data_intents", ascending=False)
    st.dataframe(dataset_summary, use_container_width=True)

    st.subheader("Content slices")
    st.dataframe(slices, use_container_width=True)

    st.subheader("Drilldown explorer")
    f1, f2, f3, f4 = st.columns(4)
    iopts = ["All"] + sorted(x for x in derived["intent_primary"].dropna().astype(str).unique())
    copts = ["All"] + sorted(x for x in derived["completion_state"].dropna().astype(str).unique())
    ropts = ["All"] + sorted(x for x in derived["needs_user_input_reason"].dropna().astype(str).unique() if x)
    dopts = ["All"] + sorted(x for x in derived["dataset_family"].dropna().astype(str).unique())
    intent_filter = f1.selectbox("intent_primary", iopts)
    completion_filter = f2.selectbox("completion_state", copts)
    reason_filter = f3.selectbox("needs_user_input_reason", ropts)
    family_filter = f4.selectbox("dataset_family", dopts)

    view = derived.copy()
    if intent_filter != "All":
        view = view[view["intent_primary"] == intent_filter]
    if completion_filter != "All":
        view = view[view["completion_state"] == completion_filter]
    if reason_filter != "All":
        view = view[view["needs_user_input_reason"] == reason_filter]
    if family_filter != "All":
        view = view[view["dataset_family"] == family_filter]

    view = view.copy()
    view["prompt"] = view["prompt"].map(_truncate)
    view["response"] = view["response"].map(_truncate)
    view["link"] = view["sessionId"].fillna("").map(lambda s: f"{base_thread_url}/{s}" if s else "")

    cols_to_show = [
        "timestamp", "trace_id", "sessionId", "thread_id", "userId",
        "intent_primary", "completion_state", "needs_user_input_reason", "struct_fail_reason",
        "dataset_name", "dataset_family", "aoi_name", "time_start", "time_end",
        "prompt", "response", "link",
    ]
    cols_to_show = [c for c in cols_to_show if c in view.columns]
    st.dataframe(
        view[cols_to_show],
        use_container_width=True,
        column_config={"link": st.column_config.LinkColumn("Thread link")},
    )

    st.subheader("Downloads")
    d1, d2 = st.columns(2)
    d1.download_button(
        "Download derived_interactions.csv",
        data=derived.to_csv(index=False).encode("utf-8"),
        file_name="derived_interactions.csv",
        mime="text/csv",
    )
    d2.download_button(
        "Download content_summary.json",
        data=json.dumps(summary, indent=2, default=str).encode("utf-8"),
        file_name="content_summary.json",
        mime="application/json",
    )
