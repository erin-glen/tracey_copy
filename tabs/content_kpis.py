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
from utils.docs_ui import render_page_help, metric_with_help


def _pct_str(v: Any) -> str:
    if v is None:
        return "—"
    try:
        return f"{float(v) * 100:.1f}%"
    except Exception:
        return "—"


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

    render_page_help("content_kpis", expanded=False)

    traces = st.session_state.get("stats_traces") or []
    if not traces:
        st.info("No traces loaded. Load traces from the sidebar/date filters first, then return to this page.")
        return

    normed = [normalize_trace_format(t) for t in traces]
    trace_ids = sorted(str(t.get("id") or "") for t in normed)
    fingerprint = hashlib.sha256("|".join(trace_ids).encode("utf-8")).hexdigest()

    cached_fingerprint = st.session_state.get("content_kpis_fingerprint")
    derived = st.session_state.get("content_kpis_df")
    if not isinstance(derived, pd.DataFrame) or fingerprint != cached_fingerprint:
        derived = compute_derived_interactions(normed)
        st.session_state["content_kpis_df"] = derived
        st.session_state["content_kpis_derived"] = derived
        st.session_state["content_kpis_fingerprint"] = fingerprint

    derived = st.session_state.get("content_kpis_df")
    if not isinstance(derived, pd.DataFrame):
        derived = st.session_state.get("content_kpis_derived", pd.DataFrame())
        st.session_state["content_kpis_df"] = derived

    if fingerprint != cached_fingerprint or "content_kpis_summary" not in st.session_state or "content_kpis_slices" not in st.session_state:
        summary = summarize_content(derived)
        slices = build_content_slices(derived)
        st.session_state["content_kpis_summary"] = summary
        st.session_state["content_kpis_slices"] = slices

    derived = st.session_state.get("content_kpis_df", pd.DataFrame())
    summary: dict[str, Any] = st.session_state.get("content_kpis_summary", {})
    slices: pd.DataFrame = st.session_state.get("content_kpis_slices", pd.DataFrame())

    if derived.empty:
        st.warning("No prompt-bearing traces found for content KPI analysis.")
        return

    window_utc = summary.get("window_utc") or {}
    context_parts = [f"Loaded traces: {len(traces):,}"]
    if window_utc.get("start") and window_utc.get("end"):
        context_parts.append(f"Window (UTC): {window_utc['start']} → {window_utc['end']}")
    if summary.get("unique_users") is not None:
        context_parts.append(f"Unique users: {summary.get('unique_users')}")
    st.caption(" | ".join(context_parts))

    kpis = summary.get("kpis", {})
    cols = st.columns(4)
    primary_metric_keys = [
        ("Complete (scored)", "complete_answer_rate_scored_intents"),
        ("Needs input (scored)", "needs_user_input_rate_scored_intents"),
        ("Errors (scored)", "error_rate_scored_intents"),
        ("Citations shown (scored)", "citations_shown_rate_scored_intents"),
    ]
    for col, (label, key) in zip(cols, primary_metric_keys):
        with col:
            metric_with_help(
                label,
                _pct_str(kpis.get(key, 0.0)),
                metric_id=key,
                key=f"content_kpis_{key}",
            )

    with st.expander("Additional KPIs", expanded=False):
        a1, a2, a3 = st.columns(3)
        for col, (label, key) in zip(
            [a1, a2, a3],
            [
                ("Dataset identifiable (scored)", "global_dataset_identifiable_rate_scored_intents"),
                ("Citation metadata present (scored)", "citation_metadata_present_rate_scored_intents"),
                ("Threads ending in needs-input", "threads_ended_after_needs_user_input_rate"),
            ],
        ):
            with col:
                metric_with_help(
                    label,
                    _pct_str(kpis.get(key, 0.0)),
                    metric_id=key,
                    key=f"content_kpis_{key}",
                )

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

    intent_summary = pd.DataFrame.from_dict(summary.get("intent_summary", {}), orient="index").reset_index().rename(columns={"index": "intent_primary"})
    dataset_summary = pd.DataFrame.from_dict(summary.get("dataset_family_summary", {}), orient="index").reset_index().rename(columns={"index": "dataset_family"})
    if not dataset_summary.empty and "count_data_intents" in dataset_summary.columns:
        dataset_summary = dataset_summary.sort_values("count_data_intents", ascending=False)

    with st.expander("Details (tables)", expanded=False):
        show_full = st.checkbox("Show full table", value=False)

        st.subheader("Intent summary")
        intent_display = intent_summary if show_full else intent_summary.head(15)
        st.dataframe(intent_display, use_container_width=True)

        st.subheader("Dataset family summary")
        dataset_display = dataset_summary if show_full else dataset_summary.head(15)
        st.dataframe(dataset_display, use_container_width=True)

        st.subheader("Content slices")
        st.dataframe(slices, use_container_width=True)

    st.subheader("Drilldown explorer")
    preset = st.selectbox(
        "Preset",
        [
            "All rows",
            "Needs user input",
            "Errors",
            "Trend failures (incomplete)",
            "Trend missing citations",
            "Lookup missing dataset",
            "CodeAct present",
        ],
    )
    search = st.text_input("Search (dataset / AOI / prompt / response / sessionId / trace_id)", "")

    with st.expander("Advanced filters", expanded=False):
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
    if preset == "Needs user input" and "completion_state" in view.columns:
        view = view[view["completion_state"] == "needs_user_input"]
    elif preset == "Errors":
        err_mask = pd.Series(False, index=view.index)
        if "completion_state" in view.columns:
            err_mask = err_mask | (view["completion_state"] == "error")
        if "answer_type" in view.columns:
            err_mask = err_mask | view["answer_type"].isin({"model_error", "missing_output", "empty_or_short"})
        view = view[err_mask]
    elif preset == "Trend failures (incomplete)" and {"intent_primary", "completion_state"}.issubset(view.columns):
        view = view[
            (view["intent_primary"] == "trend_over_time")
            & (~view["completion_state"].isin({"complete_answer", "needs_user_input"}))
        ]
    elif preset == "Trend missing citations" and "struct_fail_reason" in view.columns:
        view = view[view["struct_fail_reason"].fillna("").str.contains("no_citation", case=False)]
    elif preset == "Lookup missing dataset" and {"intent_primary", "struct_fail_reason"}.issubset(view.columns):
        view = view[
            (view["intent_primary"] == "data_lookup")
            & (view["struct_fail_reason"].fillna("").str.contains("missing_dataset", case=False))
        ]
    elif preset == "CodeAct present" and "codeact_present" in view.columns:
        view = view[view["codeact_present"] == True]

    if search:
        search_cols = [
            "prompt",
            "response",
            "dataset_name",
            "dataset_family",
            "aoi_name",
            "sessionId",
            "trace_id",
        ]
        search_cols = [c for c in search_cols if c in view.columns]
        if search_cols:
            search_mask = pd.Series(False, index=view.index)
            for col in search_cols:
                search_mask = search_mask | view[col].fillna("").astype(str).str.contains(search, case=False, regex=False)
            view = view[search_mask]

    if intent_filter != "All":
        view = view[view["intent_primary"] == intent_filter]
    if completion_filter != "All":
        view = view[view["completion_state"] == completion_filter]
    if reason_filter != "All":
        view = view[view["needs_user_input_reason"] == reason_filter]
    if family_filter != "All":
        view = view[view["dataset_family"] == family_filter]

    view = view.copy()
    if "prompt" in view.columns:
        view["prompt"] = view["prompt"].map(_truncate)
    if "response" in view.columns:
        view["response"] = view["response"].map(_truncate)
    thread_base = base_thread_url.rstrip("/")
    if "sessionId" in view.columns:
        view["link"] = view["sessionId"].fillna("").map(lambda s: f"{thread_base}/{s}" if s else "")

    st.caption(f"{len(view):,} rows match filters")

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
    st.download_button(
        "Download filtered rows (CSV)",
        data=view[cols_to_show].to_csv(index=False).encode("utf-8"),
        file_name="content_kpis_filtered.csv",
        mime="text/csv",
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
