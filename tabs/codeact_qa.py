"""Consolidated CodeAct QA tab with triage/templates/browse views."""

from __future__ import annotations

import hashlib
from typing import Any

import pandas as pd
import streamlit as st

from tabs.components.codeact_viewer import render_codeact_trace_viewer
from utils.codeact_explorer_features import add_codeact_explorer_columns
from utils.codeact_qaqc import add_codeact_qaqc_columns, build_codeact_template_rollups
from utils.content_kpis import compute_derived_interactions
from utils.trace_parsing import normalize_trace_format


DEFAULT_COMPACT_COLUMNS = [
    "timestamp",
    "trace_id",
    "thread_url",
    "intent_primary",
    "dataset_family",
    "dataset_name",
    "codeact_template_id",
    "flags_count",
    "flags_csv",
    "codeact_consistency_issue",
    "codeact_source_url_count",
    "codeact_chart_types",
]

ADVANCED_COLUMNS = [
    "codeact_chart_prep_mode",
    "codeact_analysis_tags",
    "codeact_endpoint_bases",
    "codeact_endpoint_base_count",
    "codeact_time_check",
    "codeact_dataset_check",
    "codeact_aoi_check",
]


def _trace_fingerprint(traces: list[dict[str, Any]]) -> str:
    keys: list[str] = []
    for trace in traces:
        tid = str((trace or {}).get("id") or "").strip()
        updated = str((trace or {}).get("updatedAt") or (trace or {}).get("timestamp") or "").strip()
        if tid:
            keys.append(f"{tid}:{updated}")
    return hashlib.sha256("|".join(sorted(keys)).encode("utf-8")).hexdigest()


def _normalize_traces(traces: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    normed: list[dict[str, Any]] = []
    traces_by_id: dict[str, dict[str, Any]] = {}
    for trace in traces:
        try:
            n = normalize_trace_format(trace)
        except Exception:
            continue
        normed.append(n)
        trace_id = str(n.get("id") or "").strip()
        if trace_id:
            traces_by_id[trace_id] = n
    return normed, traces_by_id


def _with_thread_link(df: pd.DataFrame, base_thread_url: str) -> pd.DataFrame:
    out = df.copy()
    if "sessionId" in out.columns:
        out["thread_url"] = out["sessionId"].astype(str).apply(
            lambda s: f"{base_thread_url.rstrip('/')}/{s}" if str(s).strip() else ""
        )
    else:
        out["thread_url"] = ""
    return out


def _safe_multiselect(df: pd.DataFrame, column: str, label: str) -> list[str]:
    options = sorted({str(v) for v in df.get(column, pd.Series(dtype=str)).fillna("") if str(v).strip()})
    return st.multiselect(label, options)


def _add_needs_review_columns(df_ca: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    out = df_ca.copy()
    flags_cols = [c for c in out.columns if c.startswith("flag_")]

    def _row_flags(row: pd.Series) -> list[str]:
        active: list[str] = []
        for col in flags_cols:
            if bool(row.get(col, False)):
                active.append(col.replace("flag_", ""))
        return active

    active_flags = out.apply(_row_flags, axis=1) if not out.empty else pd.Series(dtype=object)
    out["flags_count"] = active_flags.apply(len)
    out["flags_csv"] = active_flags.apply(lambda vals: ", ".join(vals))

    consistency_issue = out.get("codeact_consistency_issue", False)
    if not isinstance(consistency_issue, pd.Series):
        consistency_issue = pd.Series([False] * len(out), index=out.index)

    answer_type = out.get("answer_type", "normal")
    if not isinstance(answer_type, pd.Series):
        answer_type = pd.Series(["normal"] * len(out), index=out.index)

    completion_state = out.get("completion_state", "")
    if not isinstance(completion_state, pd.Series):
        completion_state = pd.Series([""] * len(out), index=out.index)

    out["needs_review"] = (
        (out["flags_count"] > 0)
        | consistency_issue.fillna(False).astype(bool)
        | (answer_type.fillna("normal").astype(str) != "normal")
        | (completion_state.fillna("").astype(str) == "error")
    )
    return out, flags_cols


def _render_trace_table(df: pd.DataFrame, show_advanced: bool) -> None:
    cols = DEFAULT_COMPACT_COLUMNS[:]
    if show_advanced:
        cols.extend(ADVANCED_COLUMNS)
    cols = [c for c in cols if c in df.columns]
    st.dataframe(
        df[cols],
        use_container_width=True,
        column_config={"thread_url": st.column_config.LinkColumn("Thread URL")},
    )


def _trace_viewer_panel(df: pd.DataFrame, traces_by_id: dict[str, dict], base_thread_url: str, *, key_prefix: str) -> None:
    trace_options = df.get("trace_id", pd.Series(dtype=str)).astype(str).tolist() if not df.empty else []
    selected_trace_id = st.selectbox("Select trace_id", options=[""] + trace_options, key=f"trace_sel_{key_prefix}")
    if not selected_trace_id:
        st.info("Select a trace to open Trace Viewer.")
        return
    trace = traces_by_id.get(selected_trace_id, {})
    row = df[df.get("trace_id", "").astype(str) == selected_trace_id].head(1)
    row_dict = row.iloc[0].to_dict() if not row.empty else {}
    render_codeact_trace_viewer(trace=trace, row=row_dict, base_thread_url=base_thread_url)


def _render_triage_view(df_ca: pd.DataFrame, traces_by_id: dict[str, dict], base_thread_url: str) -> None:
    st.caption("Default triage queue for CodeAct traces needing QA review.")
    df_triage, flags_cols = _add_needs_review_columns(df_ca)
    df_triage = df_triage[df_triage["needs_review"]].copy()

    left, right = st.columns([3, 2])
    with left:
        dataset_families = _safe_multiselect(df_triage, "dataset_family", "dataset_family")
        retrieval_modes = _safe_multiselect(df_triage, "codeact_retrieval_mode", "retrieval_mode")
        selected_flags = st.multiselect("flags", flags_cols)
        only_consistency = st.checkbox("Only consistency issues", value=False)

        with st.expander("Advanced filters", expanded=False):
            dataset_names = _safe_multiselect(df_triage, "dataset_name", "dataset_name")

        filtered = df_triage.copy()
        if dataset_families:
            filtered = filtered[filtered["dataset_family"].astype(str).isin(dataset_families)]
        if retrieval_modes:
            filtered = filtered[filtered["codeact_retrieval_mode"].astype(str).isin(retrieval_modes)]
        if selected_flags:
            mask = pd.Series(False, index=filtered.index)
            for flag in selected_flags:
                mask = mask | filtered.get(flag, False).fillna(False).astype(bool)
            filtered = filtered[mask]
        if only_consistency and "codeact_consistency_issue" in filtered.columns:
            filtered = filtered[filtered["codeact_consistency_issue"].fillna(False).astype(bool)]
        if dataset_names:
            filtered = filtered[filtered["dataset_name"].astype(str).isin(dataset_names)]

        show_advanced = st.toggle("Show advanced columns", value=False)
        _render_trace_table(filtered, show_advanced)

    with right:
        _trace_viewer_panel(filtered, traces_by_id, base_thread_url, key_prefix="triage")


def _render_templates_view(df_ca: pd.DataFrame, traces_by_id: dict[str, dict], base_thread_url: str) -> None:
    df_rollup, template_traces = build_codeact_template_rollups(df_ca)
    if df_rollup.empty:
        st.info("No template clusters found.")
        return

    df_rollup = df_rollup.copy()
    df_rollup["issue_rate"] = 1.0 - (
        (1.0 - df_rollup.get("time_issue_rate", 0.0))
        * (1.0 - df_rollup.get("dataset_issue_rate", 0.0))
        * (1.0 - df_rollup.get("aoi_issue_rate", 0.0))
    )
    consistency_issue_by_template = (
        template_traces.groupby("codeact_template_id", dropna=False)["codeact_consistency_issue"].mean().rename("consistency_issue_rate")
    )
    df_rollup = df_rollup.merge(consistency_issue_by_template, on="codeact_template_id", how="left")

    top_n = int(st.slider("Top N templates", min_value=5, max_value=200, value=25, step=5))
    summary_cols = [
        "codeact_template_id",
        "n_traces",
        "issue_rate",
        "consistency_issue_rate",
        "intents_top3",
        "representative_trace_id",
    ]
    if "codeact_raw_dataset_names" in df_ca.columns:
        top_datasets = (
            df_ca.groupby("codeact_template_id")["dataset_name"]
            .apply(lambda s: ", ".join(pd.Series(s).dropna().astype(str).value_counts().head(3).index.tolist()))
            .rename("top_datasets")
        )
        df_rollup = df_rollup.merge(top_datasets, on="codeact_template_id", how="left")
        summary_cols.insert(4, "top_datasets")

    summary = df_rollup.sort_values("n_traces", ascending=False).head(top_n)
    shown_cols = [c for c in summary_cols if c in summary.columns]
    st.dataframe(summary[shown_cols], use_container_width=True)

    selected_template = st.selectbox(
        "Select template_id",
        options=[""] + summary["codeact_template_id"].astype(str).tolist(),
        key="template_select",
    )
    if not selected_template:
        return

    traces = df_ca[df_ca.get("codeact_template_id", "").astype(str) == selected_template].copy()
    traces, _ = _add_needs_review_columns(traces)
    show_advanced = st.toggle("Show advanced columns", value=False, key="template_adv_cols")
    _render_trace_table(traces, show_advanced)
    _trace_viewer_panel(traces, traces_by_id, base_thread_url, key_prefix="template")


def _render_browse_view(df_ca: pd.DataFrame, traces_by_id: dict[str, dict], base_thread_url: str) -> None:
    df_browse, flags_cols = _add_needs_review_columns(df_ca)
    filtered = df_browse.copy()

    with st.expander("Advanced filters", expanded=True):
        dataset_families = _safe_multiselect(filtered, "dataset_family", "dataset_family")
        dataset_names = _safe_multiselect(filtered, "dataset_name", "dataset_name")
        intents = _safe_multiselect(filtered, "intent_primary", "intent_primary")
        prep_modes = _safe_multiselect(filtered, "codeact_chart_prep_mode", "chart_prep_mode")
        retrieval_modes = _safe_multiselect(filtered, "codeact_retrieval_mode", "retrieval_mode")
        chart_types = _safe_multiselect(filtered, "codeact_chart_types", "chart_types")
        analysis_tags = _safe_multiselect(filtered, "codeact_analysis_tags", "analysis_tags")
        template_ids = _safe_multiselect(filtered, "codeact_template_id", "template_id")
        selected_flags = st.multiselect("flags", flags_cols, key="browse_flags")
        only_needs_review = st.checkbox("Only needs review", value=False)

    if dataset_families:
        filtered = filtered[filtered["dataset_family"].astype(str).isin(dataset_families)]
    if dataset_names:
        filtered = filtered[filtered["dataset_name"].astype(str).isin(dataset_names)]
    if intents:
        filtered = filtered[filtered["intent_primary"].astype(str).isin(intents)]
    if prep_modes:
        filtered = filtered[filtered["codeact_chart_prep_mode"].astype(str).isin(prep_modes)]
    if retrieval_modes:
        filtered = filtered[filtered["codeact_retrieval_mode"].astype(str).isin(retrieval_modes)]
    if chart_types:
        filtered = filtered[filtered["codeact_chart_types"].astype(str).isin(chart_types)]
    if analysis_tags:
        filtered = filtered[filtered["codeact_analysis_tags"].astype(str).isin(analysis_tags)]
    if template_ids:
        filtered = filtered[filtered["codeact_template_id"].astype(str).isin(template_ids)]
    if selected_flags:
        mask = pd.Series(False, index=filtered.index)
        for flag in selected_flags:
            mask = mask | filtered.get(flag, False).fillna(False).astype(bool)
        filtered = filtered[mask]
    if only_needs_review:
        filtered = filtered[filtered["needs_review"]]

    left, right = st.columns([3, 2])
    with left:
        show_advanced = st.toggle("Show advanced columns", value=False, key="browse_adv_cols")
        _render_trace_table(filtered, show_advanced)
    with right:
        _trace_viewer_panel(filtered, traces_by_id, base_thread_url, key_prefix="browse")


def render(base_thread_url: str) -> None:
    st.title("🧩 CodeAct QA")

    traces = st.session_state.get("stats_traces", [])
    if not traces:
        st.info("Fetch traces in sidebar first")
        return

    normed, traces_by_id = _normalize_traces(traces)
    if not normed:
        st.warning("No valid traces were available after normalization.")
        return

    fp = _trace_fingerprint(normed)
    derived_cache = st.session_state.get("codeact_df")
    derived_fp = str(st.session_state.get("codeact_df_fingerprint") or "")

    if isinstance(derived_cache, pd.DataFrame) and not derived_cache.empty and derived_fp == fp:
        enriched = derived_cache.copy()
    else:
        derived = st.session_state.get("content_kpis_df")
        base_fp = str(st.session_state.get("content_kpis_fingerprint") or "")
        if not isinstance(derived, pd.DataFrame) or derived.empty or base_fp != fp:
            derived = compute_derived_interactions(normed)
            st.session_state["content_kpis_df"] = derived
            st.session_state["content_kpis_fingerprint"] = fp

        enriched = add_codeact_qaqc_columns(derived, traces_by_id)
        enriched = add_codeact_explorer_columns(enriched, traces_by_id)
        st.session_state["codeact_df"] = enriched
        st.session_state["codeact_df_fingerprint"] = fp

    df_ca = enriched[enriched.get("codeact_present", False).fillna(False).astype(bool)].copy()
    df_ca = _with_thread_link(df_ca, base_thread_url)

    if df_ca.empty:
        st.info("No CodeAct traces found in current dataset.")
        return

    view = st.radio("View", ["Triage", "Templates", "Browse"], horizontal=True, index=0)

    if view == "Triage":
        _render_triage_view(df_ca, traces_by_id, base_thread_url)
    elif view == "Templates":
        _render_templates_view(df_ca, traces_by_id, base_thread_url)
    else:
        _render_browse_view(df_ca, traces_by_id, base_thread_url)
