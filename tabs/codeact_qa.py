"""Simplified CodeAct QA tab: Inbox -> Trace Viewer."""

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


FLAG_SHORT_LABELS = {
    "flag_hardcoded_chart_data": "hardcoded_chart",
    "flag_requery_in_later_block": "requery",
    "flag_multi_source_provenance": "multi_source",
    "flag_pie_percent_sum_off": "pie_sum_off",
    "flag_missing_final_insight": "missing_insight",
    "flag_endpoint_base_differs_from_dataset_endpoint": "endpoint_diff",
}

PRESETS = [
    "Needs review (recommended)",
    "All CodeAct",
    "Hardcoded chart data",
    "Requery/reload in later block",
    "Multi-source provenance",
    "Pie % sum off",
    "Consistency issues (time/dataset/AOI)",
    "Errors / empty outputs",
]

COMPACT_INBOX_COLUMNS = [
    "timestamp",
    "dataset_family",
    "dataset_name",
    "aoi_name",
    "time_start",
    "time_end",
    "codeact_chart_types",
    "codeact_source_url_count",
    "codeact_code_blocks_count",
    "review_reason",
    "trace_id",
    "sessionId",
    "thread_url",
]

ADVANCED_INBOX_COLUMNS = [
    "codeact_retrieval_mode",
    "codeact_chart_prep_mode",
    "codeact_analysis_tags",
    "codeact_time_check",
    "codeact_dataset_check",
    "codeact_aoi_check",
    "codeact_template_id",
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


def _series_or_default(df: pd.DataFrame, col: str, default: Any) -> pd.Series:
    if col in df.columns:
        return df[col]
    return pd.Series([default] * len(df), index=df.index)


def _is_truthy(series: pd.Series) -> pd.Series:
    normalized = series.fillna(False).astype(str).str.strip().str.lower()
    return normalized.isin(["true", "1", "yes", "y"])


def _pipe_dedupe(values: list[str]) -> str:
    seen: list[str] = []
    for value in values:
        token = str(value).strip()
        if token and token not in seen:
            seen.append(token)
    return "|".join(seen)


def _add_inbox_columns(df_ca: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    out = df_ca.copy()
    flags_cols = [c for c in out.columns if c.startswith("flag_")]

    if flags_cols:
        flags_frame = out[flags_cols].apply(_is_truthy)
        out["flags_count"] = flags_frame.sum(axis=1)
    else:
        flags_frame = pd.DataFrame(index=out.index)
        out["flags_count"] = 0

    def _row_flags(idx: Any) -> str:
        labels: list[str] = []
        if not flags_cols:
            return ""
        for col in flags_cols:
            if bool(flags_frame.at[idx, col]):
                labels.append(FLAG_SHORT_LABELS.get(col, col))
        return _pipe_dedupe(labels)

    out["flags_csv"] = [_row_flags(idx) for idx in out.index]

    consistency_series = _is_truthy(_series_or_default(out, "codeact_consistency_issue", False))
    answer_type_series = _series_or_default(out, "answer_type", "normal").fillna("normal").astype(str)
    completion_series = _series_or_default(out, "completion_state", "").fillna("").astype(str)

    review_reasons: list[str] = []
    for idx in out.index:
        tokens: list[str] = []
        flags_token = str(out.at[idx, "flags_csv"]).strip()
        if flags_token:
            tokens.extend(flags_token.split("|"))
        if bool(consistency_series.at[idx]):
            tokens.append("consistency_issue")
        if answer_type_series.at[idx] != "normal" or completion_series.at[idx] == "error":
            tokens.append("error")
        if completion_series.at[idx] == "no_data":
            tokens.append("no_data")
        review_reasons.append(_pipe_dedupe(tokens))

    out["review_reason"] = review_reasons
    out["needs_review"] = (
        (out["flags_count"] > 0)
        | consistency_series
        | (answer_type_series != "normal")
        | (completion_series == "error")
        | (completion_series == "no_data")
    )
    return out, flags_cols


def _apply_search(df: pd.DataFrame, search: str) -> pd.DataFrame:
    query = (search or "").strip().lower()
    if not query:
        return df
    columns = [c for c in ["aoi_name", "dataset_name", "prompt"] if c in df.columns]
    if not columns:
        return df
    text_blob = pd.Series([""] * len(df), index=df.index, dtype=str)
    for col in columns:
        text_blob = text_blob + " " + df[col].fillna("").astype(str).str.lower()
    return df[text_blob.str.contains(query, na=False)]


def _apply_preset(df_ca: pd.DataFrame, preset: str) -> pd.DataFrame:
    if preset == "Needs review (recommended)":
        return df_ca[_is_truthy(_series_or_default(df_ca, "needs_review", False))]
    if preset == "All CodeAct":
        return df_ca
    if preset == "Hardcoded chart data":
        return df_ca[_is_truthy(_series_or_default(df_ca, "flag_hardcoded_chart_data", False))]
    if preset == "Requery/reload in later block":
        return df_ca[_is_truthy(_series_or_default(df_ca, "flag_requery_in_later_block", False))]
    if preset == "Multi-source provenance":
        return df_ca[_is_truthy(_series_or_default(df_ca, "flag_multi_source_provenance", False))]
    if preset == "Pie % sum off":
        return df_ca[_is_truthy(_series_or_default(df_ca, "flag_pie_percent_sum_off", False))]
    if preset == "Consistency issues (time/dataset/AOI)":
        consistency_issue = _is_truthy(_series_or_default(df_ca, "codeact_consistency_issue", False))
        time_issue = _series_or_default(df_ca, "codeact_time_check", "").astype(str).isin(["missing", "mismatch"])
        dataset_issue = _series_or_default(df_ca, "codeact_dataset_check", "").astype(str) != "ok"
        aoi_issue = _series_or_default(df_ca, "codeact_aoi_check", "").astype(str) != "ok"
        return df_ca[consistency_issue | time_issue | dataset_issue | aoi_issue]
    if preset == "Errors / empty outputs":
        answer_type = _series_or_default(df_ca, "answer_type", "normal").astype(str)
        completion_state = _series_or_default(df_ca, "completion_state", "").astype(str)
        return df_ca[
            answer_type.isin(["model_error", "missing_output", "empty_or_short"]) | (completion_state == "error")
        ]
    return df_ca


def _add_thread_url(df: pd.DataFrame, base_thread_url: str) -> pd.DataFrame:
    out = df.copy()
    has_session = "sessionId" in out.columns
    has_base_url = bool(str(base_thread_url).strip())
    if has_session and has_base_url:
        out["thread_url"] = out["sessionId"].fillna("").astype(str).apply(
            lambda s: f"{base_thread_url.rstrip('/')}/{s}" if s.strip() else ""
        )
    else:
        out["thread_url"] = ""
    return out


def _render_inbox_table(df: pd.DataFrame, show_advanced: bool) -> None:
    columns = [c for c in COMPACT_INBOX_COLUMNS if c in df.columns]
    if show_advanced:
        columns.extend([c for c in ADVANCED_INBOX_COLUMNS if c in df.columns])
    st.dataframe(
        df[columns],
        use_container_width=True,
        column_config={"thread_url": st.column_config.LinkColumn("Thread")},
    )


def render(base_thread_url: str) -> None:
    st.title("🧩 CodeAct QA")
    st.caption("Use presets to triage CodeAct traces, then inspect the decoded timeline, charts, and provenance.")

    traces = st.session_state.get("stats_traces", [])
    if not traces:
        st.info("Fetch traces in sidebar first")
        return

    normed, traces_by_id = _normalize_traces(traces)
    if not normed:
        st.warning("No valid traces were available after normalization.")
        return

    fp = _trace_fingerprint(normed)
    cached_df = st.session_state.get("codeact_df")
    cached_fp = str(st.session_state.get("codeact_df_fingerprint") or "")

    if isinstance(cached_df, pd.DataFrame) and not cached_df.empty and cached_fp == fp:
        enriched = cached_df.copy()
    else:
        derived = st.session_state.get("content_kpis_df")
        derived_fp = str(st.session_state.get("content_kpis_fingerprint") or "")
        if not isinstance(derived, pd.DataFrame) or derived.empty or derived_fp != fp:
            derived = compute_derived_interactions(normed)
            st.session_state["content_kpis_df"] = derived
            st.session_state["content_kpis_fingerprint"] = fp

        enriched = add_codeact_qaqc_columns(derived, traces_by_id)
        enriched = add_codeact_explorer_columns(enriched, traces_by_id)
        st.session_state["codeact_df"] = enriched
        st.session_state["codeact_df_fingerprint"] = fp

    codeact_present = _is_truthy(_series_or_default(enriched, "codeact_present", False))
    df_ca = enriched[codeact_present].copy()

    if df_ca.empty:
        st.warning("No CodeAct traces found in this fetch window.")
        return

    df_ca, _ = _add_inbox_columns(df_ca)
    df_ca = _add_thread_url(df_ca, base_thread_url)

    preset = st.selectbox("Preset", PRESETS, index=0)
    filtered = _apply_preset(df_ca, preset)

    if "dataset_family" in filtered.columns:
        family_options = sorted({str(v) for v in filtered["dataset_family"].fillna("") if str(v).strip()})
        selected_families = st.multiselect("dataset_family", family_options)
        if selected_families:
            filtered = filtered[filtered["dataset_family"].astype(str).isin(selected_families)]

    search = st.text_input("Search (AOI / dataset / prompt)", "")
    filtered = _apply_search(filtered, search)

    with st.expander("Advanced filters"):
        if "codeact_retrieval_mode" in filtered.columns:
            opts = sorted({str(v) for v in filtered["codeact_retrieval_mode"].fillna("") if str(v).strip()})
            selected = st.multiselect("retrieval_mode", opts)
            if selected:
                filtered = filtered[filtered["codeact_retrieval_mode"].astype(str).isin(selected)]

        if "codeact_chart_types" in filtered.columns:
            chart_type_options: set[str] = set()
            for val in filtered["codeact_chart_types"].fillna("").astype(str):
                for token in [x.strip() for x in val.split(",") if x.strip()]:
                    chart_type_options.add(token)
            selected_chart_types = st.multiselect("chart_types", sorted(chart_type_options))
            if selected_chart_types:
                filtered = filtered[
                    filtered["codeact_chart_types"].fillna("").astype(str).apply(
                        lambda s: any(token in [x.strip() for x in s.split(",") if x.strip()] for token in selected_chart_types)
                    )
                ]

        if "codeact_chart_prep_mode" in filtered.columns:
            opts = sorted({str(v) for v in filtered["codeact_chart_prep_mode"].fillna("") if str(v).strip()})
            selected = st.multiselect("chart_prep_mode", opts)
            if selected:
                filtered = filtered[filtered["codeact_chart_prep_mode"].astype(str).isin(selected)]

        if "codeact_template_id" in filtered.columns:
            opts = sorted({str(v) for v in filtered["codeact_template_id"].fillna("") if str(v).strip()})
            selected = st.multiselect("template_id", opts)
            if selected:
                filtered = filtered[filtered["codeact_template_id"].astype(str).isin(selected)]

    st.caption(f"Inbox rows: {len(filtered)}")
    show_advanced_columns = st.checkbox("Show advanced columns", value=False)
    _render_inbox_table(filtered, show_advanced_columns)

    trace_ids = filtered.get("trace_id", pd.Series(dtype=str)).fillna("").astype(str)
    trace_options = [tid for tid in trace_ids.tolist() if tid.strip()]
    selected_trace_id = st.selectbox("Select trace_id", options=[""] + trace_options)

    if selected_trace_id:
        trace = traces_by_id.get(selected_trace_id, {})
        row = filtered[filtered.get("trace_id", "").astype(str) == selected_trace_id].head(1)
        row_dict = row.iloc[0].to_dict() if not row.empty else {}
        render_codeact_trace_viewer(
            trace=trace,
            row=row_dict,
            base_thread_url=base_thread_url,
            redact_by_default=True,
        )
    else:
        st.info("Select a trace to open the Trace Viewer.")

    csv_data = filtered.drop(columns=[c for c in ["decoded_code_blocks"] if c in filtered.columns]).to_csv(index=False)
    st.download_button(
        "Download inbox CSV",
        data=csv_data,
        file_name="codeact_inbox.csv",
        mime="text/csv",
    )

    with st.expander("Advanced exports"):
        try:
            template_summary_df, _template_traces_df = build_codeact_template_rollups(filtered)
        except Exception:
            template_summary_df = pd.DataFrame()

        if template_summary_df.empty:
            st.info("No template rollups available for current filters.")
        else:
            st.download_button(
                "Download template summary CSV",
                data=template_summary_df.to_csv(index=False),
                file_name="codeact_template_summary.csv",
                mime="text/csv",
            )
