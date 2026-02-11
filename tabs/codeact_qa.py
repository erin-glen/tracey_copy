"""Simplified CodeAct QA tab: Inbox -> Trace Viewer."""

from __future__ import annotations

import hashlib
from typing import Any

import pandas as pd
import streamlit as st

from tabs.components.codeact_viewer import render_codeact_trace_viewer
from utils.codeact_explorer_features import add_codeact_explorer_columns
from utils.codeact_qaqc import add_codeact_qaqc_columns
from utils.content_kpis import compute_derived_interactions
from utils.shared_ui import render_glossary_popover
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
    "Consistency issues",
    "Hardcoded chart data",
    "Errors / no data",
]

CODEACT_GLOSSARY = {
    "CodeAct": "A trace format where the assistant emits code blocks and execution outputs to compute results (charts/metrics).",
    "Preset": "A saved filter for common QA review slices (e.g., needs review, errors).",
    "Needs review": "A trace that triggered at least one deterministic QA flag, a consistency issue, or an error/no-data outcome.",
    "Review reason": "A short summary of which QA checks triggered review for this trace.",
    "Consistency issue": "The code suggests the run changed dataset/AOI/time unexpectedly across blocks, or parameters don’t match outputs.",
    "Provenance": "Source URL(s) in raw_data used to compute results. Helps verify data origin and mixing.",
    "Hardcoded chart data": "Chart values appear embedded directly in the code/output instead of derived from analytics/raw data.",
    "Multi-source provenance": "More than one distinct source URL was used in a single run (can indicate mixing or re-query).",
    "Pie % sum off": "Pie-chart percentages don’t sum to ~100% (within tolerance), suggesting formatting/math issues.",
    "Final insight": "The last narrative summary produced from the CodeAct run (what a user would read as the takeaway).",
}

COMPACT_INBOX_COLUMNS = [
    "timestamp",
    "completion_state",
    "dataset_family",
    "dataset_name",
    "aoi_name",
    "time_start",
    "time_end",
    "codeact_chart_types",
    "codeact_source_url_count",
    "flags_count",
    "review_reason",
    "trace_id",
    "thread_url",
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
        if completion_series.at[idx] == "error":
            tokens.append("error")
        if completion_series.at[idx] == "no_data":
            tokens.append("no_data")
        review_reasons.append(_pipe_dedupe(tokens))

    out["review_reason"] = review_reasons
    out["needs_review"] = (
        (out["flags_count"] > 0)
        | consistency_series
        | completion_series.isin(["error", "no_data"])
        | (answer_type_series != "normal")
    )
    return out, flags_cols


def _apply_search(df: pd.DataFrame, search: str) -> pd.DataFrame:
    query = (search or "").strip().lower()
    if not query:
        return df
    columns = [c for c in ["aoi_name", "dataset_name", "dataset_family", "prompt", "response", "sessionId", "trace_id"] if c in df.columns]
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
    if preset == "Consistency issues":
        consistency_issue = _is_truthy(_series_or_default(df_ca, "codeact_consistency_issue", False))
        mismatch_tokens = {"missing", "mismatch"}
        time_issue = _series_or_default(df_ca, "codeact_time_check", "").astype(str).isin(mismatch_tokens)
        dataset_issue = _series_or_default(df_ca, "codeact_dataset_check", "").astype(str).isin(mismatch_tokens)
        aoi_issue = _series_or_default(df_ca, "codeact_aoi_check", "").astype(str).isin(mismatch_tokens)
        return df_ca[consistency_issue | time_issue | dataset_issue | aoi_issue]
    if preset == "Hardcoded chart data":
        if "flag_hardcoded_chart_data" in df_ca.columns:
            return df_ca[_is_truthy(_series_or_default(df_ca, "flag_hardcoded_chart_data", False))]
        return df_ca[df_ca.get("flags_csv", pd.Series([""] * len(df_ca), index=df_ca.index)).astype(str).str.contains("hardcoded_chart", na=False)]
    if preset == "Errors / no data":
        answer_type = _series_or_default(df_ca, "answer_type", "normal").astype(str)
        completion_state = _series_or_default(df_ca, "completion_state", "").astype(str)
        return df_ca[
            completion_state.isin(["error", "no_data"])
            | answer_type.isin(["model_error", "missing_output", "empty_or_short"])
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


def _render_inbox_table(df: pd.DataFrame) -> None:
    columns = [c for c in COMPACT_INBOX_COLUMNS if c in df.columns]
    st.dataframe(
        df[columns],
        use_container_width=True,
        column_config={"thread_url": st.column_config.LinkColumn("Thread")},
    )


def _trace_option_label(row: pd.Series) -> str:
    timestamp = str(row.get("timestamp") or "").strip() or "unknown-time"
    family = str(row.get("dataset_family") or "").strip() or "unknown-family"
    aoi = str(row.get("aoi_name") or "").strip() or "unknown-aoi"
    reason = str(row.get("review_reason") or "").strip() or "no_reason"
    return f"{timestamp} • {family} • {aoi} • {reason}"


def render(base_thread_url: str) -> None:
    st.title("🧩 CodeAct QA")
    st.caption("Use presets to triage CodeAct traces, then inspect the decoded timeline, charts, and provenance.")
    st.info(
        "CodeAct traces include code blocks + execution outputs used to generate charts and the final narrative insight.\n\n"
        "Use **Preset** to triage, optionally **Search**, then select a trace to inspect the timeline, charts, and provenance."
    )
    render_glossary_popover("📚 Glossary", CODEACT_GLOSSARY)

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

    preset = st.selectbox("Preset", PRESETS, index=0, help=CODEACT_GLOSSARY["Preset"])
    filtered = _apply_preset(df_ca, preset)

    search = st.text_input(
        "Search (AOI, dataset, prompt text)",
        value="",
        help="Substring search across AOI name, dataset fields, and prompt/response text when available.",
    )
    filtered = _apply_search(filtered, search)

    with st.expander("Optional filters", expanded=False):
        if "dataset_family" in filtered.columns:
            family_options = sorted({str(v) for v in filtered["dataset_family"].fillna("") if str(v).strip()})
            selected_families = st.multiselect(
                "Dataset family",
                options=family_options,
                default=[],
                help="High-level grouping of dataset_name (e.g., tree_cover_loss, alerts).",
            )
            if selected_families:
                filtered = filtered[filtered["dataset_family"].astype(str).isin(selected_families)]

    st.caption(f"{len(filtered):,} CodeAct traces match this view")
    _render_inbox_table(filtered)

    trace_options_df = filtered[filtered.get("trace_id", pd.Series(dtype=str)).fillna("").astype(str).str.strip() != ""].copy()

    if trace_options_df.empty:
        st.info("No trace IDs available for this view.")
    else:
        trace_options_df = trace_options_df.reset_index(drop=True)
        trace_labels = [_trace_option_label(row) for _, row in trace_options_df.iterrows()]
        selected_idx = st.selectbox(
            "Select trace",
            options=list(range(len(trace_options_df))),
            index=0,
            format_func=lambda idx: trace_labels[idx],
        )
        selected_trace_id = str(trace_options_df.iloc[selected_idx].get("trace_id") or "").strip()
        trace = traces_by_id.get(selected_trace_id, {})
        row_dict = trace_options_df.iloc[selected_idx].to_dict()
        render_codeact_trace_viewer(
            trace=trace,
            row=row_dict,
            base_thread_url=base_thread_url,
            redact_by_default=True,
        )

    csv_data = filtered.drop(columns=[c for c in ["decoded_code_blocks"] if c in filtered.columns]).to_csv(index=False)
    st.download_button(
        "Download inbox CSV",
        data=csv_data,
        file_name="codeact_inbox.csv",
        mime="text/csv",
    )
