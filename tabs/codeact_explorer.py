"""CodeAct explorer tab with deterministic filtering and trace viewing."""

from __future__ import annotations

import hashlib
import json

import altair as alt
import pandas as pd
import streamlit as st

from utils.codeact_explorer_features import (
    add_codeact_explorer_columns,
    extract_codeact_timeline,
)
from utils.codeact_qaqc import add_codeact_qaqc_columns
from utils.codeact_utils import redact_secrets
from utils.content_kpis import compute_derived_interactions
from utils.trace_parsing import normalize_trace_format


FLAG_COLUMNS = [
    "flag_missing_final_insight",
    "flag_hardcoded_chart_data",
    "flag_requery_in_later_block",
    "flag_multi_source_provenance",
    "flag_endpoint_base_differs_from_dataset_endpoint",
    "flag_pie_percent_sum_off",
]


def _trace_fingerprint(traces: list[dict]) -> str:
    ids = []
    for trace in traces:
        tid = str((trace or {}).get("id") or "").strip()
        if tid:
            ids.append(tid)
    joined = "|".join(sorted(ids))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def _build_provenance_rows(raw_data: object) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    def _walk(node: object) -> None:
        if isinstance(node, dict):
            source_url = node.get("source_url")
            if isinstance(source_url, str) and source_url.strip():
                rows.append(
                    {
                        "dataset_name": str(node.get("dataset_name") or ""),
                        "aoi_name": str(node.get("aoi_name") or ""),
                        "start_date": str(node.get("start_date") or ""),
                        "end_date": str(node.get("end_date") or ""),
                        "source_url": source_url.strip(),
                    }
                )
            for value in node.values():
                if isinstance(value, (dict, list)):
                    _walk(value)
        elif isinstance(node, list):
            for item in node:
                if isinstance(item, (dict, list)):
                    _walk(item)

    _walk(raw_data)

    dedup: dict[str, dict[str, str]] = {}
    for row in rows:
        url = row.get("source_url", "")
        if url and url not in dedup:
            dedup[url] = row
    return list(dedup.values())


def _to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def render(base_thread_url: str) -> None:
    st.subheader("🔎 CodeAct Explorer")
    st.caption("Browse CodeAct traces, apply deterministic categories/flags, and inspect decoded timelines.")

    traces = st.session_state.get("stats_traces", [])
    if not traces:
        st.info("Fetch traces in sidebar first")
        return

    normed: list[dict] = []
    traces_by_id: dict[str, dict] = {}
    for trace in traces:
        try:
            n = normalize_trace_format(trace)
            normed.append(n)
            tid = str(n.get("id") or "").strip()
            if tid:
                traces_by_id[tid] = n
        except Exception:
            continue

    if not normed:
        st.warning("No valid traces were available after normalization.")
        return

    fp = _trace_fingerprint(normed)
    cached_df = st.session_state.get("content_kpis_df")
    cached_fp = str(st.session_state.get("content_kpis_fingerprint") or "")
    if isinstance(cached_df, pd.DataFrame) and not cached_df.empty and cached_fp == fp:
        derived = cached_df.copy()
    else:
        derived = compute_derived_interactions(normed)
        st.session_state["content_kpis_df"] = derived
        st.session_state["content_kpis_fingerprint"] = fp

    qaqc_df = add_codeact_qaqc_columns(derived, traces_by_id)
    explorer_df = add_codeact_explorer_columns(qaqc_df, traces_by_id)
    st.session_state["content_kpis_df"] = explorer_df

    left_col, right_col = st.columns([3, 2])

    with left_col:
        only_codeact = st.checkbox("Only CodeAct traces", value=True)

        filtered = explorer_df.copy()
        if only_codeact:
            filtered = filtered[filtered.get("codeact_present", False).fillna(False).astype(bool)]

        def _ms(label: str, col: str) -> list[str]:
            opts = sorted({str(x) for x in filtered.get(col, pd.Series(dtype=str)).fillna("") if str(x).strip()})
            return st.multiselect(label, opts)

        dataset_families = _ms("dataset_family", "dataset_family")
        dataset_names = _ms("dataset_name", "dataset_name")
        intents = _ms("intent_primary", "intent_primary")
        prep_modes = _ms("codeact_chart_prep_mode", "codeact_chart_prep_mode")
        retrieval_modes = _ms("codeact_retrieval_mode", "codeact_retrieval_mode")
        template_ids = _ms("codeact_template_id", "codeact_template_id")

        chart_type_opts: set[str] = set()
        for val in filtered.get("codeact_chart_types", pd.Series(dtype=str)).fillna(""):
            for part in str(val).split(","):
                if part.strip():
                    chart_type_opts.add(part.strip())
        selected_chart_types = st.multiselect("chart types", sorted(chart_type_opts))

        available_flags = [c for c in FLAG_COLUMNS if c in filtered.columns]
        selected_flags = st.multiselect("flags", available_flags)
        only_issues = st.checkbox("Only consistency issues", value=False)

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
        if template_ids and "codeact_template_id" in filtered.columns:
            filtered = filtered[filtered["codeact_template_id"].astype(str).isin(template_ids)]
        if selected_chart_types:
            mask = []
            for val in filtered.get("codeact_chart_types", pd.Series(dtype=str)).fillna(""):
                parts = {p.strip() for p in str(val).split(",") if p.strip()}
                mask.append(bool(parts.intersection(selected_chart_types)))
            filtered = filtered[pd.Series(mask, index=filtered.index)]
        if selected_flags:
            mask = pd.Series(False, index=filtered.index)
            for flag_col in selected_flags:
                mask = mask | filtered.get(flag_col, False).fillna(False).astype(bool)
            filtered = filtered[mask]
        if only_issues and "codeact_consistency_issue" in filtered.columns:
            filtered = filtered[filtered["codeact_consistency_issue"].fillna(False).astype(bool)]

        filtered = filtered.copy()
        if "sessionId" in filtered.columns:
            filtered["thread_url"] = filtered["sessionId"].apply(
                lambda s: f"{base_thread_url.rstrip('/')}/{s}" if str(s).strip() else ""
            )
        else:
            filtered["thread_url"] = ""

        table_cols = [
            "timestamp",
            "trace_id",
            "sessionId",
            "thread_url",
            "intent_primary",
            "dataset_name",
            "dataset_family",
            "codeact_template_id",
            "codeact_retrieval_mode",
            "codeact_chart_prep_mode",
            "codeact_chart_count",
            "codeact_chart_types",
            "codeact_source_url_count",
            "flag_missing_final_insight",
            "flag_hardcoded_chart_data",
            "flag_requery_in_later_block",
            "flag_multi_source_provenance",
            "codeact_consistency_issue",
        ]
        present_cols = [c for c in table_cols if c in filtered.columns]
        st.dataframe(
            filtered[present_cols],
            use_container_width=True,
            column_config={
                "thread_url": st.column_config.LinkColumn("Thread URL"),
            },
        )

        export_df = filtered.copy()
        st.download_button(
            "Download filtered CodeAct CSV",
            data=_to_csv_bytes(export_df),
            file_name="codeact_explorer_filtered.csv",
            mime="text/csv",
        )

        trace_options = filtered.get("trace_id", pd.Series(dtype=str)).astype(str).tolist() if not filtered.empty else []
        selected_trace_id = st.selectbox("Select trace_id", options=[""] + trace_options)

    with right_col:
        if not selected_trace_id:
            st.info("Select a trace to open Trace Viewer.")
            return

        trace = traces_by_id.get(selected_trace_id, {})
        output_obj = trace.get("output") if isinstance(trace, dict) else {}
        output_obj = output_obj if isinstance(output_obj, dict) else {}

        row = explorer_df[explorer_df.get("trace_id", "").astype(str) == selected_trace_id].head(1)
        row_dict = row.iloc[0].to_dict() if not row.empty else {}

        st.markdown("### Trace Viewer")
        st.markdown("**Prompt**")
        st.write(str(row_dict.get("prompt") or ""))
        st.markdown("**Response**")
        st.write(str(row_dict.get("response") or ""))
        st.markdown("**Final Insight**")
        st.write(str(row_dict.get("codeact_final_insight") or ""))

        true_flags = [flag for flag in FLAG_COLUMNS if bool(row_dict.get(flag, False))]
        st.markdown("**Flags**")
        if true_flags:
            for flag in true_flags:
                label = f"{flag} (informational)" if flag == "flag_endpoint_base_differs_from_dataset_endpoint" else flag
                st.write(f"- {label}")
        else:
            st.write("None")

        checks = [
            ("codeact_time_check", row_dict.get("codeact_time_check")),
            ("codeact_dataset_check", row_dict.get("codeact_dataset_check")),
            ("codeact_aoi_check", row_dict.get("codeact_aoi_check")),
        ]
        check_lines = [f"{k}: {v}" for k, v in checks if v not in (None, "")]
        if check_lines:
            st.markdown("**QAQC param checks**")
            for line in check_lines:
                st.write(f"- {line}")

        st.markdown("### Timeline")
        reveal_unredacted = st.checkbox("Reveal unredacted code (danger)", value=False)
        timeline = extract_codeact_timeline(output_obj)
        redacted_blocks: list[str] = []
        full_blocks: list[str] = []

        for i, part in enumerate(timeline):
            ptype = str(part.get("type") or "unknown")
            decoded = str(part.get("decoded") or "")
            char_len = int(part.get("char_len") or len(decoded))
            with st.expander(f"{i + 1}. {ptype} ({char_len} chars)", expanded=False):
                if ptype == "code_block":
                    redacted = redact_secrets(decoded)
                    redacted_blocks.append(redacted)
                    full_blocks.append(decoded)
                    shown = decoded if reveal_unredacted else redacted
                    st.code(shown, language="python", line_numbers=True, wrap_lines=True)
                elif ptype == "execution_output":
                    if len(decoded) <= 1500:
                        st.code(decoded)
                    else:
                        st.code(decoded[:1500] + "...")
                        with st.expander("Show full output", expanded=False):
                            st.code(decoded)
                elif ptype == "text_output":
                    if len(decoded) <= 1500:
                        st.markdown(decoded)
                    else:
                        st.markdown(decoded[:1500] + "...")
                        with st.expander("Show full text", expanded=False):
                            st.markdown(decoded)
                else:
                    st.text(decoded)

        code_payload = "\n\n".join(full_blocks if reveal_unredacted else redacted_blocks)
        st.download_button(
            "Download decoded code blocks (.txt)",
            data=code_payload.encode("utf-8"),
            file_name=f"codeact_{selected_trace_id}.txt",
            mime="text/plain",
        )

        st.markdown("### Charts preview")
        charts = output_obj.get("charts_data")
        if not isinstance(charts, list) or not charts:
            st.info("No charts_data found.")
        else:
            for i, chart in enumerate(charts):
                if not isinstance(chart, dict):
                    continue
                ctype = str(chart.get("type") or "").lower()
                title = str(chart.get("title") or f"Chart {i + 1}")
                st.markdown(f"**{title}** ({ctype or 'unknown'})")
                data = chart.get("data")
                rendered = False
                if ctype in {"pie", "bar", "line"} and isinstance(data, list) and data and all(isinstance(r, dict) for r in data):
                    cdf = pd.DataFrame(data)
                    if {"label", "value"}.issubset(cdf.columns):
                        if ctype == "pie":
                            chart_obj = alt.Chart(cdf).mark_arc().encode(theta="value:Q", color="label:N")
                        elif ctype == "bar":
                            chart_obj = alt.Chart(cdf).mark_bar().encode(x="label:N", y="value:Q")
                        else:
                            chart_obj = alt.Chart(cdf).mark_line(point=True).encode(x="label:N", y="value:Q")
                        st.altair_chart(chart_obj, use_container_width=True)
                        rendered = True
                if not rendered:
                    st.dataframe(pd.DataFrame(data) if isinstance(data, list) else pd.DataFrame([{"value": data}]), use_container_width=True)
                with st.expander("Show chart data", expanded=False):
                    st.dataframe(pd.DataFrame(data) if isinstance(data, list) else pd.DataFrame([{"value": data}]), use_container_width=True)

        st.markdown("### Provenance")
        prov_rows = _build_provenance_rows(output_obj.get("raw_data"))
        if not prov_rows:
            st.info("No source_url records in raw_data.")
        else:
            prov_df = pd.DataFrame(prov_rows)
            st.dataframe(
                prov_df,
                use_container_width=True,
                column_config={"source_url": st.column_config.LinkColumn("source_url")},
            )

        with st.expander("Trace output JSON", expanded=False):
            st.code(json.dumps(output_obj, indent=2, ensure_ascii=False), language="json")
