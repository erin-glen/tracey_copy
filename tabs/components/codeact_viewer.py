"""Shared CodeAct trace viewer component."""

from __future__ import annotations

from typing import Any

import altair as alt
import pandas as pd
import streamlit as st

from utils.codeact_explorer_features import extract_codeact_timeline
from utils.codeact_utils import redact_secrets


_TEXT_PREVIEW_LIMIT = 1500
_EXEC_PREVIEW_LIMIT = 1500


VIEWER_GLOSSARY = {
    "Final insight": "The last narrative summary produced from the CodeAct run (what a user would read as the takeaway).",
    "Provenance": "Source URL(s) in raw_data used to compute results. Helps verify data origin and mixing.",
}


def _to_str(value: Any) -> str:
    return str(value or "").strip()


def _truncate_with_expander(text: str, limit: int, *, kind: str) -> None:
    if len(text) <= limit:
        if kind == "markdown":
            st.markdown(text)
        else:
            st.code(text, language="text", line_numbers=False, wrap_lines=True)
        return
    preview = text[:limit] + "..."
    if kind == "markdown":
        st.markdown(preview)
        with st.expander("Show more", expanded=False):
            st.markdown(text)
    else:
        st.code(preview, language="text", line_numbers=False, wrap_lines=True)
        with st.expander("Show more", expanded=False):
            st.code(text, language="text", line_numbers=False, wrap_lines=True)


def _build_provenance_rows(raw_data: object) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    def _walk(node: object) -> None:
        if isinstance(node, dict):
            source_url = node.get("source_url")
            if isinstance(source_url, str) and source_url.strip():
                rows.append(
                    {
                        "dataset_name": _to_str(node.get("dataset_name")),
                        "aoi_name": _to_str(node.get("aoi_name")),
                        "time_start": _to_str(node.get("start_date")),
                        "time_end": _to_str(node.get("end_date")),
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

    dedup: dict[tuple[str, str, str, str, str], dict[str, str]] = {}
    for row in rows:
        key = (
            row.get("source_url", ""),
            row.get("dataset_name", ""),
            row.get("aoi_name", ""),
            row.get("time_start", ""),
            row.get("time_end", ""),
        )
        if key not in dedup:
            dedup[key] = row
    return list(dedup.values())


def _render_charts_preview(charts: object) -> None:
    st.subheader("Charts preview", help="Charts derived from charts_data (and/or output tables).")
    if not isinstance(charts, list) or not charts:
        st.info("No charts_data found.")
        return

    for i, chart in enumerate(charts):
        if not isinstance(chart, dict):
            continue
        ctype = _to_str(chart.get("type")).lower()
        title = _to_str(chart.get("title")) or f"Chart {i + 1}"
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
            st.dataframe(
                pd.DataFrame(data) if isinstance(data, list) else pd.DataFrame([{"value": data}]),
                use_container_width=True,
            )


def render_codeact_trace_viewer(
    *,
    trace: dict,
    row: dict,
    base_thread_url: str,
    redact_by_default: bool = True,
) -> None:
    """Render audit, timeline, charts, and provenance for one selected CodeAct trace."""

    trace_id = _to_str(row.get("trace_id") or trace.get("id"))
    timestamp = _to_str(row.get("timestamp") or trace.get("timestamp"))
    session_id = _to_str(row.get("sessionId") or trace.get("sessionId"))
    thread_link = f"{base_thread_url.rstrip('/')}/{session_id}" if session_id else ""

    st.markdown("### Trace Viewer")
    with st.container(border=True):
        st.markdown("**Audit**")
        st.write(f"trace_id: `{trace_id}`")
        if timestamp:
            st.write(f"timestamp: `{timestamp}`")
        if thread_link:
            st.link_button("Open thread", thread_link)
        st.write(f"dataset_name: `{_to_str(row.get('dataset_name'))}`")
        st.write(f"dataset_family: `{_to_str(row.get('dataset_family'))}`")
        st.write(f"aoi_name: `{_to_str(row.get('aoi_name'))}`")
        st.write(f"time_start: `{_to_str(row.get('time_start'))}`")
        st.write(f"time_end: `{_to_str(row.get('time_end'))}`")

    st.subheader("QA summary", help="Deterministic flags and checks that indicate correctness risk.")
    final_insight = _to_str(row.get("codeact_final_insight"))
    if final_insight:
        st.subheader("Final insight", help=VIEWER_GLOSSARY["Final insight"])
        st.write(final_insight)

    checks = []
    for key in ["codeact_time_check", "codeact_dataset_check", "codeact_aoi_check"]:
        val = row.get(key)
        if val not in (None, ""):
            checks.append((key, _to_str(val)))
    if checks:
        st.markdown("**Consistency checks**")
        for key, val in checks:
            st.write(f"- {key}: {val}")

    true_flags = [col for col, val in row.items() if str(col).startswith("flag_") and bool(val)]
    st.markdown("**Flags (true only)**")
    if true_flags:
        for flag in true_flags:
            st.write(f"- {flag}")
    else:
        st.write("None")

    st.subheader("Code timeline", help="Ordered parts: text output → code block → execution output.")
    output_obj = trace.get("output") if isinstance(trace, dict) else {}
    output_obj = output_obj if isinstance(output_obj, dict) else {}
    timeline = extract_codeact_timeline(output_obj)
    reveal_unredacted = st.checkbox(
        "Reveal unredacted code",
        value=not redact_by_default,
        key=f"reveal_code_{trace_id}",
    )

    for i, part in enumerate(timeline):
        ptype = _to_str(part.get("type")) or "unknown"
        decoded = _to_str(part.get("decoded"))
        char_len = int(part.get("char_len") or len(decoded))
        with st.expander(f"{i + 1}. {ptype} ({char_len} chars)", expanded=False):
            if ptype == "code_block":
                shown = decoded if reveal_unredacted else redact_secrets(decoded)
                st.code(shown, language="python", line_numbers=True, wrap_lines=True)
            elif ptype == "text_output":
                _truncate_with_expander(decoded, _TEXT_PREVIEW_LIMIT, kind="markdown")
            elif ptype == "execution_output":
                _truncate_with_expander(decoded, _EXEC_PREVIEW_LIMIT, kind="code")
            else:
                st.text(decoded)

    _render_charts_preview(output_obj.get("charts_data"))

    st.subheader("Provenance", help=VIEWER_GLOSSARY["Provenance"])
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
