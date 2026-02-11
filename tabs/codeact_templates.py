"""CodeAct template clustering + parameter consistency QA tab."""

from __future__ import annotations

import hashlib

import pandas as pd
import streamlit as st

from utils.codeact_qaqc import add_codeact_qaqc_columns, build_codeact_template_rollups
from utils.codeact_utils import iter_decoded_codeact_parts, redact_secrets, truncate_text as codeact_truncate_text
from utils.content_kpis import compute_derived_interactions
from utils.trace_parsing import normalize_trace_format
from utils.docs_ui import render_page_help, metric_with_help


def _trace_fingerprint(traces: list[dict]) -> str:
    ids = []
    for trace in traces:
        tid = str((trace or {}).get("id") or "").strip()
        ts = str((trace or {}).get("timestamp") or "").strip()
        if tid:
            ids.append(f"{tid}:{ts}")
    joined = "|".join(sorted(ids))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def _thread_link(base_thread_url: str, session_id: str) -> str:
    sid = str(session_id or "").strip()
    if not sid:
        return ""
    return f"{base_thread_url.rstrip('/')}/{sid}"


def render(base_thread_url: str) -> None:
    st.title("🧩 CodeAct Templates")
    st.caption("Deterministic template clustering and parameter-consistency QA for CodeAct traces.")

    render_page_help("codeact_templates", expanded=False)

    traces = st.session_state.get("stats_traces") or []
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
    derived = st.session_state.get("content_kpis_df")
    if not isinstance(derived, pd.DataFrame) or derived.empty:
        derived = compute_derived_interactions(normed)
        st.session_state["content_kpis_df"] = derived
        st.session_state["content_kpis_fingerprint"] = fp

    qaqc_df = st.session_state.get("codeact_qaqc_df")
    qaqc_fp = str(st.session_state.get("codeact_qaqc_fingerprint") or "")
    if not isinstance(qaqc_df, pd.DataFrame) or qaqc_df.empty or qaqc_fp != fp:
        qaqc_df = add_codeact_qaqc_columns(derived, traces_by_id)
        st.session_state["codeact_qaqc_df"] = qaqc_df
        st.session_state["codeact_qaqc_fingerprint"] = fp
        st.session_state["content_kpis_df"] = qaqc_df

    template_summary_df, template_traces_df = build_codeact_template_rollups(qaqc_df)

    codeact_df = qaqc_df[qaqc_df.get("codeact_present", False).fillna(False).astype(bool)].copy()
    n_codeact = len(codeact_df)
    n_templates = len(template_summary_df)
    issue_rate = float(codeact_df.get("codeact_consistency_issue", False).fillna(False).astype(bool).mean()) if n_codeact else 0.0

    c1, c2, c3 = st.columns(3)
    with c1:
        metric_with_help("CodeAct traces", n_codeact, metric_id="codeact_traces", key="codeact_n_traces")
    with c2:
        metric_with_help("Templates", n_templates, metric_id="codeact_templates", key="codeact_n_templates")
    with c3:
        metric_with_help(
            "Consistency issue rate",
            f"{issue_rate * 100:.1f}%",
            metric_id="codeact_consistency_issue_rate",
            key="codeact_issue_rate",
        )

    if n_codeact:
        by_intent = (
            codeact_df.groupby("intent_primary", dropna=False)["codeact_consistency_issue"].mean().reset_index().rename(columns={"codeact_consistency_issue": "issue_rate"})
        )
        st.write("Issue rates by intent")
        st.dataframe(by_intent.sort_values("issue_rate", ascending=False), use_container_width=True)

    st.subheader("Template table")
    min_size = int(st.number_input("Min template size", min_value=1, max_value=10_000, value=5, step=1))
    min_issue_rate = float(st.slider("Only templates with issue_rate >=", min_value=0.0, max_value=1.0, value=0.2, step=0.05))
    scored_only = st.checkbox("Scored intents only (trend_over_time, data_lookup)", value=False)

    filtered_summary = template_summary_df.copy()
    if not filtered_summary.empty:
        issue_series = 1.0 - (
            (1.0 - filtered_summary.get("time_issue_rate", 0.0))
            * (1.0 - filtered_summary.get("dataset_issue_rate", 0.0))
            * (1.0 - filtered_summary.get("aoi_issue_rate", 0.0))
        )
        filtered_summary["template_issue_rate_est"] = issue_series
        filtered_summary = filtered_summary[filtered_summary["n_traces"] >= min_size]
        filtered_summary = filtered_summary[filtered_summary["template_issue_rate_est"] >= min_issue_rate]

        if scored_only and not template_traces_df.empty:
            good_templates = set(
                template_traces_df.loc[
                    template_traces_df["intent_primary"].isin(["trend_over_time", "data_lookup"]), "codeact_template_id"
                ].astype(str)
            )
            filtered_summary = filtered_summary[filtered_summary["codeact_template_id"].isin(good_templates)]

    st.dataframe(filtered_summary, use_container_width=True)

    st.subheader("Template drilldown")
    options = filtered_summary["codeact_template_id"].astype(str).tolist() if not filtered_summary.empty else []
    selected_template = st.selectbox("Select template_id", options=[""] + options)

    if selected_template:
        selected_summary = template_summary_df[template_summary_df["codeact_template_id"] == selected_template].head(1)
        if not selected_summary.empty:
            st.write(selected_summary)

        trace_rows = template_traces_df[template_traces_df["codeact_template_id"] == selected_template].copy()
        rep_trace_id = (
            selected_summary.iloc[0].get("representative_trace_id", "")
            if not selected_summary.empty
            else (trace_rows.iloc[0]["trace_id"] if not trace_rows.empty else "")
        )

        if rep_trace_id:
            rep = trace_rows[trace_rows["trace_id"] == rep_trace_id].head(1)
            if rep.empty:
                rep = trace_rows.head(1)
            if not rep.empty:
                session_id = str(rep.iloc[0].get("sessionId", ""))
                st.markdown(
                    f"**Representative trace_id:** `{rep_trace_id}`  \n"
                    f"**Session link:** {_thread_link(base_thread_url, session_id)}"
                )

            st.markdown("**Representative Code**")
            if st.button("Load representative code blocks", key=f"load_code_{selected_template}"):
                trace_obj = traces_by_id.get(str(rep_trace_id), {})
                parts = iter_decoded_codeact_parts(trace_obj.get("output", {}) if isinstance(trace_obj, dict) else {})
                code_blocks = [p.get("decoded", "") for p in parts if p.get("type") == "code_block"]
                st.session_state[f"code_blocks_{selected_template}"] = code_blocks

            reveal_full = st.checkbox(
                "Reveal full code blocks (may contain sensitive data)",
                value=False,
                key=f"reveal_full_{selected_template}",
            )
            do_redact = st.checkbox("Redact likely secrets", value=True, key=f"redact_{selected_template}")
            max_chars = int(st.number_input("Per-block max chars", min_value=500, max_value=20_000, value=3000, step=250))

            blocks = st.session_state.get(f"code_blocks_{selected_template}", [])
            for i, block in enumerate(blocks):
                rendered = str(block or "")
                if do_redact:
                    rendered = redact_secrets(rendered)
                if not reveal_full:
                    rendered = codeact_truncate_text(rendered, max_chars)
                st.code(rendered, language="python")
                st.caption(f"Block {i + 1}")

        if not trace_rows.empty:
            trace_rows = trace_rows.copy()
            trace_rows["thread_link"] = trace_rows["sessionId"].map(lambda x: _thread_link(base_thread_url, x))
            cols = [
                "timestamp",
                "trace_id",
                "thread_link",
                "sessionId",
                "thread_id",
                "intent_primary",
                "completion_state",
                "codeact_time_check",
                "codeact_dataset_check",
                "codeact_aoi_check",
                "codeact_consistency_reason",
            ]
            cols = [c for c in cols if c in trace_rows.columns]
            st.dataframe(
                trace_rows[cols],
                use_container_width=True,
                column_config={"thread_link": st.column_config.LinkColumn("Thread URL")},
            )

            st.download_button(
                "Download filtered template_traces.csv",
                data=trace_rows.to_csv(index=False).encode("utf-8"),
                file_name=f"codeact_template_{selected_template}_traces.csv",
                mime="text/csv",
            )

    st.subheader("Exports")
    st.download_button(
        "Download template_summary.csv",
        data=template_summary_df.to_csv(index=False).encode("utf-8"),
        file_name="codeact_template_summary.csv",
        mime="text/csv",
    )
