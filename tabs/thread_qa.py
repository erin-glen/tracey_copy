"""Thread-level deterministic QA tab renderer."""

from __future__ import annotations

import hashlib
from datetime import datetime

import pandas as pd
import requests
import streamlit as st

from utils.content_kpis import build_thread_summary, compute_derived_interactions, compute_thread_key
from utils.langfuse_api import create_annotation_queue_item, get_langfuse_headers, list_annotation_queues
from utils.trace_parsing import normalize_trace_format
from utils.data_helpers import csv_bytes_any, init_session_state
from utils.docs_ui import render_page_help, metric_with_help


def _truncate(val: object, n: int = 160) -> str:
    txt = str(val or "")
    return txt if len(txt) <= n else txt[: n - 1] + "…"


def _split_csv_values(series: pd.Series) -> list[str]:
    vals: set[str] = set()
    for cell in series.fillna("").astype(str):
        for part in cell.split(","):
            v = part.strip()
            if v and not v.startswith("+"):
                vals.add(v)
    return sorted(vals)


def render(public_key: str, secret_key: str, base_url: str, base_thread_url: str) -> None:
    init_session_state(
        {
            "thread_qa_fingerprint": "",
            "thread_qa_thread_df": None,
            "thread_qa_selected_thread_key": "",
            "thread_qa_queue_id": "",
            "thread_qa_queue_list_cache": None,
            "thread_qa_queue_list_cache_at": None,
        }
    )

    traces = st.session_state.get("stats_traces", [])
    if not traces:
        st.info("Fetch traces in sidebar first")
        return

    normed = [normalize_trace_format(t) for t in traces]
    trace_ids = sorted(str(t.get("id") or "") for t in normed)
    fingerprint = hashlib.sha256("|".join(trace_ids).encode("utf-8")).hexdigest()

    if (
        st.session_state.get("content_kpis_df") is None
        or st.session_state.get("content_kpis_fingerprint") != fingerprint
    ):
        derived = compute_derived_interactions(normed)
        st.session_state["content_kpis_df"] = derived
        st.session_state["content_kpis_fingerprint"] = fingerprint
    else:
        derived = st.session_state.get("content_kpis_df", pd.DataFrame())

    if derived.empty:
        st.warning("No prompt-bearing traces available for thread QA.")
        return

    if fingerprint != st.session_state.get("thread_qa_fingerprint"):
        thread_df = build_thread_summary(derived, timestamp_col="timestamp")
        st.session_state["thread_qa_thread_df"] = thread_df
        st.session_state["thread_qa_fingerprint"] = fingerprint

    thread_df: pd.DataFrame = st.session_state.get("thread_qa_thread_df", pd.DataFrame())

    st.subheader("🧵 Thread QA")
    st.caption("Thread-level QA rollups derived deterministically from the currently loaded traces.")

    render_page_help("thread_qa", expanded=False)

    threads_count = len(thread_df)
    ended_nui_count = int(thread_df["ended_after_needs_user_input"].sum()) if threads_count else 0
    ended_err_count = int(thread_df["ended_after_error"].sum()) if threads_count else 0
    never_complete_count = int((~thread_df["ever_complete_answer"].fillna(False)).sum()) if threads_count else 0
    median_turns = float(thread_df["n_turns"].median()) if threads_count else 0.0

    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        metric_with_help("Threads", f"{threads_count}", metric_id="threads_total", key="threadqa_threads_total")
    with m2:
        metric_with_help(
            "Ended after needs input",
            f"{ended_nui_count}",
            metric_id="threads_ended_after_needs_input",
            delta=f"{(ended_nui_count / threads_count * 100):.1f}%" if threads_count else "0.0%",
            key="threadqa_ended_nui",
        )
    with m3:
        metric_with_help(
            "Ended after error",
            f"{ended_err_count}",
            metric_id="threads_ended_after_error",
            delta=f"{(ended_err_count / threads_count * 100):.1f}%" if threads_count else "0.0%",
            key="threadqa_ended_err",
        )
    with m4:
        metric_with_help("Never complete", f"{never_complete_count}", metric_id="threads_never_complete", key="threadqa_never_complete")
    with m5:
        metric_with_help("Median turns/thread", f"{median_turns:.1f}", metric_id="median_turns_per_thread", key="threadqa_median_turns")

    preset = st.selectbox(
        "Preset",
        [
            "Needs review (recommended)",
            "All threads",
            "Ended after needs input",
            "Ended after error",
            "Never complete",
            "Long threads (>= 5 turns)",
            "Many datasets (>= 3)",
        ],
    )
    search = st.text_input("Search (sessionId / dataset family / needs reason)", "")

    with st.expander("Advanced filters", expanded=False):
        only_ended_nui = st.checkbox("Only threads that ended after needs_user_input", value=False)
        only_ended_err = st.checkbox("Only threads that ended after error", value=False)
        only_no_complete = st.checkbox("Only threads with no complete answers", value=False)
        needs_options = _split_csv_values(thread_df.get("needs_user_input_reasons", pd.Series(dtype=str)))
        family_options = _split_csv_values(thread_df.get("dataset_families_seen", pd.Series(dtype=str)))
        selected_needs = st.multiselect("Needs reasons", options=needs_options)
        selected_families = st.multiselect("Dataset families", options=family_options)
        min_turns_opt = st.selectbox("Min turns", options=["All", "1+", "2+", "3+", "5+"], index=0)

    filtered = thread_df.copy()
    if preset == "Needs review (recommended)":
        filtered = filtered[
            (filtered["ended_after_needs_user_input"] == True)
            | (filtered["ended_after_error"] == True)
            | (filtered["ever_complete_answer"] != True)
        ]
    elif preset == "Ended after needs input":
        filtered = filtered[filtered["ended_after_needs_user_input"] == True]
    elif preset == "Ended after error":
        filtered = filtered[filtered["ended_after_error"] == True]
    elif preset == "Never complete":
        filtered = filtered[filtered["ever_complete_answer"] != True]
    elif preset == "Long threads (>= 5 turns)":
        filtered = filtered[filtered["n_turns"] >= 5]
    elif preset == "Many datasets (>= 3)":
        filtered = filtered[filtered["datasets_seen_count"] >= 3]

    if search.strip():
        search_lower = search.strip().lower()
        search_cols = [
            c
            for c in ["sessionId", "datasets_seen", "dataset_families_seen", "needs_user_input_reasons"]
            if c in filtered.columns
        ]
        if search_cols:
            search_blob = (
                filtered[search_cols]
                .fillna("")
                .astype(str)
                .agg(" ".join, axis=1)
                .str.lower()
            )
            filtered = filtered[search_blob.str.contains(search_lower, na=False)]

    if only_ended_nui:
        filtered = filtered[filtered["ended_after_needs_user_input"] == True]
    if only_ended_err:
        filtered = filtered[filtered["ended_after_error"] == True]
    if only_no_complete:
        filtered = filtered[filtered["ever_complete_answer"] != True]

    if selected_needs:
        filtered = filtered[
            filtered["needs_user_input_reasons"].fillna("").apply(
                lambda v: any(r in {p.strip() for p in str(v).split(",") if p.strip()} for r in selected_needs)
            )
        ]
    if selected_families:
        filtered = filtered[
            filtered["dataset_families_seen"].fillna("").apply(
                lambda v: any(r in {p.strip() for p in str(v).split(",") if p.strip()} for r in selected_families)
            )
        ]

    min_turns_map = {"All": 0, "1+": 1, "2+": 2, "3+": 3, "5+": 5}
    if min_turns_map[min_turns_opt] > 0:
        filtered = filtered[filtered["n_turns"] >= min_turns_map[min_turns_opt]]

    display_df = filtered.copy()
    display_df["url"] = display_df["sessionId"].fillna("").map(
        lambda s: f"{base_thread_url.rstrip('/')}/{s}" if s else ""
    )
    compact_columns = [
        "end_utc",
        "n_turns",
        "last_completion_state",
        "last_needs_user_input_reason",
        "ended_after_needs_user_input",
        "ended_after_error",
        "datasets_seen_count",
        "dataset_families_seen",
        "sessionId",
        "url",
    ]
    if "dataset_families_seen" not in display_df.columns and "dataset_families_seen_count" in display_df.columns:
        compact_columns.insert(compact_columns.index("sessionId"), "dataset_families_seen_count")

    show_advanced = st.checkbox("Show advanced columns", value=False)
    advanced_columns = [
        "first_intent_primary",
        "last_intent_primary",
        "ever_complete_answer",
        "ever_needs_user_input",
        "ever_error",
        "datasets_seen",
        "dataset_families_seen_count",
        "needs_user_input_reasons",
    ]
    columns = compact_columns + (advanced_columns if show_advanced else [])
    columns = [c for c in columns if c in display_df.columns]
    st.dataframe(
        display_df[columns],
        use_container_width=True,
        column_config={"url": st.column_config.LinkColumn("Thread", display_text="Open")},
    )

    st.download_button(
        "Download thread_summary.csv",
        data=csv_bytes_any(thread_df.to_dict("records")),
        file_name="thread_summary.csv",
        mime="text/csv",
    )

    with st.expander("Thread drilldown", expanded=False):
        if filtered.empty:
            st.info("No threads match current filters.")
            return

        label_to_key: dict[str, str] = {}
        for row in filtered.to_dict("records"):
            label = f"{row.get('end_utc', '')} • {row.get('sessionId') or row.get('thread_id') or row.get('thread_key')} • turns={row.get('n_turns', 0)} • last={row.get('last_completion_state', '')}"
            label_to_key[label] = str(row.get("thread_key") or "")

        labels = list(label_to_key.keys())
        default_label = labels[0]
        current_key = str(st.session_state.get("thread_qa_selected_thread_key") or "")
        for label, key in label_to_key.items():
            if key == current_key:
                default_label = label
                break

        selected_label = st.selectbox("Select thread", options=labels, index=labels.index(default_label))
        selected_key = label_to_key[selected_label]
        st.session_state["thread_qa_selected_thread_key"] = selected_key

        turn_df = derived.copy()
        turn_df["thread_key"] = compute_thread_key(turn_df)
        turns = turn_df[turn_df["thread_key"] == selected_key].copy()
        turns["timestamp_dt"] = pd.to_datetime(turns["timestamp"], utc=True, errors="coerce")
        turns = turns.sort_values("timestamp_dt", na_position="last")

        turns["prompt_snippet"] = turns.get("prompt", "").map(lambda x: _truncate(x, 160))
        turns["response_snippet"] = turns.get("response", "").map(lambda x: _truncate(x, 160))

        turn_cols = [
            "timestamp",
            "trace_id",
            "intent_primary",
            "completion_state",
            "needs_user_input_reason",
            "struct_fail_reason",
            "dataset_family",
            "dataset_name",
            "answer_type",
            "codeact_present",
            "prompt_snippet",
            "response_snippet",
        ]
        turn_cols = [c for c in turn_cols if c in turns.columns]
        st.dataframe(turns[turn_cols], use_container_width=True)

        c1, c2 = st.columns(2)
        c1.download_button(
            "Download selected_thread_turns.csv",
            data=csv_bytes_any(turns[turn_cols].to_dict("records")),
            file_name="selected_thread_turns.csv",
            mime="text/csv",
        )
        trace_ids_text = "\n".join(turns["trace_id"].fillna("").astype(str).tolist())
        c2.download_button(
            "Download selected_thread_trace_ids.txt",
            data=trace_ids_text.encode("utf-8"),
            file_name="selected_thread_trace_ids.txt",
            mime="text/plain",
        )

        with st.expander("📥 Add this thread to a Langfuse annotation queue", expanded=False):
            default_queue_id = (
                st.session_state.get("human_eval_active_queue_id")
                or st.session_state.get("human_eval_queue_id")
                or ""
            )
            queue_id = st.text_input(
                "Queue ID",
                value=st.session_state.get("thread_qa_queue_id") or default_queue_id,
                key="thread_qa_queue_id",
            )

            if st.button("Load queues"):
                headers = get_langfuse_headers(public_key=public_key, secret_key=secret_key)
                try:
                    queues = list_annotation_queues(base_url=base_url, headers=headers)
                    st.session_state["thread_qa_queue_list_cache"] = queues
                    st.session_state["thread_qa_queue_list_cache_at"] = datetime.utcnow().isoformat()
                except Exception as exc:
                    st.error(f"Failed to load queues: {exc}")

            queues_cache = st.session_state.get("thread_qa_queue_list_cache") or []
            if queues_cache:
                options = [""] + [str(q.get("id") or "") for q in queues_cache]
                chosen = st.selectbox("Choose loaded queue", options=options)
                if chosen:
                    st.session_state["thread_qa_queue_id"] = chosen
                    queue_id = chosen

            if st.button("Add all traces in this thread to queue"):
                if not queue_id:
                    st.warning("Provide a queue ID first.")
                else:
                    headers = get_langfuse_headers(public_key=public_key, secret_key=secret_key)
                    count_success = 0
                    count_exists = 0
                    count_fail = 0
                    for trace_id in turns["trace_id"].fillna("").astype(str):
                        if not trace_id:
                            continue
                        try:
                            create_annotation_queue_item(
                                base_url=base_url,
                                headers=headers,
                                queue_id=queue_id,
                                object_id=trace_id,
                            )
                            count_success += 1
                        except requests.HTTPError as exc:
                            if exc.response is not None and exc.response.status_code == 409:
                                count_exists += 1
                            else:
                                count_fail += 1
                        except Exception:
                            count_fail += 1

                    st.write(
                        f"Added: {count_success}  |  Already existed: {count_exists}  |  Failed: {count_fail}"
                    )
