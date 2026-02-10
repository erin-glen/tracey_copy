"""QA sample pack tab."""

from __future__ import annotations

import hashlib
import json

import pandas as pd
import requests
import streamlit as st

from utils import (
    normalize_trace_format,
    compute_derived_interactions,
    SAMPLE_PACKS,
    build_sample_pack_df,
    add_codeact_snippets_for_pack,
    get_langfuse_headers,
    create_annotation_queue_item,
    list_annotation_queues,
    build_preset_mask,
)
from utils.data_helpers import init_session_state


def _trace_fingerprint(traces: list[dict]) -> str:
    ids = []
    for trace in traces:
        tid = str((trace or {}).get("id") or "").strip()
        if tid:
            ids.append(tid)
    joined = "|".join(sorted(ids))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def render(public_key: str, secret_key: str, base_url: str, base_thread_url: str) -> None:
    """Render deterministic QA sample pack workflow."""
    st.subheader("📦 QA Sample Packs")
    st.caption("Deterministic sample packs that mirror the pipeline’s sample CSV outputs.")

    init_session_state(
        {
            "qa_samples_queue_id": "",
            "qa_samples_queues": [],
        }
    )

    traces = st.session_state.get("stats_traces", [])
    if not traces:
        st.info("Fetch traces in sidebar first")
        return

    normed: list[dict] = []
    trace_by_id: dict[str, dict] = {}
    for trace in traces:
        try:
            n = normalize_trace_format(trace)
            normed.append(n)
            tid = str(n.get("id") or "").strip()
            if tid:
                trace_by_id[tid] = n
        except Exception:
            continue

    if not normed:
        st.warning("No valid traces were available after normalization.")
        return

    fingerprint = _trace_fingerprint(normed)
    cached_df = st.session_state.get("content_kpis_df")
    cached_fp = str(st.session_state.get("content_kpis_fingerprint") or "")

    if isinstance(cached_df, pd.DataFrame) and not cached_df.empty and cached_fp == fingerprint:
        derived = cached_df
    else:
        derived = compute_derived_interactions(normed)
        st.session_state.content_kpis_df = derived
        st.session_state.content_kpis_fingerprint = fingerprint

    pack_id = st.selectbox(
        "Sample pack",
        options=list(SAMPLE_PACKS.keys()),
        format_func=lambda key: SAMPLE_PACKS[key]["label"],
    )
    max_rows = int(st.number_input("Max rows", min_value=1, max_value=5000, value=200, step=1))

    with st.expander("Pack definition", expanded=False):
        st.write(SAMPLE_PACKS[pack_id]["description"])

    pack_df = build_sample_pack_df(derived, pack_id, max_rows=max_rows)
    if pack_id in {"codeact_examples", "codeact_param_issues"}:
        pack_df = add_codeact_snippets_for_pack(pack_df, trace_by_id, max_chars=500)

    preset_id = SAMPLE_PACKS[pack_id]["preset_id"]
    uncapped_count = int(build_preset_mask(derived, preset_id).sum())
    c1, c2 = st.columns(2)
    c1.metric("Candidates (uncapped)", uncapped_count)
    c2.metric("Rows in export", len(pack_df))

    preview_df = pack_df.head(200).copy()
    if "sessionId" in preview_df.columns:
        preview_df["thread_url"] = preview_df["sessionId"].fillna("").astype(str).apply(
            lambda sid: f"{base_thread_url.rstrip('/')}/{sid}" if sid.strip() else ""
        )

    column_config = None
    if "thread_url" in preview_df.columns:
        column_config = {"thread_url": st.column_config.LinkColumn("Thread URL")}
    st.dataframe(preview_df, use_container_width=True, column_config=column_config)

    csv_bytes = pack_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=f"Download {pack_id}.csv",
        data=csv_bytes,
        file_name=f"sample_{pack_id}.csv",
        mime="text/csv",
    )

    if "trace_id" in pack_df.columns:
        trace_ids_txt = "\n".join(pack_df["trace_id"].fillna("").astype(str).tolist()).encode("utf-8")
        st.download_button(
            label="Download trace_ids.txt",
            data=trace_ids_txt,
            file_name=f"sample_{pack_id}_trace_ids.txt",
            mime="text/plain",
        )

    with st.expander("📥 Add this sample pack to a Langfuse annotation queue", expanded=False):
        col_load, col_select = st.columns([1, 2])
        with col_load:
            if st.button("Load queues", key="qa_samples_load_queues"):
                try:
                    headers = get_langfuse_headers(public_key, secret_key)
                    queues = list_annotation_queues(base_url=base_url, headers=headers, page=1, limit=200)
                    st.session_state.qa_samples_queues = queues
                    st.success(f"Loaded {len(queues)} queue(s).")
                except Exception as exc:
                    st.error(f"Failed to load queues: {exc}")

        queue_options = st.session_state.get("qa_samples_queues", [])
        option_map = {
            f"{q.get('name') or '(unnamed)'} ({q.get('id')})": str(q.get("id") or "").strip()
            for q in queue_options
            if str(q.get("id") or "").strip()
        }
        with col_select:
            if option_map:
                selected = st.selectbox("Select queue", options=[""] + list(option_map.keys()))
                if selected:
                    st.session_state.qa_samples_queue_id = option_map[selected]

        st.text_input("Queue ID", key="qa_samples_queue_id")

        if st.button("Add trace IDs to queue", key="qa_samples_push"):
            queue_id = str(st.session_state.get("qa_samples_queue_id") or "").strip()
            if not queue_id:
                st.warning("Queue ID is required.")
            elif "trace_id" not in pack_df.columns:
                st.warning("This sample pack has no trace_id column.")
            else:
                headers = get_langfuse_headers(public_key, secret_key)
                added = 0
                existed = 0
                failed = 0
                failures: list[dict[str, str]] = []
                for trace_id in pack_df["trace_id"].fillna("").astype(str):
                    if not trace_id.strip():
                        continue
                    try:
                        create_annotation_queue_item(
                            base_url=base_url,
                            headers=headers,
                            queue_id=queue_id,
                            object_id=trace_id,
                        )
                        added += 1
                    except requests.HTTPError as exc:
                        status = getattr(getattr(exc, "response", None), "status_code", None)
                        if status == 409:
                            existed += 1
                        else:
                            failed += 1
                            failures.append({"trace_id": trace_id, "error": str(exc)})
                    except Exception as exc:  # noqa: BLE001
                        failed += 1
                        failures.append({"trace_id": trace_id, "error": str(exc)})

                st.success(f"Queue update complete. Added={added}, existed={existed}, failed={failed}")
                if failures:
                    st.code(json.dumps(failures[:20], indent=2), language="json")
