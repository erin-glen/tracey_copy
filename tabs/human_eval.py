"""Human evaluation sampling tab."""

import random
import time
from typing import Any

import streamlit as st

from utils import (
    normalize_trace_format,
    first_human_prompt,
    final_ai_message,
    csv_bytes_any,
    init_session_state,
    extract_trace_context,
    get_gemini_model_options,
    truncate_text,
    parse_json_any,
)


ENCOURAGEMENT_MESSAGES = [
    "Great start! 🚀",
    "You're on a roll! 🔥",
    "Keep it up! 💪",
    "Halfway there! 🎯",
    "Almost done! 🏁",
    "Final stretch! ⭐",
]


def _get_encouragement(progress: float) -> str:
    """Return an encouraging message based on progress."""
    if progress < 0.1:
        return ENCOURAGEMENT_MESSAGES[0]
    elif progress < 0.25:
        return ENCOURAGEMENT_MESSAGES[1]
    elif progress < 0.5:
        return ENCOURAGEMENT_MESSAGES[2]
    elif progress < 0.75:
        return ENCOURAGEMENT_MESSAGES[3]
    elif progress < 0.9:
        return ENCOURAGEMENT_MESSAGES[4]
    else:
        return ENCOURAGEMENT_MESSAGES[5]


def _format_elapsed(started_at: float | None) -> str:
    """Format elapsed time as mm:ss."""
    if not started_at:
        return "00:00"
    elapsed = time.time() - started_at
    mins = int(elapsed // 60)
    secs = int(elapsed % 60)
    return f"{mins:02d}:{secs:02d}"


def render(
    base_thread_url: str,
    gemini_api_key: str,
) -> None:
    """Render the Human Eval Sampling tab."""
    st.subheader("🧪 Human Evaluation")

    init_session_state({
        "human_eval_annotations": {},
        "human_eval_samples": [],
        "human_eval_index": 0,
        "human_eval_started_at": None,
        "human_eval_completed": False,
        "human_eval_streak": 0,
        "human_eval_showed_balloons": False,
        "human_eval_evaluator_name": "",
        "human_eval_filter_criteria": "",
        "human_eval_filter_model": "",
        "human_eval_filter_cache": {},
    })

    def _call_gemini(api_key: str, model_name: str, prompt: str) -> str:
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        resp = model.generate_content(prompt)
        return str(getattr(resp, "text", "") or "")

    def _criteria_key(model_name: str, criteria: str) -> str:
        return f"{model_name}||{criteria.strip()}"

    traces: list[dict[str, Any]] = st.session_state.get("stats_traces", [])

    if not traces:
        st.info(
            "This tab helps you create a **random sample** from the currently loaded traces and record quick human eval "
            "annotations (good/bad/unclear + notes).\n\n"
            "Use the sidebar **🚀 Fetch traces** button first, then click **Sample from fetched traces** here."
        )
        return

    samples: list[dict[str, Any]] = st.session_state.human_eval_samples

    if not samples:
        st.markdown("#### Set up your evaluation session")
        evaluator_name = st.text_input(
            "Your name (for CSV filename)",
            value=st.session_state.human_eval_evaluator_name,
            placeholder="e.g. Alice",
            key="_eval_name_input",
        )
        col_size, col_seed = st.columns(2)
        with col_size:
            sample_size = st.number_input("Sample size", min_value=1, max_value=200, value=10)
        with col_seed:
            random_seed = st.number_input("Random seed", min_value=0, max_value=10_000_000, value=0)

        with st.expander(
            "🔎 Optional: Gemini pre-filter (SME topic/dataset)",
            expanded=bool(str(st.session_state.get("human_eval_filter_criteria", "") or "").strip()),
        ):
            criteria = st.text_area(
                "Filter criteria",
                value=str(st.session_state.get("human_eval_filter_criteria", "") or ""),
                height=90,
                key="human_eval_filter_criteria",
                placeholder="e.g. Only prompts about tree cover loss drivers, wildfire, or Indonesia.",
            )

            gemini_model_options = get_gemini_model_options(gemini_api_key) if gemini_api_key else []
            default_model = "gemini-2.5-flash-lite"
            if gemini_model_options and default_model not in gemini_model_options:
                default_model = gemini_model_options[0]

            model_name = st.selectbox(
                "Gemini model",
                options=gemini_model_options if gemini_model_options else [default_model],
                index=(
                    gemini_model_options.index(default_model)
                    if gemini_model_options and default_model in gemini_model_options
                    else 0
                ),
                key="human_eval_filter_model",
                disabled=not bool(gemini_api_key),
            )

            max_to_classify = st.number_input(
                "Max traces to classify",
                min_value=10,
                max_value=5000,
                value=500,
                help="Classification can be expensive; start small and increase as needed.",
                key="human_eval_filter_max_to_classify",
            )

            st.caption(
                "Sampling will automatically use this filter when criteria is set and matching traces have been classified."
            )

            if not gemini_api_key:
                st.warning("No Gemini API key configured. Set GEMINI_API_KEY or GOOGLE_API_KEY.")

            if st.button("Run Gemini filter", type="secondary", disabled=not bool(gemini_api_key)):
                if not criteria.strip():
                    st.warning("Enter filter criteria first.")
                else:
                    cache: dict[str, Any] = st.session_state.get("human_eval_filter_cache", {})
                    if not isinstance(cache, dict):
                        cache = {}

                    key = _criteria_key(str(model_name or ""), str(criteria or ""))
                    progress = st.progress(0.0, text="Classifying prompts...")

                    scanned = 0
                    updated = 0
                    to_consider = traces[: int(max_to_classify)]
                    batch_size = 20
                    pending: list[dict[str, Any]] = []

                    def _flush_pending() -> None:
                        nonlocal scanned, updated, pending
                        if not pending:
                            return

                        items = [
                            {"trace_id": p["trace_id"], "prompt": p["prompt"]}
                            for p in pending
                            if p.get("trace_id") and p.get("prompt")
                        ]
                        if not items:
                            pending = []
                            return

                        llm_prompt = (
                            "You are a strict classifier for routing user prompts to domain experts.\n"
                            "Given the criteria and each user prompt, decide if the prompt matches the criteria.\n\n"
                            "Return ONLY valid JSON as an array of objects. Each object MUST have keys: "
                            "trace_id (string), match (boolean), reason (string).\n\n"
                            f"CRITERIA:\n{criteria.strip()}\n\n"
                            "ITEMS (JSON):\n"
                            f"{json.dumps(items, ensure_ascii=False)}\n"
                        )

                        resp_txt = _call_gemini(gemini_api_key, str(model_name), llm_prompt)
                        parsed_any = parse_json_any(resp_txt)
                        if not isinstance(parsed_any, list):
                            parsed_any = []

                        by_tid: dict[str, dict[str, Any]] = {}
                        for row in parsed_any:
                            if not isinstance(row, dict):
                                continue
                            tid = str(row.get("trace_id") or "").strip()
                            if not tid:
                                continue
                            by_tid[tid] = row

                        for p in pending:
                            tid = str(p.get("trace_id") or "")
                            out = by_tid.get(tid)
                            if isinstance(out, dict):
                                cache[tid] = {
                                    "criteria_key": key,
                                    "match": bool(out.get("match")),
                                    "reason": str(out.get("reason") or ""),
                                }
                            else:
                                cache[tid] = {
                                    "criteria_key": key,
                                    "match": False,
                                    "reason": "parse_error",
                                }
                            updated += 1

                        pending = []

                    import json

                    for t in to_consider:
                        n = normalize_trace_format(t)
                        tid = str(n.get("id") or "")
                        if not tid:
                            continue

                        existing = cache.get(tid)
                        if isinstance(existing, dict) and existing.get("criteria_key") == key:
                            scanned += 1
                            if scanned % 25 == 0:
                                progress.progress(min(1.0, scanned / max(1, int(max_to_classify))))
                            continue

                        prompt_txt = first_human_prompt(n)
                        if not prompt_txt.strip():
                            cache[tid] = {
                                "criteria_key": key,
                                "match": False,
                                "reason": "empty_prompt",
                            }
                            scanned += 1
                            updated += 1
                        else:
                            pending.append(
                                {
                                    "trace_id": tid,
                                    "prompt": truncate_text(prompt_txt, 1800),
                                }
                            )
                            scanned += 1

                        if len(pending) >= batch_size:
                            _flush_pending()

                        if scanned % 10 == 0:
                            progress.progress(min(1.0, scanned / max(1, int(max_to_classify))))

                    _flush_pending()

                    st.session_state.human_eval_filter_cache = cache
                    st.success(f"Classified {updated:,} prompts (scanned {scanned:,}).")
                    progress.empty()

            cache = st.session_state.get("human_eval_filter_cache", {})
            key = _criteria_key(str(st.session_state.get("human_eval_filter_model") or ""), str(criteria or ""))
            evaluated = 0
            matches = 0
            matched_rows: list[dict[str, Any]] = []
            if isinstance(cache, dict) and key and criteria.strip():
                for t in traces:
                    n = normalize_trace_format(t)
                    tid = str(n.get("id") or "")
                    if not tid:
                        continue
                    entry = cache.get(tid)
                    if isinstance(entry, dict) and entry.get("criteria_key") == key:
                        evaluated += 1
                        if entry.get("match") is True:
                            matches += 1
                            prompt_txt = first_human_prompt(n)
                            matched_rows.append(
                                {
                                    "trace_id": tid,
                                    "session_id": str(n.get("sessionId") or ""),
                                    "prompt": truncate_text(str(prompt_txt or ""), 220),
                                    "reason": truncate_text(str(entry.get("reason") or ""), 220),
                                }
                            )

                st.caption(
                    f"Gemini filter coverage: {evaluated:,}/{len(traces):,} classified. Matches: {matches:,}."
                )

                preview_df = matched_rows[:5]
                if preview_df:
                    st.dataframe(preview_df, hide_index=True, use_container_width=True)
                else:
                    st.info("No matched examples to preview yet.")

        if st.button("🎲 Sample from fetched traces", type="primary"):
            st.session_state.human_eval_evaluator_name = evaluator_name.strip()
            normed: list[dict[str, Any]] = []

            criteria = str(st.session_state.get("human_eval_filter_criteria", "") or "")
            model_name = str(st.session_state.get("human_eval_filter_model", "") or "")
            filter_active = bool(criteria.strip())
            criteria_key = _criteria_key(model_name, criteria) if filter_active else ""
            cache = st.session_state.get("human_eval_filter_cache", {})

            if filter_active:
                if not isinstance(cache, dict) or not any(
                    isinstance(v, dict)
                    and v.get("criteria_key") == criteria_key
                    and v.get("match") is True
                    for v in cache.values()
                ):
                    st.warning(
                        "Filter criteria is set, but there are no matching classified traces yet. "
                        "Run the Gemini filter (or increase 'Max traces to classify') before sampling."
                    )
                    return

            for t in traces:
                n = normalize_trace_format(t)
                prompt = first_human_prompt(n)
                answer = final_ai_message(n)
                if not prompt or not answer:
                    continue

                if filter_active and criteria.strip():
                    tid = str(n.get("id") or "")
                    entry = cache.get(tid) if isinstance(cache, dict) and tid else None
                    if not (isinstance(entry, dict) and entry.get("criteria_key") == criteria_key and entry.get("match") is True):
                        continue

                helper_info = extract_trace_context(n)
                normed.append(
                    {
                        "trace_id": n.get("id"),
                        "timestamp": n.get("timestamp"),
                        "session_id": n.get("sessionId"),
                        "environment": n.get("environment"),
                        "prompt": prompt,
                        "answer": answer,
                        "aois": helper_info.get("aois", []),
                        "datasets": helper_info.get("datasets", []),
                        "datasets_analysed": helper_info.get("datasets_analysed", []),
                        "tools_used": helper_info.get("tools_used", []),
                        "aoi_name": helper_info.get("aoi_name", ""),
                        "aoi_type": helper_info.get("aoi_type", ""),
                        "filter_match": True if filter_active and criteria.strip() else False,
                    }
                )

            rng = random.Random(int(random_seed))
            rng.shuffle(normed)
            st.session_state.human_eval_samples = normed[: int(sample_size)]
            st.session_state.human_eval_index = 0
            st.session_state.human_eval_annotations = {}
            st.session_state.human_eval_started_at = time.time()
            st.session_state.human_eval_completed = False
            st.session_state.human_eval_streak = 0
            st.session_state.human_eval_showed_balloons = False
            st.rerun()

        st.info("Click the button above to create a shuffled sample from the shared traces.")
        return

    trace_ids = [str(r.get("trace_id") or "") for r in samples if r.get("trace_id")]
    evaluated_ids = {
        str(tid)
        for tid, ann in st.session_state.human_eval_annotations.items()
        if tid in trace_ids and ann and str(ann.get("rating") or "").strip()
    }
    evaluated_count = len(evaluated_ids)
    total_count = len(trace_ids)
    progress = (evaluated_count / total_count) if total_count else 0.0

    eval_rows = []
    for r in samples:
        tid = str(r.get("trace_id") or "")
        ann = st.session_state.human_eval_annotations.get(tid, {})
        status = "evaluated" if tid in evaluated_ids else "not_evaluated"
        eval_rows.append(
            {
                "status": status,
                "trace_id": tid,
                "timestamp": str(r.get("timestamp") or ""),
                "session_id": str(r.get("session_id") or ""),
                "environment": str(r.get("environment") or ""),
                "rating": str(ann.get("rating") or ""),
                "notes": str(ann.get("notes") or ""),
                "url": f"{base_thread_url.rstrip('/')}/{r.get('session_id')}" if r.get("session_id") else "",
                "prompt": str(r.get("prompt") or ""),
                "answer": str(r.get("answer") or ""),
            }
        )

    eval_csv_bytes = csv_bytes_any(eval_rows)
    evaluator_name = st.session_state.human_eval_evaluator_name
    csv_filename = f"human_eval_{evaluator_name}.csv" if evaluator_name else "human_eval_evaluations.csv"

    if not st.session_state.human_eval_completed and total_count and evaluated_count == total_count:
        st.session_state.human_eval_completed = True

    idx = int(st.session_state.human_eval_index)
    idx = max(0, min(idx, len(samples) - 1))
    st.session_state.human_eval_index = idx
    row = samples[idx]

    if st.session_state.human_eval_completed:
        if not st.session_state.human_eval_showed_balloons:
            st.balloons()
            st.session_state.human_eval_showed_balloons = True

        started_at = st.session_state.human_eval_started_at
        elapsed_s = (time.time() - float(started_at)) if started_at else 0.0
        elapsed_m = int(elapsed_s // 60)
        elapsed_r = int(elapsed_s % 60)
        avg_secs = elapsed_s / evaluated_count if evaluated_count else 0

        st.success("## 🎉 All done!")
        st.markdown(
            f"""
**Congratulations!** You completed **{evaluated_count}** evaluations in **{elapsed_m}m {elapsed_r}s**.

| Stat | Value |
|------|-------|
| ⏱️ Total time | {elapsed_m}m {elapsed_r}s |
| 📊 Evaluations | {evaluated_count} |
| ⚡ Avg per eval | {avg_secs:.1f}s |

Thank you for your contribution! 🙏
"""
        )

        st.download_button(
            label="⬇️ Download evaluation CSV",
            data=eval_csv_bytes,
            file_name=csv_filename,
            mime="text/csv",
            key="human_eval_evaluations_csv",
            type="primary",
        )

        if st.button("🔄 Start new session"):
            st.session_state.human_eval_samples = []
            st.session_state.human_eval_annotations = {}
            st.session_state.human_eval_completed = False
            st.session_state.human_eval_showed_balloons = False
            st.rerun()
        return

    prompt_text = str(row.get("prompt", "") or "")
    answer_text = str(row.get("answer", "") or "")
    existing = st.session_state.human_eval_annotations.get(str(row.get("trace_id") or ""), {})
    current_rating = existing.get("rating", "")
    url = f"{base_thread_url.rstrip('/')}/{row.get('session_id')}" if row.get("session_id") else ""

    def _render_content():
        """Render prompt and output content."""
        st.markdown("**`(つ ⊙_⊙)つ` User Prompt**")
        st.code(prompt_text, language=None, wrap_lines=True)

        st.markdown("**`|> °-°|>` Zeno Output**")
        st.code(answer_text, language=None, wrap_lines=True)

        # Helper info expander (collapsed by default)
        aois = row.get("aois") or []
        datasets = row.get("datasets") or []
        datasets_analysed = row.get("datasets_analysed") or []
        tools_used = row.get("tools_used") or []
        aoi_name = str(row.get("aoi_name") or "")
        aoi_type = str(row.get("aoi_type") or "")
        has_helper_info = aois or datasets or datasets_analysed or tools_used or aoi_name or aoi_type
        if has_helper_info:
            with st.expander("📋 Context Helper", expanded=False):
                cols = st.columns(4)
                with cols[0]:
                    st.caption("**AOIs Considered**")
                    st.write(", ".join(aois) if aois else "—")
                with cols[1]:
                    st.caption("**Datasets Considered**")
                    st.write(", ".join(datasets) if datasets else "—")
                with cols[2]:
                    st.caption("**Datasets analysed**")
                    st.write(", ".join(datasets_analysed) if datasets_analysed else "—")
                with cols[3]:
                    st.caption("**Tools Selected**")
                    st.write(", ".join(tools_used) if tools_used else "—")

                if aoi_name or aoi_type:
                    st.caption("**AOI selected**")
                    st.write(f"{aoi_name} ({aoi_type})".strip() if aoi_name or aoi_type else "—")

    col_content, col_controls = st.columns([13, 8], gap="small")

    with col_content:
        _render_content()

    with col_controls:
        st.progress(progress)
        stat_c1, stat_c2, stat_c3 = st.columns(3)
        with stat_c1:
            st.metric("Done", f"{evaluated_count}/{total_count}")
        with stat_c2:
            st.metric("⏱️", _format_elapsed(st.session_state.human_eval_started_at))
        with stat_c3:
            streak = st.session_state.human_eval_streak
            st.metric("🔥", streak if streak > 0 else "-")

        st.caption(_get_encouragement(progress))

        st.markdown("**Rate this response**")
        r1, r2, r3 = st.columns(3)
        with r1:
            good_style = "primary" if current_rating == "good" else "secondary"
            if st.button("👍 Good", key="btn_good", type=good_style, use_container_width=True):
                _save_and_advance(row, "good", st.session_state.get("_eval_notes", ""), idx, samples)
        with r2:
            bad_style = "primary" if current_rating == "bad" else "secondary"
            if st.button("👎 Bad", key="btn_bad", type=bad_style, use_container_width=True):
                _save_and_advance(row, "bad", st.session_state.get("_eval_notes", ""), idx, samples)
        with r3:
            unclear_style = "primary" if current_rating == "unclear" else "secondary"
            if st.button("🤔 Unclear", key="btn_unclear", type=unclear_style, use_container_width=True):
                _save_and_advance(row, "unclear", st.session_state.get("_eval_notes", ""), idx, samples)

        notes = st.text_area(
            "📝 Notes (optional)",
            value=existing.get("notes", ""),
            height=120,
            key="_eval_notes",
            placeholder="Add notes about the your rating...",
        )

        nav_c1, nav_c2, nav_c3 = st.columns([1, 1, 2])
        with nav_c1:
            if st.button("⬅️ Prev", disabled=(idx <= 0), use_container_width=True):
                st.session_state.human_eval_index = idx - 1
                st.rerun()
        with nav_c2:
            if url:
                st.link_button("🔗 GNW", url, use_container_width=True)
            else:
                st.button("🔗", disabled=True, use_container_width=True)
        with nav_c3:
            if st.button("Save & Continue ➡️", type="primary", use_container_width=True):
                _save_and_advance(row, current_rating or "unclear", notes, idx, samples)

        st.download_button(
            label="⬇️ Download CSV",
            data=eval_csv_bytes,
            file_name=csv_filename,
            mime="text/csv",
            key="human_eval_evaluations_csv",
            use_container_width=True,
        )

        if st.button("💾 Save & Finish", use_container_width=True):
            st.session_state.human_eval_completed = True
            st.download_button(
                label="⬇️ Downloading...",
                data=eval_csv_bytes,
                file_name=csv_filename,
                mime="text/csv",
                key="human_eval_finish_csv",
            )
            st.session_state.human_eval_samples = []
            st.session_state.human_eval_annotations = {}
            st.session_state.human_eval_index = 0
            st.session_state.human_eval_started_at = None
            st.session_state.human_eval_completed = False
            st.session_state.human_eval_streak = 0
            st.session_state.human_eval_showed_balloons = False
            st.toast("Session saved! ✅")
            st.rerun()

        if st.button("🔄 Restart", use_container_width=True):
            st.session_state.human_eval_samples = []
            st.session_state.human_eval_annotations = {}
            st.session_state.human_eval_index = 0
            st.session_state.human_eval_started_at = None
            st.session_state.human_eval_completed = False
            st.session_state.human_eval_streak = 0
            st.session_state.human_eval_showed_balloons = False
            st.rerun()


def _save_and_advance(row: dict, rating: str, notes: str, idx: int, samples: list) -> None:
    """Save annotation and advance to next item."""
    tid = str(row.get("trace_id") or "")
    existing_notes = st.session_state.human_eval_annotations.get(tid, {}).get("notes", "")
    st.session_state.human_eval_annotations[tid] = {
        "trace_id": row.get("trace_id"),
        "timestamp": row.get("timestamp"),
        "session_id": row.get("session_id"),
        "environment": row.get("environment"),
        "rating": rating,
        "notes": notes or existing_notes,
        "prompt": row.get("prompt"),
        "answer": row.get("answer"),
    }

    st.session_state.human_eval_streak += 1

    if idx < len(samples) - 1:
        st.session_state.human_eval_index = idx + 1

    st.toast(f"Rated: {rating} ✓")
    st.rerun()
