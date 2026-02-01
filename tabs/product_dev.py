"""Product development mining tab – redesigned with 3 modes."""

from __future__ import annotations

import json
import random
from typing import Any

import streamlit as st

from utils.prompt_fixtures import (
    DEFAULT_EVIDENCE_PROMPT,
    DEFAULT_GAP_ANALYSIS_PROMPT,
    DEFAULT_TAGGING_PROMPT,
)

from utils import (
    normalize_trace_format,
    first_human_prompt,
    final_ai_message,
    csv_bytes_any,
    get_gemini_model_options,
    chunked,
    truncate_text,
    parse_json_any,
    parse_json_dict,
)

# ---------------------------------------------------------------------------
# Tagging criteria definitions
# ---------------------------------------------------------------------------
TAGGING_CRITERIA: dict[str, str] = {
    "prompt_topic": "What is the main topic of the user prompt? (e.g. fundamentals, technicals, macro, company, sector)",
    "prompt_flavour": "What is the style/type of the prompt? (e.g. question, command, clarification, follow-up, comparison, opinion-seeking)",
    "clarity": "Is the user prompt clear and unambiguous? (clear / somewhat_clear / unclear)",
    "complexity": "How complex is the request? (simple_lookup / moderate / multi_step_reasoning)",
    "intent": "What job is the user trying to accomplish? (research / decision_support / learning / monitoring / other)",
    "domain": "What domain area does this fall into? (equity / fixed_income / fx / commodities / macro / mixed / other)",
    "success": "Did the assistant answer the question well? (good / partial / poor / unable)",
    "sentiment": "What is the apparent user sentiment or satisfaction? (positive / neutral / negative / unknown)",
}


def _normalize_rows(traces: list[dict[str, Any]], limit: int = 500) -> list[dict[str, Any]]:
    """Normalize traces to rows with prompt/answer."""
    out: list[dict[str, Any]] = []
    for t in traces:
        n = normalize_trace_format(t)
        prompt = first_human_prompt(n)
        answer = final_ai_message(n)
        if not prompt and not answer:
            continue
        out.append(
            {
                "trace_id": n.get("id"),
                "timestamp": n.get("timestamp"),
                "session_id": n.get("sessionId"),
                "environment": n.get("environment"),
                "prompt": prompt,
                "answer": answer,
            }
        )
        if len(out) >= limit:
            break
    return out


def _call_gemini(api_key: str, model_name: str, prompt: str) -> str:
    """Call Gemini and return text response."""
    import google.generativeai as genai

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
    resp = model.generate_content(prompt)
    return str(getattr(resp, "text", "") or "")


def _render_llm_settings(
    gemini_api_key: str,
    gemini_model_options: list[str],
) -> tuple[str, bool, int, int]:
    """Render LLM settings expander and return (model, use_batching, batch_size, max_chars)."""
    default_model = "gemini-2.5-flash-lite"
    if default_model not in gemini_model_options and gemini_model_options:
        default_model = gemini_model_options[0]

    with st.expander("⚙️ LLM Settings", expanded=False):
        gemini_model = st.selectbox(
            "Gemini model",
            options=gemini_model_options,
            index=gemini_model_options.index(default_model) if default_model in gemini_model_options else 0,
            key="product_dev_gemini_model",
        )
        if not gemini_api_key:
            st.warning("No Gemini API key found. Set GEMINI_API_KEY or GOOGLE_API_KEY in .env")
        else:
            st.success("Gemini API key configured ✓")

        use_batching = st.checkbox(
            "Batch traces per Gemini request",
            value=bool(st.session_state.get("product_dev_use_batching", False)),
            key="product_dev_use_batching",
        )
        st.number_input(
            "Batch size",
            min_value=1,
            max_value=50,
            value=int(st.session_state.get("product_dev_batch_size", 8)),
            key="product_dev_batch_size",
            disabled=not use_batching,
        )
        st.number_input(
            "Max chars per trace (prompt/output)",
            min_value=200,
            max_value=20000,
            value=int(st.session_state.get("product_dev_max_chars_per_trace", 4000)),
            key="product_dev_max_chars_per_trace",
            disabled=not use_batching,
        )

    return (
        gemini_model,
        bool(st.session_state.get("product_dev_use_batching", False)),
        int(st.session_state.get("product_dev_batch_size", 8)),
        int(st.session_state.get("product_dev_max_chars_per_trace", 4000)),
    )


# ---------------------------------------------------------------------------
# Mode: Evidence Mining
# ---------------------------------------------------------------------------
def _render_evidence_mining(
    traces: list[dict[str, Any]],
    base_thread_url: str,
    gemini_api_key: str,
    gemini_model: str,
    use_batching: bool,
    batch_size: int,
    max_chars_per_trace: int,
) -> None:
    st.markdown("### 🔍 Evidence Mining")
    st.caption("Search traces using natural language to find evidence supporting a product hypothesis.")

    hypothesis = st.text_area(
        "Describe what you're looking for",
        placeholder="e.g. Find prompts where users ask about land cover change trends.",
        height=80,
        key="evidence_hypothesis",
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        search_in = st.radio(
            "Search in",
            options=["Prompt only", "Output only", "Both"],
            index=2,
            horizontal=True,
            key="evidence_search_in",
        )
    with col2:
        max_results = st.number_input("Max results", min_value=5, max_value=200, value=50, key="evidence_max")

    with st.expander("📝 Edit system prompt", expanded=False):
        st.caption("Use `{hypothesis}` and `{search_text}` as placeholders.")
        evidence_prompt_template = st.text_area(
            "System prompt",
            value=DEFAULT_EVIDENCE_PROMPT,
            height=200,
            key="evidence_prompt_template",
            label_visibility="collapsed",
        )

    if "evidence_results" not in st.session_state:
        st.session_state.evidence_results = []

    if st.button("🔎 Search with LLM", type="primary", key="evidence_search_btn"):
        if not hypothesis.strip():
            st.warning("Please describe what you're looking for.")
            return
        if not gemini_api_key:
            st.error("Gemini API key required. Set GEMINI_API_KEY or GOOGLE_API_KEY in your .env file.")
            return

        rows = _normalize_rows(traces, limit=500)
        if not rows:
            st.warning("No traces available.")
            return

        results: list[dict[str, Any]] = []
        progress = st.progress(0.0, text="Scoring traces...")

        def _score_single(r: dict[str, Any]) -> dict[str, Any] | None:
            search_text = ""
            if search_in == "Prompt only":
                search_text = r.get("prompt", "")
            elif search_in == "Output only":
                search_text = r.get("answer", "")
            else:
                search_text = f"PROMPT: {r.get('prompt', '')}\n\nOUTPUT: {r.get('answer', '')}"
            search_text = truncate_text(str(search_text or ""), int(max_chars_per_trace))

            scoring_prompt = evidence_prompt_template.format(
                hypothesis=hypothesis,
                search_text=search_text,
            )
            resp_txt = _call_gemini(gemini_api_key, gemini_model, scoring_prompt)
            parsed = parse_json_dict(resp_txt)
            if parsed.get("relevant") or (isinstance(parsed.get("score"), (int, float)) and parsed["score"] >= 50):
                return {
                    **r,
                    "relevance_score": parsed.get("score", 50),
                    "relevance_reason": parsed.get("reason", ""),
                }
            return None

        scanned = 0
        if use_batching and int(batch_size) > 1:
            batches = chunked(rows, int(batch_size))
            for b in batches:
                batch_payload = []
                for r in b:
                    if search_in == "Prompt only":
                        stxt = r.get("prompt", "")
                    elif search_in == "Output only":
                        stxt = r.get("answer", "")
                    else:
                        stxt = f"PROMPT: {r.get('prompt', '')}\n\nOUTPUT: {r.get('answer', '')}"
                    batch_payload.append(
                        {
                            "trace_id": r.get("trace_id"),
                            "search_text": truncate_text(str(stxt or ""), int(max_chars_per_trace)),
                        }
                    )

                batch_prompt = (
                    "You are a relevance scorer. For each item in TRACES, score how relevant it is to the HYPOTHESIS.\n\n"
                    "Return a JSON array with one object per input item, each object having keys: trace_id, relevant (true/false), score (0-100), reason.\n\n"
                    f"HYPOTHESIS: {hypothesis}\n\n"
                    f"TRACES: {json.dumps(batch_payload)}"
                )

                parsed_any = None
                try:
                    resp_txt = _call_gemini(gemini_api_key, gemini_model, batch_prompt)
                    parsed_any = parse_json_any(resp_txt)
                except Exception:
                    parsed_any = None

                if isinstance(parsed_any, list):
                    by_tid: dict[str, dict[str, Any]] = {}
                    for obj in parsed_any:
                        if isinstance(obj, dict) and obj.get("trace_id") is not None:
                            by_tid[str(obj.get("trace_id"))] = obj
                    for r in b:
                        scanned += 1
                        obj = by_tid.get(str(r.get("trace_id") or ""))
                        if isinstance(obj, dict):
                            score = obj.get("score")
                            relevant = obj.get("relevant")
                            if relevant or (isinstance(score, (int, float)) and float(score) >= 50):
                                results.append(
                                    {
                                        **r,
                                        "relevance_score": score if score is not None else 50,
                                        "relevance_reason": obj.get("reason", ""),
                                    }
                                )
                        progress.progress(min(1.0, scanned / len(rows)), text=f"Scored {scanned}/{len(rows)}")
                        if len(results) >= int(max_results):
                            break
                else:
                    for r in b:
                        scanned += 1
                        try:
                            one = _score_single(r)
                            if one:
                                results.append(one)
                        except Exception as e:
                            st.warning(f"Error scoring trace: {e}")
                        progress.progress(min(1.0, scanned / len(rows)), text=f"Scored {scanned}/{len(rows)}")
                        if len(results) >= int(max_results):
                            break

                if len(results) >= int(max_results):
                    break
        else:
            for r in rows:
                scanned += 1
                try:
                    one = _score_single(r)
                    if one:
                        results.append(one)
                except Exception as e:
                    st.warning(f"Error scoring trace: {e}")
                progress.progress(min(1.0, scanned / len(rows)), text=f"Scored {scanned}/{len(rows)}")
                if len(results) >= int(max_results):
                    break

        progress.empty()
        results.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
        st.session_state.evidence_results = results
        st.rerun()

    results = st.session_state.evidence_results
    if results:
        st.success(f"Found **{len(results)}** relevant traces")

        st.markdown("**Results table**")
        table_rows: list[dict[str, Any]] = []
        for r in results:
            table_rows.append(
                {
                    "relevance_score": r.get("relevance_score", ""),
                    "relevance_reason": r.get("relevance_reason", ""),
                    "prompt": r.get("prompt", ""),
                    "answer": r.get("answer", ""),
                    "url": f"{base_thread_url.rstrip('/')}/{r.get('session_id')}" if r.get("session_id") else "",
                }
            )
        st.dataframe(table_rows, width="stretch")

        evidence_csv_bytes = csv_bytes_any(table_rows)

        st.download_button(
            "⬇️ Download results CSV",
            evidence_csv_bytes,
            "evidence_mining_results.csv",
            "text/csv",
            key="evidence_results_csv",
        )

        st.markdown("**Top matches (expand for details)**")
        for r in results[:20]:
            with st.expander(f"🎯 Score: {r.get('relevance_score', '?')} — {r.get('prompt', '')[:80]}..."):
                st.caption(f"**Reason:** {r.get('relevance_reason', 'N/A')}")
                st.markdown("**Prompt:**")
                st.code(r.get("prompt", ""), language=None)
                st.markdown("**Output:**")
                st.code(r.get("answer", "")[:1000] + ("..." if len(r.get("answer", "")) > 1000 else ""), language=None)
                url = f"{base_thread_url.rstrip('/')}/{r.get('session_id')}" if r.get("session_id") else ""
                if url:
                    st.link_button("🔗 View in GNW", url)



# ---------------------------------------------------------------------------
# Mode: Tagging
# ---------------------------------------------------------------------------
def _render_tagging(
    traces: list[dict[str, Any]],
    base_thread_url: str,
    gemini_api_key: str,
    gemini_model: str,
    use_batching: bool,
    batch_size: int,
    max_chars_per_trace: int,
) -> None:
    st.markdown("### 🏷️ Tagging (LLM-as-Judge)")
    st.caption("Batch-tag traces with predefined or custom criteria.")

    st.markdown("**Select tagging criteria:**")
    selected_criteria: list[str] = []
    cols = st.columns(4)
    criteria_keys = list(TAGGING_CRITERIA.keys())
    for i, key in enumerate(criteria_keys):
        with cols[i % 4]:
            if st.checkbox(key.replace("_", " ").title(), value=key in ["prompt_topic", "prompt_flavour", "intent"], key=f"tag_crit_{key}"):
                selected_criteria.append(key)

    custom_criteria = st.text_input(
        "Custom criteria (comma-separated)",
        placeholder="e.g. urgency, data_source_needed",
        key="tag_custom_criteria",
    )
    if custom_criteria.strip():
        for c in custom_criteria.split(","):
            c = c.strip().lower().replace(" ", "_")
            if c and c not in selected_criteria:
                selected_criteria.append(c)

    col1, col2 = st.columns([1, 1])
    with col1:
        sample_size = st.number_input("Sample size", min_value=5, max_value=500, value=50, key="tag_sample_size")
    with col2:
        random_seed = st.number_input("Random seed", min_value=0, max_value=999999, value=42, key="tag_seed")

    with st.expander("📝 Edit system prompt", expanded=False):
        st.caption("Use `{criteria_desc}`, `{user_prompt}`, `{assistant_output}`, `{criteria_keys}` as placeholders.")
        tagging_prompt_template = st.text_area(
            "System prompt",
            value=DEFAULT_TAGGING_PROMPT,
            height=250,
            key="tagging_prompt_template",
            label_visibility="collapsed",
        )

    if "tagging_results" not in st.session_state:
        st.session_state.tagging_results = []

    if st.button("🏷️ Run tagging", type="primary", key="tag_run_btn"):
        if not selected_criteria:
            st.warning("Please select at least one tagging criterion.")
            return
        if not gemini_api_key:
            st.error("Gemini API key required.")
            return

        rows = _normalize_rows(traces, limit=2000)
        if not rows:
            st.warning("No traces available.")
            return

        rng = random.Random(int(random_seed))
        rng.shuffle(rows)
        sample = rows[:int(sample_size)]

        criteria_desc = "\n".join([f"- {k}: {TAGGING_CRITERIA.get(k, 'Provide a short label')}" for k in selected_criteria])

        results: list[dict[str, Any]] = []
        progress = st.progress(0.0, text="Tagging traces...")

        def _tag_single(r: dict[str, Any]) -> dict[str, Any]:
            tagging_prompt = tagging_prompt_template.format(
                criteria_desc=criteria_desc,
                user_prompt=truncate_text(str(r.get("prompt", "") or ""), int(max_chars_per_trace)),
                assistant_output=truncate_text(str(r.get("answer", "") or "")[:2000], int(max_chars_per_trace)),
                criteria_keys=json.dumps(selected_criteria),
            )
            resp_txt = _call_gemini(gemini_api_key, gemini_model, tagging_prompt)
            parsed = parse_json_dict(resp_txt)
            tagged_row = {**r}
            for k in selected_criteria:
                tagged_row[f"tag_{k}"] = parsed.get(k, "")
            return tagged_row

        done = 0
        if use_batching and int(batch_size) > 1:
            batches = chunked(sample, int(batch_size))
            for b in batches:
                payload = []
                for r in b:
                    payload.append(
                        {
                            "trace_id": r.get("trace_id"),
                            "user_prompt": truncate_text(str(r.get("prompt", "") or ""), int(max_chars_per_trace)),
                            "assistant_output": truncate_text(str(r.get("answer", "") or ""), int(max_chars_per_trace)),
                        }
                    )

                batch_prompt = (
                    "You are a trace tagger. For each item in TRACES, apply the tagging criteria and return results.\n\n"
                    "Return a JSON array with one object per input item. Each object MUST include trace_id and keys exactly matching CRITERIA_KEYS.\n\n"
                    f"CRITERIA_DESC:\n{criteria_desc}\n\n"
                    f"CRITERIA_KEYS: {json.dumps(selected_criteria)}\n\n"
                    f"TRACES: {json.dumps(payload)}"
                )

                parsed_any = None
                try:
                    resp_txt = _call_gemini(gemini_api_key, gemini_model, batch_prompt)
                    parsed_any = parse_json_any(resp_txt)
                except Exception:
                    parsed_any = None

                if isinstance(parsed_any, list):
                    by_tid: dict[str, dict[str, Any]] = {}
                    for obj in parsed_any:
                        if isinstance(obj, dict) and obj.get("trace_id") is not None:
                            by_tid[str(obj.get("trace_id"))] = obj

                    for r in b:
                        done += 1
                        obj = by_tid.get(str(r.get("trace_id") or ""))
                        if isinstance(obj, dict):
                            tagged_row = {**r}
                            for k in selected_criteria:
                                tagged_row[f"tag_{k}"] = obj.get(k, "")
                            results.append(tagged_row)
                        else:
                            try:
                                results.append(_tag_single(r))
                            except Exception as e:
                                results.append({**r, "tag_error": str(e)})
                        progress.progress(min(1.0, done / len(sample)), text=f"Tagged {done}/{len(sample)}")
                else:
                    for r in b:
                        done += 1
                        try:
                            results.append(_tag_single(r))
                        except Exception as e:
                            results.append({**r, "tag_error": str(e)})
                        progress.progress(min(1.0, done / len(sample)), text=f"Tagged {done}/{len(sample)}")
        else:
            for r in sample:
                done += 1
                try:
                    results.append(_tag_single(r))
                except Exception as e:
                    results.append({**r, "tag_error": str(e)})
                progress.progress(min(1.0, done / len(sample)), text=f"Tagged {done}/{len(sample)}")

        progress.empty()
        st.session_state.tagging_results = results
        st.rerun()

    results = st.session_state.tagging_results
    if results:
        st.success(f"Tagged **{len(results)}** traces")

        tag_cols = [c for c in results[0].keys() if c.startswith("tag_")]
        display_cols = ["prompt", "answer"] + tag_cols
        display_data = [{k: r.get(k, "") for k in display_cols} for r in results]

        for d in display_data:
            d["prompt"] = (d.get("prompt") or "")[:100] + "..."
            d["answer"] = (d.get("answer") or "")[:100] + "..."

        st.dataframe(display_data, width="stretch")

        csv_bytes = csv_bytes_any([{**r, "url": f"{base_thread_url.rstrip('/')}/{r.get('session_id')}" if r.get("session_id") else ""} for r in results])

        st.download_button("⬇️ Download tagged CSV", csv_bytes, "tagging_results.csv", "text/csv", key="tagging_csv")


# ---------------------------------------------------------------------------
# Mode: Gap Analysis
# ---------------------------------------------------------------------------
def _render_gap_analysis(
    traces: list[dict[str, Any]],
    base_thread_url: str,
    gemini_api_key: str,
    gemini_model: str,
) -> None:
    # Fix scroll issue in tabs by forcing parent overflow to auto
    st.markdown(
        """
        <style>
        [data-testid="stTabsContent"] {
            overflow: visible !important;
            max-height: none !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("### 📊 Gap Analysis")
    st.caption("Analyze patterns across traces to identify user jobs, coverage, success rates, and gaps.")

    sample_size = st.number_input("Sample size for analysis", min_value=20, max_value=500, value=100, key="gap_sample_size")

    with st.expander("📝 Edit system prompt", expanded=False):
        st.caption("Use `{traces_summary}` as a placeholder for the trace data.")
        gap_prompt_template = st.text_area(
            "System prompt",
            value=DEFAULT_GAP_ANALYSIS_PROMPT,
            height=350,
            key="gap_prompt_template",
            label_visibility="collapsed",
        )

    if "gap_report" not in st.session_state:
        st.session_state.gap_report = ""
    if "gap_rows" not in st.session_state:
        st.session_state.gap_rows = []

    if st.button("📊 Generate Gap Report", type="primary", key="gap_run_btn"):
        if not gemini_api_key:
            st.error("Gemini API key required.")
            return

        rows = _normalize_rows(traces, limit=2000)
        if not rows:
            st.warning("No traces available.")
            return

        rng = random.Random(42)
        rng.shuffle(rows)
        sample = rows[:int(sample_size)]

        st.info(f"Analyzing {len(sample)} traces...")

        st.session_state.gap_rows = list(sample)

        traces_summary = ""
        for i, r in enumerate(sample[:50]):
            traces_summary += f"\n---\nTRACE {i+1}:\nPROMPT: {r.get('prompt', '')[:300]}\nOUTPUT: {r.get('answer', '')[:300]}\n"

        analysis_prompt = gap_prompt_template.format(traces_summary=traces_summary)

        try:
            resp_txt = _call_gemini(gemini_api_key, gemini_model, analysis_prompt)
            st.session_state.gap_report = resp_txt
            st.rerun()
        except Exception as e:
            st.error(f"Error generating report: {e}")

    report = st.session_state.gap_report
    if report:
        st.subheader("**Gap report**")
        st.markdown(report)

        gap_rows: list[dict[str, Any]] = st.session_state.get("gap_rows", [])
        if gap_rows:
            st.markdown("**Traces used for this report**")
            gap_table_rows: list[dict[str, Any]] = []
            for r in gap_rows:
                gap_table_rows.append(
                    {
                        **{k: v for k, v in r.items()},
                        "url": f"{base_thread_url.rstrip('/')}/{r.get('session_id')}" if r.get("session_id") else "",
                    }
                )
            st.dataframe(gap_table_rows, width="stretch")

            gap_csv_bytes = csv_bytes_any(gap_table_rows)

            st.download_button(
                "⬇️ Download trace sample (CSV)",
                gap_csv_bytes,
                "gap_analysis_trace_sample.csv",
                "text/csv",
                key="gap_trace_sample_csv",
            )

        st.download_button(
            "⬇️ Download report (Markdown)",
            report.encode("utf-8"),
            "gap_analysis_report.md",
            "text/markdown",
            key="gap_report_md",
        )

        with st.expander("📋 Copy for Slack"):
            st.code(report, language="markdown")


# ---------------------------------------------------------------------------
# Main render
# ---------------------------------------------------------------------------
def render(
    base_thread_url: str,
    gemini_api_key: str,
) -> None:
    """Render the Product Dev Mining tab."""
    st.subheader("⛏️ Product Development Mining")

    traces: list[dict[str, Any]] = st.session_state.get("stats_traces", [])

    if not traces:
        st.info(
            "This tab helps you **mine traces for product insights**:\n\n"
            "- 🔍 **Evidence Mining** – Find traces supporting a product hypothesis\n"
            "- 🏷️ **Tagging** – Batch-tag traces with LLM-as-judge criteria\n"
            "- 📊 **Gap Analysis** – Identify user jobs, coverage, and product gaps\n\n"
            "Use the sidebar **🚀 Fetch traces** button first."
        )
        return

    gemini_model_options = get_gemini_model_options(gemini_api_key) if gemini_api_key else [
        "gemini-2.5-flash-lite",
        "gemini-2.0-flash",
        "gemini-1.5-pro",
        "gemini-1.5-flash",
    ]

    gemini_model, use_batching, batch_size, max_chars_per_trace = _render_llm_settings(
        gemini_api_key,
        gemini_model_options,
    )

    t_evidence, t_tagging, t_gap = st.tabs(["🔍 Evidence Mining", "🏷️ Tagging", "📊 Gap Analysis"])

    with t_evidence:
        _render_evidence_mining(
            traces,
            base_thread_url,
            gemini_api_key,
            gemini_model,
            use_batching,
            batch_size,
            max_chars_per_trace,
        )
    with t_tagging:
        _render_tagging(
            traces,
            base_thread_url,
            gemini_api_key,
            gemini_model,
            use_batching,
            batch_size,
            max_chars_per_trace,
        )
    with t_gap:
        _render_gap_analysis(traces, base_thread_url, gemini_api_key, gemini_model)
