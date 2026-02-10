import json
from typing import Any

import streamlit as st

from utils import (
    normalize_trace_format,
    current_human_prompt,
    slice_output_to_current_turn,
)


def _as_dict(x: Any) -> dict[str, Any]:
    return x if isinstance(x, dict) else {}


def _as_list(x: Any) -> list[Any]:
    return x if isinstance(x, list) else []


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out: list[str] = []
        for c in content:
            if isinstance(c, dict):
                t = c.get("text")
                if isinstance(t, str) and t.strip():
                    out.append(t)
                elif isinstance(c.get("content"), str) and str(c.get("content") or "").strip():
                    out.append(str(c.get("content") or ""))
        return "\n".join(out)
    return ""


def _strip_noise(obj: Any) -> Any:
    if isinstance(obj, dict):
        cleaned: dict[str, Any] = {}
        for k, v in obj.items():
            if k == "__gemini_function_call_thought_signatures__":
                continue
            cleaned[k] = _strip_noise(v)
        return cleaned
    if isinstance(obj, list):
        return [_strip_noise(x) for x in obj]
    return obj


def _first_user_prompt_snippet(t: dict[str, Any], max_len: int = 80) -> str:
    input_msgs = _as_list(_as_dict(t.get("input")).get("messages"))
    for m in input_msgs:
        md = _as_dict(m)
        if str(md.get("type") or "") != "human":
            continue
        text = _content_text(md.get("content")).strip()
        if not text:
            continue
        if len(text) > max_len:
            return text[: max_len - 1] + "…"
        return text
    return ""


def _trace_label(t: dict[str, Any]) -> str:
    snippet = _first_user_prompt_snippet(t)
    if snippet:
        return snippet
    return "(empty prompt)"


def render(base_thread_url: str) -> None:
    st.subheader("🔎 Trace Explorer")
    st.caption(
        "Inspect individual traces in detail: view the current user turn, assistant output, tool calls, metadata, and raw JSON. "
        "Useful for debugging odd outputs, latency/cost spikes, and tool failures."
    )

    traces: list[dict[str, Any]] = st.session_state.get("stats_traces", [])
    if not traces:
        st.info("Use the sidebar **🚀 Fetch traces** button first.")
        return

    col1, col2 = st.columns([2, 1])

    with col1:
        idx = st.selectbox(
            "Select trace",
            options=list(range(len(traces))),
            format_func=lambda i: _trace_label(traces[int(i)]),
            index=0,
            key="trace_explorer_selected_idx",
        )

    with col2:
        hide_empty = st.checkbox("Hide empty messages", value=True, key="trace_explorer_hide_empty")
        show_raw = st.checkbox("Show raw JSON", value=False, key="trace_explorer_show_raw")

    trace = normalize_trace_format(traces[int(idx)])
    trace_clean = _strip_noise(trace)

    tid = str(trace.get("id") or "")
    sid = str(trace.get("sessionId") or "")
    if sid:
        st.link_button("🔗 Open session in GNW", f"{base_thread_url.rstrip('/')}/{sid}")

    st.markdown("### Summary")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Environment", str(trace.get("environment") or ""))
    with c2:
        st.metric("Latency (s)", str(trace.get("latency") or ""))
    with c3:
        st.metric("Total cost", str(trace.get("totalCost") or ""))
    with c4:
        obs = trace.get("observations")
        st.metric("Observations", str(len(obs)) if isinstance(obs, list) else "")

    st.markdown("### Input messages")
    current_prompt = current_human_prompt(trace)
    if not current_prompt:
        st.info("No input.messages")
    else:
        st.code(current_prompt, language=None)

    output_msgs_all = _as_list(_as_dict(trace.get("output")).get("messages"))
    output_msgs = slice_output_to_current_turn(trace, output_msgs_all)

    with st.expander("### Tool calls", expanded=False):
        tool_results_by_call_id: dict[str, dict[str, Any]] = {}
        for m in output_msgs:
            md = _as_dict(m)
            if str(md.get("type") or "") == "tool" and md.get("tool_call_id") is not None:
                tool_results_by_call_id[str(md.get("tool_call_id"))] = md

        tool_calls_found = 0
        for i, m in enumerate(output_msgs):
            md = _as_dict(m)
            if str(md.get("type") or "") != "ai":
                continue
            for tc in _as_list(md.get("tool_calls")):
                tcd = _as_dict(tc)
                call_id = str(tcd.get("id") or "")
                name = str(tcd.get("name") or "")
                args = tcd.get("args")
                tool_calls_found += 1

                result = tool_results_by_call_id.get(call_id)
                status = str(_as_dict(result).get("status") or "") if isinstance(result, dict) else ""
                exp_label = f"{tool_calls_found}. {name}"
                if status:
                    exp_label = f"{exp_label} ({status})"
                with st.expander(exp_label, expanded=False):
                    st.markdown("**Call**")
                    if call_id:
                        st.code(call_id, language=None)
                    st.json(_strip_noise({"name": name, "args": args}))

                    st.markdown("**Result**")
                    if isinstance(result, dict):
                        rtext = _content_text(result.get("content"))
                        if str(rtext or "").strip():
                            st.code(rtext, language=None)
                        st.json(_strip_noise(result))
                    else:
                        st.info("No tool result message found for this tool_call_id")

        if tool_calls_found == 0:
            st.info("No tool calls found in output.messages")

    with st.expander("### Output messages", expanded=False):
        if not output_msgs:
            st.info("No output.messages")
        else:
            for i, m in enumerate(output_msgs):
                md = _as_dict(m)
                mtype = str(md.get("type") or "")
                name = str(md.get("name") or "")
                text = _content_text(md.get("content"))
                if hide_empty and not str(text or "").strip() and mtype not in {"ai", "tool"}:
                    continue
                label = f"{i}: {mtype or 'message'}"
                if name:
                    label = f"{label} ({name})"
                with st.expander(label, expanded=False):
                    if str(text or "").strip():
                        st.code(text, language=None)
                    st.json(_strip_noise(md))

    st.markdown("### Cleaned trace JSON")
    with st.expander("View cleaned JSON (noise removed)", expanded=False):
        st.json(trace_clean)

    if show_raw:
        st.markdown("### Raw trace JSON")
        st.json(trace)

    st.download_button(
        "⬇️ Download trace JSON",
        data=json.dumps(trace, indent=2, ensure_ascii=False).encode("utf-8"),
        file_name=f"trace_{tid or 'unknown'}.json",
        mime="application/json",
        key=f"trace_explorer_dl_{tid or 'unknown'}",
    )
