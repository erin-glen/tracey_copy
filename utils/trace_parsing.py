"""Utilities for parsing and classifying Langfuse traces."""

import re
import json
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse


def normalize_trace_format(row: dict[str, Any]) -> dict[str, Any]:
    """Normalize a trace row by parsing JSON fields."""
    def parse_json_field(value: Any) -> Any:
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
                return parse_json_field(parsed) if isinstance(parsed, str) else parsed
            except (json.JSONDecodeError, TypeError):
                return value
        return value

    normalized = dict(row)
    for field in ["input", "output", "metadata"]:
        if field in normalized:
            normalized[field] = parse_json_field(normalized[field])
    return normalized


def parse_trace_dt(row: dict[str, Any]) -> datetime | None:
    """Parse timestamp from a trace row."""
    raw = row.get("timestamp") or row.get("createdAt") or row.get("created_at")
    if isinstance(raw, datetime):
        return raw
    if isinstance(raw, (int, float)):
        try:
            ts = float(raw)
            if ts > 10_000_000_000:
                ts = ts / 1000.0
            return datetime.fromtimestamp(ts, tz=timezone.utc)
        except Exception:
            return None
    if not isinstance(raw, str) or not raw.strip():
        return None
    s = raw.strip()
    try:
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        try:
            return datetime.fromisoformat(s)
        except ValueError:
            s2 = s.replace(" ", "T")
            if s2.endswith("Z"):
                s2 = s2[:-1] + "+00:00"
            return datetime.fromisoformat(s2)
    except Exception:
        return None


def _msg_text(content: Any) -> str:
    """Extract text from message content (string or list of content blocks)."""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        for key in ["text", "content", "value"]:
            v = content.get(key)
            if isinstance(v, str):
                return v
        return ""
    if isinstance(content, list):
        out: list[str] = []
        for c in content:
            if isinstance(c, str):
                if c.strip():
                    out.append(c)
                continue
            if isinstance(c, dict):
                for key in ["text", "content", "value"]:
                    v = c.get(key)
                    if isinstance(v, str):
                        out.append(v)
                        break
        return "\n".join(out)
    return ""


def _msg_role(msg: Any) -> str:
    if not isinstance(msg, dict):
        return ""
    return str(msg.get("type") or msg.get("role") or "").strip().lower()


def _is_user_msg(msg: Any) -> bool:
    return _msg_role(msg) in {"human", "user"}


def _is_ai_msg(msg: Any) -> bool:
    return _msg_role(msg) in {"ai", "assistant"}


def _is_tool_msg(msg: Any) -> bool:
    return _msg_role(msg) in {"tool", "function"}


def first_human_prompt(row: dict[str, Any]) -> str:
    """Extract the first human message from a trace's input."""
    msgs = (((row.get("input") or {}).get("messages")) or [])
    for m in msgs:
        if _is_user_msg(m):
            t = _msg_text(m.get("content"))
            if t and t.strip():
                return t.strip()
    return ""


def final_ai_message(row: dict[str, Any]) -> str:
    """Extract the final AI message from a trace's output."""
    msgs = (((row.get("output") or {}).get("messages")) or [])
    for m in reversed(msgs):
        if _is_ai_msg(m):
            t = _msg_text(m.get("content"))
            if t and t.strip():
                return t.strip()
    return ""


def current_human_prompt(row: dict[str, Any]) -> str:
    """Extract the current (last) human/user message from a trace input."""
    msgs = (((row.get("input") or {}).get("messages")) or [])
    for m in reversed(msgs):
        if _is_user_msg(m):
            t = _msg_text(m.get("content"))
            if t and t.strip():
                return t.strip()
    return ""


def slice_output_to_current_turn(
    trace: dict[str, Any], output_msgs: list[Any] | None = None
) -> list[Any]:
    """Slice output.messages to the current turn based on current input prompt."""
    msgs = output_msgs
    if msgs is None:
        msgs = ((trace.get("output") or {}).get("messages") or [])
    if not isinstance(msgs, list):
        return msgs

    cur_prompt = current_human_prompt(trace)
    if not cur_prompt:
        return msgs

    start_idx: int | None = None
    for i, m in enumerate(msgs):
        if not _is_user_msg(m):
            continue
        text = _msg_text(m.get("content")).strip()
        if text == cur_prompt.strip():
            start_idx = i

    if start_idx is None:
        return msgs
    return msgs[start_idx:]


def current_turn_ai_message(row: dict[str, Any]) -> str:
    """Extract assistant response scoped to the current turn only."""
    out_msgs = slice_output_to_current_turn(row)
    if isinstance(out_msgs, list):
        for m in reversed(out_msgs):
            if _is_ai_msg(m):
                t = _msg_text(m.get("content"))
                if t and t.strip():
                    return t.strip()

    output = row.get("output")
    if isinstance(output, dict):
        for k in ["response", "answer", "final", "text"]:
            v = output.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
    return ""


def current_turn_prompt_answer(row: dict[str, Any]) -> tuple[str, str]:
    """Get current prompt/answer pair for a trace."""
    return current_human_prompt(row), current_turn_ai_message(row)


def looks_like_error_answer(text: str) -> bool:
    """Check if a response text looks like an error message."""
    t = (text or "").strip().lower()
    if not t:
        return True
    needles = [
        "error",
        "exception",
        "traceback",
        "something went wrong",
        "i can't",
        "i cannot",
        "unable to",
        "sorry",
        "failed to",
    ]
    return any(n in t for n in needles)


def trace_used_tools(row: dict[str, Any]) -> bool:
    """Check if a trace used any tools."""
    obs = row.get("observations")
    if isinstance(obs, list) and obs:
        return True
    out_msgs = slice_output_to_current_turn(row)
    return any(_is_tool_msg(m) for m in out_msgs)


def classify_outcome(row: dict[str, Any], answer: str) -> str:
    """Classify the outcome of a trace based on answer content and tool usage."""
    if not answer.strip():
        return "ERROR"
    if looks_like_error_answer(answer):
        return "SOFT_ERROR"
    if not trace_used_tools(row):
        return "DEFER"
    return "ANSWER"


def extract_trace_context(trace: dict[str, Any]) -> dict[str, Any]:
    """Extract AOIs, datasets, and tool names from a trace for context/eval.

    Returns a dict with keys:
        - aoi_name: str - The selected AOI name (from pick_aoi tool)
        - aoi_type: str - The selected AOI type
        - aois: list[str] - AOIs mentioned in tool call args
        - datasets: list[str] - Datasets mentioned in tool call args
        - datasets_analysed: list[str] - Datasets found in API URLs
        - tools_used: list[str] - Names of tools called
        - pull_data_calls: list[dict[str, Any]] - pull_data endpoint + params per call
        - chart_insight_text: str - Raw text output from generate_insight tool (if present)
    """
    aois: list[str] = []
    datasets: list[str] = []
    datasets_analysed: list[str] = []
    tools_used: list[str] = []
    pull_data_calls: list[dict[str, Any]] = []
    chart_insight_text = ""
    aoi_name = ""
    aoi_type = ""

    out_msgs = slice_output_to_current_turn(trace)
    for m in out_msgs:
        if not isinstance(m, dict):
            continue

        if _is_tool_msg(m) and m.get("name") == "pick_aoi":
            content = str(m.get("content") or "")
            m_aoi = re.search(
                r"Selected AOI:\s*(.*?)(?:,\s*type:\s*(.*))?$", content
            )
            if m_aoi:
                aoi_name = (m_aoi.group(1) or "").strip()
                aoi_type = (m_aoi.group(2) or "").strip()

        if (
            not chart_insight_text
            and _is_tool_msg(m)
            and str(m.get("name") or "") in {"generate_insight", "generate_insights"}
        ):
            chart_insight_text = str(m.get("content") or "").strip()

        if not chart_insight_text and _is_ai_msg(m):
            for tc in (m.get("tool_calls") or []):
                if not isinstance(tc, dict):
                    continue
                if str(tc.get("name") or "") not in {"generate_insight", "generate_insights"}:
                    continue
                args = tc.get("args") or {}
                if not isinstance(args, dict):
                    continue
                for k in ["insight", "text", "content", "summary", "analysis"]:
                    v = args.get(k)
                    if isinstance(v, str) and v.strip():
                        chart_insight_text = v.strip()
                        break
                if chart_insight_text:
                    break

        for tc in (m.get("tool_calls") or []):
            if not isinstance(tc, dict):
                continue
            name = str(tc.get("name") or "")
            if name and name not in tools_used:
                tools_used.append(name)
            args = tc.get("args") or {}
            if isinstance(args, dict):
                if name == "pull_data":
                    raw_url = args.get("url") or args.get("endpoint") or args.get("path")
                    endpoint = ""
                    if isinstance(raw_url, str) and raw_url.strip():
                        try:
                            parsed = urlparse(raw_url)
                            endpoint = parsed.path.lstrip("/")
                        except Exception:
                            endpoint = raw_url.strip().lstrip("/")

                    start_date = args.get("start_date")
                    end_date = args.get("end_date")
                    aoi_names = args.get("aoi_names") or args.get("aoi_name")
                    dataset_name = args.get("dataset_name") or args.get("dataset")
                    query = args.get("query")

                    raw_params: Any = args.get("params")
                    if raw_params is None:
                        raw_params = args.get("parameters")
                    if raw_params is None:
                        raw_params = args.get("query_params")
                    if raw_params is None:
                        raw_params = args.get("request_params")

                    params: Any = raw_params
                    if isinstance(raw_params, str):
                        try:
                            parsed_params = json.loads(raw_params)
                            params = parsed_params
                        except Exception:
                            params = raw_params

                    pull_data_calls.append(
                        {
                            "endpoint": endpoint,
                            "start_date": start_date,
                            "end_date": end_date,
                            "aoi_names": aoi_names,
                            "dataset_name": dataset_name,
                            "query": query,
                            "params": params,
                        }
                    )

                for k in ["aoi", "aoi_name", "aoi_id", "area_of_interest"]:
                    v = args.get(k)
                    if v and str(v).strip() and str(v).strip() not in aois:
                        aois.append(str(v).strip())
                for k in ["dataset", "dataset_name", "dataset_id", "layer", "layer_name"]:
                    v = args.get(k)
                    if v and str(v).strip() and str(v).strip() not in datasets:
                        datasets.append(str(v).strip())

    try:
        output_str = str(trace.get("output") or "")
        hits = re.findall(r"/land_change/([^/]+)/", output_str)
        for h in hits:
            h = str(h).strip()
            if h and h not in datasets_analysed:
                datasets_analysed.append(h)
    except Exception:
        pass

    return {
        "aoi_name": aoi_name,
        "aoi_type": aoi_type,
        "aois": aois,
        "datasets": datasets,
        "datasets_analysed": datasets_analysed,
        "tools_used": tools_used,
        "pull_data_calls": pull_data_calls,
        "chart_insight_text": chart_insight_text,
    }


# ---------------------------------------------------------------------------
# Tool call extraction utilities (for agentic flow analysis)
# ---------------------------------------------------------------------------

_AMBIGUITY_PATTERNS = [
    "multiple locations named",
    "which one you meant",
    "please tell me which",
    "did you mean",
    "could you clarify",
    "which location are you looking for",
]


def extract_tool_calls_and_results(trace: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract all tool calls and their results from a trace.

    Returns a list of dicts with keys:
        - tool_name: str
        - tool_call_id: str
        - status: str ("success", "error", or "pending" if no result found)
        - has_ambiguity: bool (True if tool result asks for clarification)
        - is_semantic_error: bool (True if result content contains error patterns)
        - input_tokens: int (from usage_metadata on calling AI message)
        - output_tokens: int
        - reasoning_tokens: int
    """
    out_msgs = slice_output_to_current_turn(trace)

    # Build lookup of tool results by tool_call_id
    tool_results: dict[str, dict[str, Any]] = {}
    for m in out_msgs:
        if not isinstance(m, dict):
            continue
        if _is_tool_msg(m) and m.get("tool_call_id"):
            tool_results[str(m.get("tool_call_id"))] = m

    calls: list[dict[str, Any]] = []
    for m in out_msgs:
        if not isinstance(m, dict) or not _is_ai_msg(m):
            continue

        # Extract usage metadata from this AI message
        usage = m.get("usage_metadata") or {}
        input_tokens = int(usage.get("input_tokens") or 0)
        output_tokens = int(usage.get("output_tokens") or 0)
        output_details = usage.get("output_token_details") or {}
        reasoning_tokens = int(output_details.get("reasoning") or 0)

        for tc in (m.get("tool_calls") or []):
            if not isinstance(tc, dict):
                continue
            tool_name = str(tc.get("name") or "")
            tool_call_id = str(tc.get("id") or "")

            result = tool_results.get(tool_call_id)
            status = "pending"
            has_ambiguity = False
            is_semantic_error = False

            if result:
                status = str(result.get("status") or "success")
                content = str(result.get("content") or "").lower()

                # Check for ambiguity patterns
                for pat in _AMBIGUITY_PATTERNS:
                    if pat in content:
                        has_ambiguity = True
                        break

                # Check for semantic errors (API returned error in content)
                error_patterns = [
                    "error",
                    "exception",
                    "traceback",
                    "failed",
                    "missing 'status' key",
                    "'detail':",
                    "validation error",
                ]
                for pat in error_patterns:
                    if pat in content:
                        is_semantic_error = True
                        break

            calls.append({
                "tool_name": tool_name,
                "tool_call_id": tool_call_id,
                "status": status,
                "has_ambiguity": has_ambiguity,
                "is_semantic_error": is_semantic_error,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "reasoning_tokens": reasoning_tokens,
            })

    return calls


def extract_tool_flow(trace: dict[str, Any]) -> list[tuple[str, str]]:
    """Extract tool call sequence as list of (tool_name, status) tuples.

    Status is one of: "success", "error", "ambiguity", "semantic_error".
    """
    calls = extract_tool_calls_and_results(trace)
    flow: list[tuple[str, str]] = []
    for c in calls:
        if c["has_ambiguity"]:
            status = "ambiguity"
        elif c["is_semantic_error"]:
            status = "semantic_error"
        elif c["status"] == "error":
            status = "error"
        else:
            status = "success"
        flow.append((c["tool_name"], status))
    return flow


def extract_usage_metadata(trace: dict[str, Any]) -> dict[str, Any]:
    """Aggregate usage metadata across all AI messages in a trace.

    Returns dict with keys:
        - total_input_tokens: int
        - total_output_tokens: int
        - total_reasoning_tokens: int
        - reasoning_ratio: float (reasoning / output, or 0 if no output)
        - tool_call_count: int
    """
    out_msgs = slice_output_to_current_turn(trace)

    total_input = 0
    total_output = 0
    total_reasoning = 0
    tool_call_count = 0

    for m in out_msgs:
        if not isinstance(m, dict) or not _is_ai_msg(m):
            continue

        usage = m.get("usage_metadata") or {}
        total_input += int(usage.get("input_tokens") or 0)
        total_output += int(usage.get("output_tokens") or 0)
        output_details = usage.get("output_token_details") or {}
        total_reasoning += int(output_details.get("reasoning") or 0)

        tool_call_count += len(m.get("tool_calls") or [])

    reasoning_ratio = total_reasoning / total_output if total_output > 0 else 0.0

    return {
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "total_reasoning_tokens": total_reasoning,
        "reasoning_ratio": reasoning_ratio,
        "tool_call_count": tool_call_count,
    }


def trace_has_internal_error(trace: dict[str, Any]) -> bool:
    """Check if any tool call in trace had a semantic/API error."""
    calls = extract_tool_calls_and_results(trace)
    return any(c["is_semantic_error"] or c["status"] == "error" for c in calls)
