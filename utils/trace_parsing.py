"""Utilities for parsing and classifying Langfuse traces."""

import re
import json
from datetime import datetime, timezone
from typing import Any


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
    if isinstance(content, list):
        out: list[str] = []
        for c in content:
            if isinstance(c, dict):
                if "text" in c and isinstance(c["text"], str):
                    out.append(c["text"])
                elif "content" in c and isinstance(c["content"], str):
                    out.append(c["content"])
        return "\n".join(out)
    return ""


def first_human_prompt(row: dict[str, Any]) -> str:
    """Extract the first human message from a trace's input."""
    msgs = (((row.get("input") or {}).get("messages")) or [])
    for m in msgs:
        if isinstance(m, dict) and m.get("type") == "human":
            t = _msg_text(m.get("content"))
            if t and t.strip():
                return t.strip()
    return ""


def final_ai_message(row: dict[str, Any]) -> str:
    """Extract the final AI message from a trace's output."""
    msgs = (((row.get("output") or {}).get("messages")) or [])
    for m in reversed(msgs):
        if isinstance(m, dict) and m.get("type") == "ai":
            t = _msg_text(m.get("content"))
            if t and t.strip():
                return t.strip()
    return ""


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
    out_msgs = (((row.get("output") or {}).get("messages")) or [])
    return any(isinstance(m, dict) and m.get("type") == "tool" for m in out_msgs)


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
    """
    aois: list[str] = []
    datasets: list[str] = []
    datasets_analysed: list[str] = []
    tools_used: list[str] = []
    aoi_name = ""
    aoi_type = ""

    out_msgs = ((trace.get("output") or {}).get("messages") or [])
    for m in out_msgs:
        if not isinstance(m, dict):
            continue

        if m.get("type") == "tool" and m.get("name") == "pick_aoi":
            content = str(m.get("content") or "")
            m_aoi = re.search(
                r"Selected AOI:\s*(.*?)(?:,\s*type:\s*(.*))?$", content
            )
            if m_aoi:
                aoi_name = (m_aoi.group(1) or "").strip()
                aoi_type = (m_aoi.group(2) or "").strip()

        for tc in (m.get("tool_calls") or []):
            if not isinstance(tc, dict):
                continue
            name = str(tc.get("name") or "")
            if name and name not in tools_used:
                tools_used.append(name)
            args = tc.get("args") or {}
            if isinstance(args, dict):
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
    }
