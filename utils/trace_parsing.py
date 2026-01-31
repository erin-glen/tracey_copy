"""Utilities for parsing and classifying Langfuse traces."""

import json
from datetime import datetime
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
    if not isinstance(raw, str) or not raw.strip():
        return None
    s = raw.strip()
    try:
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        return datetime.fromisoformat(s)
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
    for m in out_msgs:
        if isinstance(m, dict) and m.get("type") == "tool":
            return True
    return False


def traces_to_rows(
    traces: list[dict[str, Any]],
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Normalize a list of traces to flat rows with prompt/answer extracted.
    
    This is the canonical way to convert raw Langfuse traces into a list of
    dictionaries suitable for display, CSV export, or further processing.
    """
    rows: list[dict[str, Any]] = []
    for t in traces:
        n = normalize_trace_format(t)
        prompt = first_human_prompt(n)
        answer = final_ai_message(n)
        dt = parse_trace_dt(n)
        rows.append({
            "trace_id": n.get("id"),
            "timestamp": dt,
            "date": dt.date() if dt else None,
            "session_id": n.get("sessionId"),
            "environment": n.get("environment"),
            "user_id": n.get("userId") or (n.get("metadata") or {}).get("user_id") or (n.get("metadata") or {}).get("userId"),
            "latency_seconds": None,  # caller can enrich if needed
            "total_cost": None,
            "prompt": prompt,
            "answer": answer,
        })
        if limit and len(rows) >= limit:
            break
    return rows


def classify_outcome(row: dict[str, Any], answer: str) -> str:
    """Classify the outcome of a trace based on answer content and tool usage."""
    if not answer.strip():
        return "ERROR"
    if looks_like_error_answer(answer):
        return "SOFT_ERROR"
    if not trace_used_tools(row):
        return "DEFER"
    return "ANSWER"
