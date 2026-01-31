"""General data processing and formatting utilities."""

import csv
import io
import json
from datetime import datetime, timezone
from typing import Any


def maybe_load_dotenv() -> None:
    """Attempt to load environment variables from .env file."""
    try:
        from dotenv import load_dotenv
        load_dotenv(override=False)
    except Exception:
        return


def iso_utc(dt: datetime) -> str:
    """Convert a datetime to ISO format in UTC."""
    return dt.astimezone(timezone.utc).isoformat()


def as_float(x: Any) -> float | None:
    """Safely convert a value to float."""
    try:
        return float(x) if x is not None else None
    except Exception:
        return None


def strip_code_fences(text: str) -> str:
    """Remove markdown code fences from text."""
    t = (text or "").strip()
    if not t.startswith("```"):
        return t
    lines = [ln for ln in t.splitlines() if ln.strip()]
    if len(lines) >= 2 and lines[0].startswith("```") and lines[-1].startswith("```"):
        return "\n".join(lines[1:-1]).strip()
    return t


def safe_json_loads(text: str) -> dict[str, Any]:
    """Safely parse JSON, handling code fences and returning a dict."""
    t = strip_code_fences(text)
    try:
        out = json.loads(t)
        return out if isinstance(out, dict) else {"raw": out}
    except Exception:
        return {"raw": t}


def msg_text(content: Any) -> str:
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


def csv_bytes(rows: list[dict[str, str]]) -> bytes:
    """Convert rows with url/datasets columns to CSV bytes."""
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=["url", "datasets"])
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue().encode("utf-8")


def csv_bytes_any(rows: list[dict[str, Any]]) -> bytes:
    """Convert arbitrary dict rows to CSV bytes."""
    if not rows:
        return b""
    fields: list[str] = sorted({k for r in rows for k in r.keys()})
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fields)
    writer.writeheader()
    for r in rows:
        writer.writerow({k: r.get(k) for k in fields})
    return buf.getvalue().encode("utf-8")


def init_session_state(defaults: dict[str, Any]) -> None:
    """Initialize multiple session state keys with defaults if not already set.
    
    Example:
        init_session_state({
            "my_list": [],
            "my_flag": False,
            "my_count": 0,
        })
    """
    import streamlit as st
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default
