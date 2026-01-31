"""General data processing and formatting utilities."""

import csv
import io
import json
import os
from datetime import datetime, timezone
from pathlib import Path
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


def save_bytes_to_local_path(data: bytes, destination: str, default_filename: str) -> str:
    dest = (destination or "").strip()
    if not dest:
        raise ValueError("Missing destination")

    dest = os.path.expanduser(dest)
    p = Path(dest)

    if str(p).lower().endswith(".csv"):
        out_path = p
        out_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = p
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / default_filename

    out_path.write_bytes(data or b"")
    return str(out_path)


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
