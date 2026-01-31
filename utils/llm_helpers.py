from __future__ import annotations

import json
from typing import Any

import streamlit as st

from utils.data_helpers import strip_code_fences


def get_gemini_model_options(api_key: str, cache_key: str = "gemini_model_options") -> list[str]:
    cached = st.session_state.get(cache_key)
    if isinstance(cached, list) and all(isinstance(x, str) for x in cached) and len(cached):
        return list(cached)

    fallback = [
        "gemini-2.5-flash-lite",
        "gemini-2.0-flash",
        "gemini-1.5-pro",
        "gemini-1.5-flash",
    ]

    def _parse_version(name: str) -> tuple[int, int, int]:
        if not isinstance(name, str) or not name.startswith("gemini-"):
            return (0, 0, 0)
        rest = name[len("gemini-") :]
        ver = rest.split("-", 1)[0]
        parts = ver.split(".")
        out: list[int] = []
        for p in parts[:3]:
            try:
                out.append(int(p))
            except Exception:
                out.append(0)
        while len(out) < 3:
            out.append(0)
        return (out[0], out[1], out[2])

    try:
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        models: list[str] = []
        for m in genai.list_models():
            name = getattr(m, "name", None)
            if not isinstance(name, str) or not name.strip():
                continue
            methods = getattr(m, "supported_generation_methods", None)
            if isinstance(methods, (list, tuple)) and "generateContent" not in methods:
                continue
            cleaned = name.replace("models/", "")
            if not cleaned.startswith("gemini-"):
                continue
            if "image" in cleaned.lower():
                continue
            models.append(cleaned)

        models = sorted(set(models), key=lambda n: _parse_version(n), reverse=True)
        st.session_state[cache_key] = models if models else fallback
        return list(st.session_state[cache_key])
    except Exception:
        st.session_state[cache_key] = fallback
        return fallback


def chunked(items: list[dict[str, Any]], batch_size: int) -> list[list[dict[str, Any]]]:
    if batch_size <= 1:
        return [items]
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def truncate_text(txt: str, max_chars: int) -> str:
    if max_chars <= 0:
        return txt
    if len(txt) <= max_chars:
        return txt
    if max_chars <= 3:
        return txt[:max_chars]
    return txt[: max_chars - 3] + "..."


def parse_json_any(txt: str) -> Any:
    cleaned = strip_code_fences(str(txt or "")).strip()
    if not cleaned:
        return None
    try:
        return json.loads(cleaned)
    except Exception:
        return None


def parse_json_dict(txt: str) -> dict[str, Any]:
    cleaned = strip_code_fences(str(txt or "")).strip()
    if not cleaned:
        return {}
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
        return {"raw": parsed}
    except Exception:
        return {"raw": cleaned}
