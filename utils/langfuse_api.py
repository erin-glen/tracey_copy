"""Langfuse API utilities for fetching and managing traces/sessions."""

import base64
import time as time_mod
from typing import Any

import requests

try:
    import streamlit as st
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False


def get_langfuse_headers(public_key: str, secret_key: str) -> dict[str, str]:
    """Build authorization headers for Langfuse API requests."""
    auth = base64.b64encode(f"{public_key}:{secret_key}".encode("utf-8")).decode("utf-8")
    return {"Authorization": f"Basic {auth}"}


def fetch_traces_window(
    *,
    base_url: str,
    headers: dict[str, str],
    from_iso: str,
    to_iso: str,
    envs: list[str] | None,
    page_limit: int,
    max_traces: int,
    retry: int,
    backoff: float,
) -> list[dict[str, Any]]:
    """Fetch traces from Langfuse within a time window with pagination."""
    url = f"{base_url.rstrip('/')}/api/public/traces"
    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    session = requests.Session()
    page = 1
    limit = 100

    while page <= page_limit and len(rows) < max_traces:
        params: dict[str, Any] = {
            "fromTimestamp": from_iso,
            "toTimestamp": to_iso,
            "limit": limit,
            "page": page,
        }
        if envs:
            params["environment"] = envs

        attempts = 0
        last_error: str | None = None
        while True:
            r = session.get(url, headers=headers, params=params, timeout=30)
            if r.status_code < 400:
                last_error = None
                break
            if 500 <= r.status_code < 600 and attempts < retry:
                attempts += 1
                last_error = f"Langfuse 5xx on page {page} (status {r.status_code}), retry {attempts}/{retry}"
                time_mod.sleep(backoff * attempts)
                continue

            last_error = f"Langfuse error on page {page} (status {r.status_code}): {r.text[:500]}"
            break

        if last_error:
            if HAS_STREAMLIT:
                try:
                    st.warning(f"Stopping fetch early: {last_error}")
                except Exception:
                    pass
            break

        data = r.json()
        batch = data.get("data") if isinstance(data, dict) else data
        if not batch:
            break

        for it in batch:
            if not isinstance(it, dict):
                continue
            _id = it.get("id")
            if isinstance(_id, str) and _id in seen_ids:
                continue
            if isinstance(_id, str):
                seen_ids.add(_id)
            rows.append(it)
            if len(rows) >= max_traces:
                break

        if len(batch) < limit:
            break

        page += 1
        time_mod.sleep(0.05)

    return rows


def extract_datasets_from_session(langfuse: Any, session_id: str) -> str:
    """Extract dataset names from a Langfuse session's traces."""
    from langfuse.api.core.api_error import ApiError

    try:
        session = langfuse.api.sessions.get(session_id=session_id)
    except ApiError:
        raise
    datasets: set[str] = set()

    for trace in getattr(session, "traces", []) or []:
        try:
            output = getattr(trace, "output", None)
            if output and isinstance(output, dict):
                dataset_info = output.get("dataset", {})
                if dataset_info and isinstance(dataset_info, dict):
                    dataset_name = dataset_info.get("dataset_name")
                    if dataset_name:
                        datasets.add(str(dataset_name))
        except (AttributeError, TypeError, KeyError):
            continue

    return ", ".join(sorted(datasets))
