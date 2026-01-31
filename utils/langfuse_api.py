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
    page_size: int,
    page_limit: int,
    max_traces: int,
    retry: int,
    backoff: float,
    debug_out: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Fetch traces from Langfuse within a time window with pagination."""
    url = f"{base_url.rstrip('/')}/api/public/traces"
    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    session = requests.Session()
    page = 1
    limit = int(page_size)
    if limit <= 0:
        limit = 100
    min_limit = 25

    if isinstance(debug_out, dict):
        debug_out.clear()
        debug_out.update(
            {
                "url": url,
                "from_iso": from_iso,
                "to_iso": to_iso,
                "envs": envs,
                "page_limit": page_limit,
                "max_traces": max_traces,
                "page_size": limit,
                "min_page_size": min_limit,
                "pages": [],
                "limit_adjustments": [],
                "stopped_early_reason": None,
            }
        )

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
            t0 = time_mod.time()
            r = session.get(url, headers=headers, params=params, timeout=30)
            elapsed_s = time_mod.time() - t0
            if r.status_code < 400:
                last_error = None
                break

            # Langfuse/ClickHouse can error on large pages/time windows (esp. when reading the `output` column).
            # If we detect a server-side memory error, back off the per-page limit and retry the same page.
            text_preview = ""
            try:
                text_preview = str(getattr(r, "text", "") or "")
            except Exception:
                text_preview = ""
            is_memory_error = (
                r.status_code == 500
                and isinstance(text_preview, str)
                and (
                    "memory limit exceeded" in text_preview.lower()
                    or "overcommittracker" in text_preview.lower()
                )
            )
            if is_memory_error and limit > min_limit:
                new_limit = max(min_limit, int(limit // 2))
                if new_limit != limit:
                    if isinstance(debug_out, dict):
                        debug_out["limit_adjustments"].append(
                            {
                                "page": page,
                                "from_limit": int(limit),
                                "to_limit": int(new_limit),
                                "reason": "server_memory_limit_exceeded",
                            }
                        )
                    limit = new_limit
                    params["limit"] = limit
                    # Reset attempts when we change strategy; wait briefly to avoid hammering the server.
                    attempts = 0
                    time_mod.sleep(max(0.1, float(backoff)))
                    continue

            if 500 <= r.status_code < 600 and attempts < retry:
                attempts += 1
                last_error = f"Langfuse 5xx on page {page} (status {r.status_code}), retry {attempts}/{retry}"
                time_mod.sleep(backoff * attempts)
                continue

            last_error = f"Langfuse error on page {page} (status {r.status_code}): {text_preview[:500]}"
            break

        if last_error:
            if isinstance(debug_out, dict):
                debug_out["stopped_early_reason"] = last_error
                debug_out["final_page_size"] = int(limit)
            if HAS_STREAMLIT:
                try:
                    st.warning(f"Stopping fetch early: {last_error}")
                except Exception:
                    pass
            break

        data: Any
        try:
            data = r.json()
        except Exception:
            data = None
        batch = data.get("data") if isinstance(data, dict) else data
        if not batch:
            break

        if isinstance(debug_out, dict):
            page_entry: dict[str, Any] = {
                "page": page,
                "params": dict(params),
                "status_code": int(getattr(r, "status_code", 0) or 0),
                "elapsed_s": float(elapsed_s),
                "effective_page_size": int(limit),
            }
            if isinstance(data, dict):
                page_entry["response_keys"] = sorted([str(k) for k in data.keys()])
            if isinstance(batch, list):
                page_entry["items_returned"] = len(batch)
            debug_out["pages"].append(page_entry)

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
