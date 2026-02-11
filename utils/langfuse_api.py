"""Langfuse API utilities for fetching and managing traces/sessions."""

import base64
import hashlib
import json
import os
import time as time_mod
from pathlib import Path
from typing import Any
from datetime import date
from urllib.parse import urlencode

import requests

from utils.config_utils import normalize_langfuse_base_url

try:
    import streamlit as st
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False


def _http_debug_enabled() -> bool:
    v = str(os.getenv("LANGFUSE_HTTP_DEBUG", "") or "").strip().lower()
    return v in {"1", "true", "yes", "y", "on"}


def _preview_text(s: Any, limit: int = 1400) -> str:
    try:
        txt = str(s)
    except Exception:
        return ""
    if len(txt) <= limit:
        return txt
    return txt[:limit] + "…"


def _log_http(
    *,
    method: str,
    url: str,
    params: dict[str, Any] | None,
    json_payload: Any | None,
    response: requests.Response,
) -> None:
    if not _http_debug_enabled():
        return
    try:
        qs = ""
        if isinstance(params, dict) and params:
            try:
                qs = urlencode(params, doseq=True)
            except Exception:
                qs = ""
        full_url = f"{url}?{qs}" if qs else url
        status = int(getattr(response, "status_code", 0) or 0)
        print(f"[Langfuse HTTP] {method.upper()} {full_url} -> {status}")
        if json_payload is not None:
            print(f"[Langfuse HTTP] request.json: {_preview_text(json_payload)}")
        try:
            data = response.json()
            print(f"[Langfuse HTTP] response.json: {_preview_text(data)}")
        except Exception:
            print(f"[Langfuse HTTP] response.text: {_preview_text(getattr(response, 'text', '') or '')}")
    except Exception:
        return


def get_langfuse_headers(public_key: str, secret_key: str) -> dict[str, str]:
    """Build authorization headers for Langfuse API requests."""
    auth = base64.b64encode(f"{public_key}:{secret_key}".encode("utf-8")).decode("utf-8")
    return {"Authorization": f"Basic {auth}"}


def _auth_namespace_from_headers(headers: dict[str, str]) -> str:
    """Build a short auth namespace from Basic auth public key without storing credentials."""
    try:
        authorization = str(headers.get("Authorization") or "")
        if not authorization.startswith("Basic "):
            return ""
        encoded = authorization.split(" ", 1)[1].strip()
        decoded = base64.b64decode(encoded).decode("utf-8")
        public_key, _ = decoded.split(":", 1)
        if not public_key.strip():
            return ""
        return hashlib.sha256(public_key.encode("utf-8")).hexdigest()[:8]
    except Exception:
        return ""


def fetch_score_configs(
    *,
    base_url: str,
    headers: dict[str, str],
    page: int = 1,
    limit: int = 100,
    http_timeout_s: float = 30,
) -> list[dict[str, Any]]:
    base = normalize_langfuse_base_url(base_url)
    url = f"{base}/api/public/score-configs"
    params = {"page": int(page), "limit": int(limit)}
    r = requests.get(url, headers=headers, params=params, timeout=float(http_timeout_s))
    _log_http(method="GET", url=url, params=params, json_payload=None, response=r)
    r.raise_for_status()
    data = r.json()
    rows = data.get("data") if isinstance(data, dict) else None
    if isinstance(rows, list):
        return [x for x in rows if isinstance(x, dict)]
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    return []


def get_annotation_queue_item(
    *,
    base_url: str,
    headers: dict[str, str],
    queue_id: str,
    item_id: str,
    http_timeout_s: float = 30,
) -> dict[str, Any]:
    base = normalize_langfuse_base_url(base_url)
    url = f"{base}/api/public/annotation-queues/{queue_id}/items/{item_id}"
    r = requests.get(url, headers=headers, timeout=float(http_timeout_s))
    _log_http(method="GET", url=url, params=None, json_payload=None, response=r)
    r.raise_for_status()
    out = r.json()
    return out if isinstance(out, dict) else {}


def get_annotation_queue(
    *,
    base_url: str,
    headers: dict[str, str],
    queue_id: str,
    http_timeout_s: float = 30,
) -> dict[str, Any]:
    base = normalize_langfuse_base_url(base_url)
    url = f"{base}/api/public/annotation-queues/{queue_id}"
    r = requests.get(url, headers=headers, timeout=float(http_timeout_s))
    _log_http(method="GET", url=url, params=None, json_payload=None, response=r)
    r.raise_for_status()
    out = r.json()
    return out if isinstance(out, dict) else {}


def list_annotation_queue_items(
    *,
    base_url: str,
    headers: dict[str, str],
    queue_id: str,
    status: str | None = None,
    page: int = 1,
    limit: int = 100,
    http_timeout_s: float = 30,
) -> list[dict[str, Any]]:
    base = normalize_langfuse_base_url(base_url)
    url = f"{base}/api/public/annotation-queues/{queue_id}/items"
    params: dict[str, Any] = {"page": int(page), "limit": int(limit)}
    if status is not None and str(status).strip():
        params["status"] = str(status)
    r = requests.get(url, headers=headers, params=params, timeout=float(http_timeout_s))
    _log_http(method="GET", url=url, params=params, json_payload=None, response=r)
    r.raise_for_status()
    data = r.json()
    rows = data.get("data") if isinstance(data, dict) else None
    if isinstance(rows, list):
        return [x for x in rows if isinstance(x, dict)]
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    return []


def list_annotation_queues(
    *,
    base_url: str,
    headers: dict[str, str],
    page: int = 1,
    limit: int = 100,
    http_timeout_s: float = 30,
) -> list[dict[str, Any]]:
    base = normalize_langfuse_base_url(base_url)
    url = f"{base}/api/public/annotation-queues"
    params = {"page": int(page), "limit": int(limit)}
    r = requests.get(url, headers=headers, params=params, timeout=float(http_timeout_s))
    _log_http(method="GET", url=url, params=params, json_payload=None, response=r)
    r.raise_for_status()
    data = r.json()
    rows = data.get("data") if isinstance(data, dict) else None
    if isinstance(rows, list):
        return [x for x in rows if isinstance(x, dict)]
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    return []


def create_annotation_queue(
    *,
    base_url: str,
    headers: dict[str, str],
    name: str,
    score_config_ids: list[str],
    description: str | None = None,
    http_timeout_s: float = 30,
) -> dict[str, Any]:
    base = normalize_langfuse_base_url(base_url)
    url = f"{base}/api/public/annotation-queues"
    payload: dict[str, Any] = {
        "name": str(name),
        "scoreConfigIds": [str(x) for x in score_config_ids if str(x).strip()],
    }
    if description is not None:
        payload["description"] = str(description)
    r = requests.post(url, headers=headers, json=payload, timeout=float(http_timeout_s))
    _log_http(method="POST", url=url, params=None, json_payload=payload, response=r)
    r.raise_for_status()
    out = r.json()
    return out if isinstance(out, dict) else {}


def create_annotation_queue_item(
    *,
    base_url: str,
    headers: dict[str, str],
    queue_id: str,
    object_id: str,
    object_type: str = "TRACE",
    status: str | None = None,
    http_timeout_s: float = 30,
) -> dict[str, Any]:
    base = normalize_langfuse_base_url(base_url)
    url = f"{base}/api/public/annotation-queues/{queue_id}/items"
    payload: dict[str, Any] = {
        "objectId": str(object_id),
        "objectType": str(object_type),
    }
    if status is not None:
        payload["status"] = str(status)
    r = requests.post(url, headers=headers, json=payload, timeout=float(http_timeout_s))
    _log_http(method="POST", url=url, params=None, json_payload=payload, response=r)
    r.raise_for_status()
    out = r.json()
    return out if isinstance(out, dict) else {}


def update_annotation_queue_item(
    *,
    base_url: str,
    headers: dict[str, str],
    queue_id: str,
    item_id: str,
    status: str,
    http_timeout_s: float = 30,
) -> dict[str, Any]:
    base = normalize_langfuse_base_url(base_url)
    url = f"{base}/api/public/annotation-queues/{queue_id}/items/{item_id}"
    payload: dict[str, Any] = {"status": str(status)}
    r = requests.patch(url, headers=headers, json=payload, timeout=float(http_timeout_s))
    _log_http(method="PATCH", url=url, params=None, json_payload=payload, response=r)
    r.raise_for_status()
    out = r.json()
    return out if isinstance(out, dict) else {}


def fetch_projects(
    *,
    base_url: str,
    headers: dict[str, str],
    http_timeout_s: float = 30,
) -> list[dict[str, Any]]:
    base = normalize_langfuse_base_url(base_url)
    url = f"{base}/api/public/projects"
    r = requests.get(url, headers=headers, timeout=float(http_timeout_s))
    r.raise_for_status()
    data = r.json()
    rows = data.get("data") if isinstance(data, dict) else None
    if isinstance(rows, list):
        return [x for x in rows if isinstance(x, dict)]
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    return []


def create_score(
    *,
    base_url: str,
    headers: dict[str, str],
    trace_id: str,
    name: str,
    value: str | float | int,
    environment: str | None = None,
    comment: str | None = None,
    metadata: dict[str, Any] | None = None,
    config_id: str | None = None,
    queue_id: str | None = None,
    score_id: str | None = None,
    http_timeout_s: float = 30,
) -> dict[str, Any]:
    base = normalize_langfuse_base_url(base_url)
    url = f"{base}/api/public/scores"
    payload: dict[str, Any] = {
        "name": str(name),
        "value": value,
        "traceId": str(trace_id),
    }
    if score_id is not None and str(score_id).strip():
        payload["id"] = str(score_id)
    if environment is not None and str(environment).strip():
        payload["environment"] = str(environment)
    if comment is not None:
        payload["comment"] = str(comment)
    if metadata is not None:
        payload["metadata"] = metadata if isinstance(metadata, dict) else {"_raw": metadata}
    if config_id is not None and str(config_id).strip():
        payload["configId"] = str(config_id)
    if queue_id is not None and str(queue_id).strip():
        payload["queueId"] = str(queue_id)
        if not isinstance(payload.get("metadata"), dict):
            payload["metadata"] = {}
        payload["metadata"]["queue_id"] = str(queue_id)
    payload["source"] = "API"

    r = requests.post(url, headers=headers, json=payload, timeout=float(http_timeout_s))
    _log_http(method="POST", url=url, params=None, json_payload=payload, response=r)
    if int(getattr(r, "status_code", 0) or 0) >= 400:
        details: str | None = None
        try:
            err = r.json()
            if isinstance(err, dict):
                details = str(
                    err.get("message")
                    or err.get("error")
                    or err.get("detail")
                    or err.get("errors")
                    or ""
                ).strip() or None
            elif err is not None:
                details = str(err).strip() or None
        except Exception:
            details = None

        if not details:
            try:
                details = str(getattr(r, "text", "") or "").strip() or None
            except Exception:
                details = None

        prefix = f"Langfuse create_score failed (HTTP {int(r.status_code)})"
        if details:
            raise Exception(f"{prefix}: {details[:1200]}")
        raise Exception(prefix)

    try:
        out = r.json()
    except Exception as e:
        raw = ""
        try:
            raw = str(getattr(r, "text", "") or "")
        except Exception:
            raw = ""
        raise Exception(f"Langfuse create_score returned invalid JSON: {raw[:1200]}") from e

    if not isinstance(out, dict):
        raise Exception(f"Langfuse create_score returned unexpected payload type: {type(out).__name__}")

    expected_id = payload.get("id")
    if isinstance(expected_id, str) and expected_id.strip():
        returned_id = out.get("id")
        if expected_id != returned_id:
            raise Exception(f"Expected score id {expected_id} but got {returned_id}")

    if not out.get("id"):
        raise Exception(
            f"Langfuse create_score succeeded but response missing id: {json.dumps(out, default=str)[:1200]}"
        )

    return out


def delete_score(
    *,
    base_url: str,
    headers: dict[str, str],
    score_id: str,
    http_timeout_s: float = 30,
) -> None:
    base = normalize_langfuse_base_url(base_url)
    url = f"{base}/api/public/scores/{score_id}"
    r = requests.delete(url, headers=headers, timeout=float(http_timeout_s))
    _log_http(method="DELETE", url=url, params=None, json_payload=None, response=r)
    if r.status_code in (200, 204, 404):
        return
    r.raise_for_status()


def fetch_scores_by_queue(
    *,
    base_url: str,
    headers: dict[str, str],
    queue_id: str,
    page: int = 1,
    limit: int = 100,
    http_timeout_s: float = 30,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Fetch scores filtered by annotation queueId using the v2 API.
    
    Returns a tuple of (scores_list, metadata_dict).
    Metadata includes totalCount and pagination info.
    """
    base = normalize_langfuse_base_url(base_url)
    url = f"{base}/api/public/v2/scores"
    all_scores: list[dict[str, Any]] = []
    meta: dict[str, Any] = {"totalCount": 0, "page": page, "limit": limit}
    
    current_page = page
    while True:
        params: dict[str, Any] = {
            "queueId": str(queue_id),
            "page": int(current_page),
            "limit": int(limit),
        }
        r = requests.get(url, headers=headers, params=params, timeout=float(http_timeout_s))
        _log_http(method="GET", url=url, params=params, json_payload=None, response=r)
        r.raise_for_status()
        data = r.json()
       
        if isinstance(data, dict):
            meta["totalCount"] = data.get("meta", {}).get("totalCount", 0)
            rows = data.get("data", [])
        else:
            rows = data if isinstance(data, list) else []
        
        if not rows:
            break
            
        for row in rows:
            if isinstance(row, dict):
                all_scores.append(row)
        
        if len(rows) < limit:
            break
            
        current_page += 1
    
    meta["fetchedCount"] = len(all_scores)
    return all_scores, meta


def _default_cache_dir() -> Path:
    return Path(__file__).resolve().parent / "__pycache__" / "langfuse_cache"


def _cache_path(prefix: str, payload: dict[str, Any], cache_dir: Path) -> Path:
    raw = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    key = hashlib.sha256(raw).hexdigest()
    return cache_dir / f"{prefix}_{key}.json"


def _try_load_cache(path: Path) -> Any | None:
    try:
        if not path.exists():
            return None
        txt = path.read_text(encoding="utf-8")
        return json.loads(txt)
    except Exception:
        return None


def _try_write_cache(path: Path, data: Any) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, path)
    except Exception:
        return


def clear_langfuse_disk_cache(cache_dir: str | None = None) -> dict[str, Any]:
    """Clear Langfuse disk cache files without raising exceptions."""
    cache_root = Path(cache_dir) if isinstance(cache_dir, str) and cache_dir.strip() else _default_cache_dir()
    files_removed = 0
    errors = 0

    try:
        if cache_root.exists():
            for path in cache_root.rglob("*"):
                if not path.is_file():
                    continue
                try:
                    path.unlink()
                    files_removed += 1
                except Exception:
                    errors += 1
    except Exception:
        errors += 1

    return {"cache_root": str(cache_root), "files_removed": files_removed, "errors": errors}


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
    http_timeout_s: float = 30,
    debug_out: dict[str, Any] | None = None,
    use_disk_cache: bool = True,
    cache_dir: str | None = None,
) -> list[dict[str, Any]]:
    """Fetch traces from Langfuse within a time window with pagination."""
    normalized_base = normalize_langfuse_base_url(base_url)
    url = f"{normalized_base}/api/public/traces"
    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    cache_root = Path(cache_dir) if isinstance(cache_dir, str) and cache_dir.strip() else _default_cache_dir()
    cache_payload = {
        "kind": "fetch_traces_window",
        "base_url": normalized_base,
        "from_iso": from_iso,
        "to_iso": to_iso,
        "envs": envs or [],
        "page_size": int(page_size),
        "page_limit": int(page_limit),
        "max_traces": int(max_traces),
        "auth_ns": _auth_namespace_from_headers(headers),
    }
    cache_path = _cache_path("traces", cache_payload, cache_root)
    if use_disk_cache:
        cached = _try_load_cache(cache_path)
        if isinstance(cached, list) and all(isinstance(x, dict) for x in cached):
            if isinstance(debug_out, dict):
                debug_out.clear()
                debug_out.update({"cache_hit": True, "cache_path": str(cache_path)})
            return list(cached)

    session = requests.Session()
    page = 1
    requested_limit = int(page_size)
    if requested_limit <= 0:
        requested_limit = 100

    # Langfuse enforces `limit <= 100`.
    limit = min(100, requested_limit)
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
                "requested_page_size": requested_limit,
                "page_size": limit,
                "min_page_size": min_limit,
                "http_timeout_s": float(http_timeout_s),
                "pages": [],
                "limit_adjustments": [],
                "stopped_early_reason": None,
            }
        )

    stopped_early = False
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
            r = session.get(url, headers=headers, params=params, timeout=float(http_timeout_s))
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
            stopped_early = True
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

    if use_disk_cache and not stopped_early:
        _try_write_cache(cache_path, rows)
        if isinstance(debug_out, dict):
            debug_out["cache_hit"] = False
            debug_out["cache_path"] = str(cache_path)

    return rows


def fetch_trace(
    *,
    base_url: str,
    headers: dict[str, str],
    trace_id: str,
    http_timeout_s: float = 30,
) -> dict[str, Any]:
    base = normalize_langfuse_base_url(base_url)
    url = f"{base}/api/public/traces/{trace_id}"
    r = requests.get(url, headers=headers, timeout=float(http_timeout_s))
    r.raise_for_status()
    out = r.json()
    return out if isinstance(out, dict) else {}


def fetch_user_first_seen(
    *,
    base_url: str,
    headers: dict[str, str],
    from_iso: str,
    envs: list[str] | None,
    page_size: int,
    page_limit: int,
    retry: int,
    backoff: float,
    http_timeout_s: float = 30,
    debug_out: dict[str, Any] | None = None,
    use_disk_cache: bool = True,
    cache_dir: str | None = None,
) -> dict[str, str]:
    """Fetch per-user first-seen timestamps by paging traces oldest -> newest.

    Uses the public traces endpoint with `fields=core` to keep payload small.
    Returns a mapping of userId -> first timestamp (ISO string) seen in the scan.
    """

    base = normalize_langfuse_base_url(base_url)
    url = f"{base}/api/public/traces"
    session = requests.Session()

    cache_root = Path(cache_dir) if isinstance(cache_dir, str) and cache_dir.strip() else _default_cache_dir()
    cache_payload = {
        "kind": "fetch_user_first_seen",
        "cache_day": date.today().isoformat(),
        "base_url": base,
        "from_iso": from_iso,
        "envs": envs or [],
        "page_size": int(page_size),
        "page_limit": int(page_limit),
        "exclude_machine_users": True,
    }
    cache_path = _cache_path("user_first_seen", cache_payload, cache_root)
    if use_disk_cache:
        cached = _try_load_cache(cache_path)
        if isinstance(cached, dict) and all(isinstance(k, str) and isinstance(v, str) for k, v in cached.items()):
            if isinstance(debug_out, dict):
                debug_out.clear()
                debug_out.update({"cache_hit": True, "cache_path": str(cache_path)})
            return dict(cached)

    requested_limit = int(page_size)
    if requested_limit <= 0:
        requested_limit = 100
    limit = min(100, requested_limit)

    if isinstance(debug_out, dict):
        debug_out.clear()
        debug_out.update(
            {
                "url": url,
                "from_iso": from_iso,
                "envs": envs,
                "page_limit": int(page_limit),
                "requested_page_size": int(requested_limit),
                "page_size": int(limit),
                "http_timeout_s": float(http_timeout_s),
                "pages": [],
                "stopped_early_reason": None,
            }
        )

    first_seen: dict[str, str] = {}
    stopped_early = False
    page = 1
    while page <= page_limit:
        params: dict[str, Any] = {
            "fromTimestamp": from_iso,
            "limit": limit,
            "page": page,
            "orderBy": "timestamp.asc",
            "fields": "core",
        }
        if envs:
            params["environment"] = envs

        attempts = 0
        last_error: str | None = None
        while True:
            t0 = time_mod.time()
            r = session.get(url, headers=headers, params=params, timeout=float(http_timeout_s))
            elapsed_s = time_mod.time() - t0
            if r.status_code < 400:
                last_error = None
                break

            text_preview = ""
            try:
                text_preview = str(getattr(r, "text", "") or "")
            except Exception:
                text_preview = ""

            if 500 <= r.status_code < 600 and attempts < retry:
                attempts += 1
                last_error = (
                    f"Langfuse 5xx on page {page} (status {r.status_code}), retry {attempts}/{retry}"
                )
                time_mod.sleep(backoff * attempts)
                continue

            last_error = f"Langfuse error on page {page} (status {r.status_code}): {text_preview[:500]}"
            break

        if last_error:
            if isinstance(debug_out, dict):
                debug_out["stopped_early_reason"] = last_error
            stopped_early = True
            if HAS_STREAMLIT:
                try:
                    st.warning(f"Stopping user scan early: {last_error}")
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
                "items_returned": len(batch) if isinstance(batch, list) else None,
            }
            debug_out["pages"].append(page_entry)

        for it in batch:
            if not isinstance(it, dict):
                continue
            user_id = it.get("userId")
            ts = it.get("timestamp")
            if not isinstance(user_id, str) or not user_id.strip():
                continue
            if "machine" in user_id.lower():
                continue
            if not isinstance(ts, str) or not ts.strip():
                continue
            if user_id not in first_seen:
                first_seen[user_id] = ts

        if isinstance(batch, list) and len(batch) < limit:
            break

        page += 1
        time_mod.sleep(0.05)

    if isinstance(debug_out, dict):
        debug_out["users_found"] = int(len(first_seen))
        debug_out["pages_scanned"] = int(page - 1) if page > 1 else 0

    if use_disk_cache and not stopped_early:
        _try_write_cache(cache_path, first_seen)
        if isinstance(debug_out, dict):
            debug_out["cache_hit"] = False
            debug_out["cache_path"] = str(cache_path)

    return first_seen


def user_first_seen_cache_path(
    *,
    base_url: str,
    from_iso: str,
    envs: list[str] | None,
    page_size: int,
    page_limit: int,
    cache_dir: str | None = None,
) -> Path:
    cache_root = Path(cache_dir) if isinstance(cache_dir, str) and cache_dir.strip() else _default_cache_dir()
    cache_payload = {
        "kind": "fetch_user_first_seen",
        "cache_day": date.today().isoformat(),
        "base_url": normalize_langfuse_base_url(base_url),
        "from_iso": from_iso,
        "envs": envs or [],
        "page_size": int(page_size),
        "page_limit": int(page_limit),
        "exclude_machine_users": True,
    }
    return _cache_path("user_first_seen", cache_payload, cache_root)


def invalidate_user_first_seen_cache(
    *,
    base_url: str,
    from_iso: str,
    envs: list[str] | None,
    page_size: int,
    page_limit: int,
    cache_dir: str | None = None,
) -> bool:
    path = user_first_seen_cache_path(
        base_url=base_url,
        from_iso=from_iso,
        envs=envs,
        page_size=page_size,
        page_limit=page_limit,
        cache_dir=cache_dir,
    )
    try:
        if path.exists():
            path.unlink()
            return True
    except Exception:
        return False
    return False


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
