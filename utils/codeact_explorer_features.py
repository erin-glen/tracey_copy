"""Deterministic feature extraction for CodeAct explorer workflows."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any
from urllib.parse import urlparse

import pandas as pd

from utils.codeact_utils import decode_maybe_base64, find_codeact_parts


_ANALYTICS_URL_RE = re.compile(r"https?://analytics\.globalnaturewatch\.org[^\s\"'`<>)]*", re.IGNORECASE)
_UUIDISH_SUFFIX_RE = re.compile(r"/[0-9a-f]{8,}(?:-[0-9a-f-]{4,})?$", re.IGNORECASE)
_PERCENT_MATH_RE = re.compile(r"\*\s*100(?:\.0+)?\b")


def _safe_str(value: Any) -> str:
    return str(value or "")


def _truncate(value: Any, limit: int) -> str:
    text = _safe_str(value)
    if len(text) <= limit:
        return text
    if limit <= 3:
        return "." * max(limit, 0)
    return text[: limit - 3] + "..."


def _is_retrieval_call(text: str) -> bool:
    lowered = text.lower()
    return (
        "analytics.globalnaturewatch.org" in lowered
        or "pd.read_json" in lowered
        or "requests.get(" in lowered
        or "httpx.get(" in lowered
    )


def _walk_source_records(obj: Any) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []

    def _walk(node: Any) -> None:
        if isinstance(node, dict):
            source_url = node.get("source_url")
            if isinstance(source_url, str) and source_url.strip():
                records.append(
                    {
                        "source_url": source_url.strip(),
                        "dataset_name": _safe_str(node.get("dataset_name")).strip(),
                        "aoi_name": _safe_str(node.get("aoi_name")).strip(),
                        "start_date": _safe_str(node.get("start_date")).strip(),
                        "end_date": _safe_str(node.get("end_date")).strip(),
                    }
                )
            for value in node.values():
                if isinstance(value, (dict, list)):
                    _walk(value)
        elif isinstance(node, list):
            for item in node:
                if isinstance(item, (dict, list)):
                    _walk(item)

    _walk(obj)
    return records


def _extract_chart_rows(output_obj: dict[str, Any]) -> list[dict[str, Any]]:
    charts = output_obj.get("charts_data")
    if not isinstance(charts, list):
        return []
    rows: list[dict[str, Any]] = []
    for chart in charts:
        if isinstance(chart, dict):
            rows.append(chart)
    return rows


def _extract_analytics_urls_from_blocks(decoded_code_blocks: Iterable[str]) -> list[str]:
    urls: list[str] = []
    seen: set[str] = set()
    for block in decoded_code_blocks:
        for raw in _ANALYTICS_URL_RE.findall(_safe_str(block)):
            url = raw.strip().rstrip(".,;")
            if url and url not in seen:
                seen.add(url)
                urls.append(url)
    return urls


def _derive_endpoint_base(url: str) -> str:
    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        return ""
    path = parsed.path or ""
    marker = "/analytics"
    idx = path.lower().find(marker)
    if idx == -1:
        return f"{parsed.scheme}://{parsed.netloc}"
    base_path = path[: idx + len(marker)]
    base_path = _UUIDISH_SUFFIX_RE.sub("", base_path)
    base_path = base_path.rstrip("/")
    return f"{parsed.scheme}://{parsed.netloc}{base_path}"


def extract_codeact_timeline(output_obj: dict) -> list[dict]:
    timeline: list[dict] = []
    parts = find_codeact_parts(output_obj if isinstance(output_obj, dict) else {})
    for part in parts:
        ptype = _safe_str(part.get("type")).strip().lower() or "unknown"
        decoded, from_b64, _ = decode_maybe_base64(part.get("content"))
        decoded_text = _safe_str(decoded)
        timeline.append(
            {
                "type": ptype,
                "decoded": decoded_text,
                "decoded_from_base64": bool(from_b64),
                "char_len": len(decoded_text),
            }
        )
    return timeline


def extract_codeact_summary(output_obj: dict) -> dict:
    out = output_obj if isinstance(output_obj, dict) else {}
    timeline = extract_codeact_timeline(out)
    code_blocks = [p for p in timeline if p.get("type") == "code_block"]
    exec_outputs = [p for p in timeline if p.get("type") == "execution_output"]
    text_outputs = [p for p in timeline if p.get("type") == "text_output"]

    final_insight = ""
    has_final_marker = False
    decode_errors = 0
    for part in find_codeact_parts(out):
        decoded, _from_b64, decode_error = decode_maybe_base64(part.get("content"))
        decode_errors += int(decode_error)
        if _safe_str(part.get("type")).strip().lower() == "text_output":
            text = _safe_str(decoded)
            if "FINAL DATA-DRIVEN INSIGHT" in text:
                has_final_marker = True
                final_insight = text
            elif not final_insight and text.strip():
                final_insight = text

    chart_rows = _extract_chart_rows(out)
    chart_types = sorted({str(c.get("type") or "").strip().lower() for c in chart_rows if str(c.get("type") or "").strip()})
    chart_titles: list[str] = []
    seen_titles: set[str] = set()
    for chart in chart_rows:
        title = _safe_str(chart.get("title")).strip()
        if title and title not in seen_titles:
            seen_titles.add(title)
            chart_titles.append(title)
    chart_titles_joined = ",".join(chart_titles)
    chart_titles_joined = _truncate(chart_titles_joined, 200)

    provenance_records = _walk_source_records(out.get("raw_data"))
    unique_source_urls: list[str] = []
    source_seen: set[str] = set()
    for record in provenance_records:
        url = _safe_str(record.get("source_url")).strip()
        if url and url not in source_seen:
            source_seen.add(url)
            unique_source_urls.append(url)

    source_urls_sample = " | ".join(_truncate(url, 120) for url in unique_source_urls[:2])

    dataset_names: list[str] = []
    dataset_seen: set[str] = set()
    for record in provenance_records:
        name = _safe_str(record.get("dataset_name")).strip()
        if name and name not in dataset_seen:
            dataset_seen.add(name)
            dataset_names.append(name)
    raw_dataset_names = _truncate(",".join(dataset_names), 200)

    decoded_code_blocks = [_safe_str(p.get("decoded")) for p in code_blocks]
    analytics_urls = _extract_analytics_urls_from_blocks(decoded_code_blocks)
    endpoint_bases = sorted({base for base in (_derive_endpoint_base(url) for url in analytics_urls) if base})

    return {
        "codeact_parts_count": int(len(timeline)),
        "codeact_code_blocks_count": int(len(code_blocks)),
        "codeact_exec_outputs_count": int(len(exec_outputs)),
        "codeact_text_outputs_count": int(len(text_outputs)),
        "codeact_decoded_chars_total": int(sum(int(p.get("char_len") or 0) for p in timeline)),
        "codeact_decode_errors": int(decode_errors),
        "codeact_final_insight": final_insight,
        "codeact_has_final_insight_marker": bool(has_final_marker),
        "codeact_chart_count": int(len(chart_rows)),
        "codeact_chart_types": ",".join(chart_types),
        "codeact_chart_titles": chart_titles_joined,
        "codeact_source_url_count": int(len(unique_source_urls)),
        "codeact_source_urls_sample": source_urls_sample,
        "codeact_raw_dataset_names": raw_dataset_names,
        "codeact_analytics_url_count": int(len(analytics_urls)),
        "codeact_analytics_urls_sample": " | ".join(_truncate(url, 120) for url in analytics_urls[:2]),
        "codeact_endpoint_bases": ",".join(endpoint_bases),
        "codeact_endpoint_base_count": int(len(endpoint_bases)),
    }


def classify_codeact_retrieval_mode(decoded_code_blocks: list[str], output_obj: dict) -> str:
    out = output_obj if isinstance(output_obj, dict) else {}
    has_analytics = any("analytics.globalnaturewatch.org" in _safe_str(block).lower() for block in decoded_code_blocks)
    source_count = extract_codeact_summary(out).get("codeact_source_url_count", 0)
    has_sources = int(source_count) >= 1

    if has_analytics and has_sources:
        return "mixed"
    if has_analytics:
        return "analytics_api"
    if has_sources:
        return "prefetched_only"
    return "unknown"


def classify_codeact_chart_prep_mode(decoded_code_blocks: list[str]) -> str:
    if not decoded_code_blocks:
        return "unknown"

    first_retrieval_idx = -1
    later_retrieval_idx = -1

    for idx, block in enumerate(decoded_code_blocks):
        text = _safe_str(block)
        if "pd.DataFrame({" in text or "pd.DataFrame([" in text:
            return "hardcoded_chart_data"
        if _is_retrieval_call(text):
            if first_retrieval_idx == -1:
                first_retrieval_idx = idx
            elif idx > first_retrieval_idx:
                later_retrieval_idx = idx

    if later_retrieval_idx > first_retrieval_idx >= 0 and len(decoded_code_blocks) > 1:
        return "requery_or_reload"

    if first_retrieval_idx == 0 and len(decoded_code_blocks) > 1:
        if all(not _is_retrieval_call(_safe_str(block)) for block in decoded_code_blocks[1:]):
            return "transform_only"

    return "unknown"


def extract_codeact_analysis_tags(decoded_code_blocks: list[str], output_obj: dict) -> str:
    tags: list[str] = []
    joined = "\n".join(_safe_str(block) for block in decoded_code_blocks)
    lowered = joined.lower()

    if ".groupby(" in joined:
        tags.append("groupby")
    if "pivot_table" in lowered or ".pivot(" in lowered:
        tags.append("pivot")
    if ".merge(" in joined:
        tags.append("merge")
    if "percent" in lowered or _PERCENT_MATH_RE.search(joined):
        tags.append("percent_math")
    if ".dt." in joined:
        tags.append("datetime_ops")

    analytics_urls = _extract_analytics_urls_from_blocks(decoded_code_blocks)
    if len(analytics_urls) >= 2:
        tags.append("multi_endpoint")

    subregion_aois = (output_obj or {}).get("subregion_aois") if isinstance(output_obj, dict) else None
    if isinstance(subregion_aois, list) and len(subregion_aois) > 0:
        tags.append("subregion")

    return "|".join(tags)


def compute_codeact_flags(decoded_code_blocks: list[str], output_obj: dict) -> dict:
    out = output_obj if isinstance(output_obj, dict) else {}
    summary = extract_codeact_summary(out)
    chart_prep_mode = classify_codeact_chart_prep_mode(decoded_code_blocks)

    charts = _extract_chart_rows(out)
    pie_percent_sum_off = False
    for chart in charts:
        if str(chart.get("type") or "").strip().lower() != "pie":
            continue
        rows = chart.get("data")
        if not isinstance(rows, list) or not rows:
            continue
        values: list[float] = []
        valid = True
        for row in rows:
            if not isinstance(row, dict):
                valid = False
                break
            value = row.get("value")
            try:
                number = float(value)
            except (TypeError, ValueError):
                valid = False
                break
            values.append(number)
        if not valid or not values:
            continue
        if all(0.0 <= num <= 100.0 for num in values):
            if abs(sum(values) - 100.0) > 2.0:
                pie_percent_sum_off = True
                break

    dataset_endpoint = ""
    dataset = out.get("dataset")
    if isinstance(dataset, dict):
        dataset_endpoint = _safe_str(dataset.get("analytics_api_endpoint")).strip()

    endpoint_bases = [b for b in _safe_str(summary.get("codeact_endpoint_bases")).split(",") if b]
    endpoint_differs = bool(
        dataset_endpoint
        and len(endpoint_bases) == 1
        and endpoint_bases[0] != dataset_endpoint
    )

    chart_insight = ""
    if charts and isinstance(charts[0], dict):
        chart_insight = _safe_str(charts[0].get("insight")).strip()

    return {
        "flag_missing_final_insight": bool(
            not summary.get("codeact_has_final_insight_marker", False) and not chart_insight
        ),
        "flag_hardcoded_chart_data": chart_prep_mode == "hardcoded_chart_data",
        "flag_requery_in_later_block": chart_prep_mode == "requery_or_reload",
        "flag_multi_source_provenance": int(summary.get("codeact_source_url_count", 0)) >= 2,
        "flag_endpoint_base_differs_from_dataset_endpoint": endpoint_differs,
        "flag_pie_percent_sum_off": pie_percent_sum_off,
    }


def add_codeact_explorer_columns(
    derived_df: pd.DataFrame,
    traces_by_id: dict[str, dict],
) -> pd.DataFrame:
    out = derived_df.copy()

    default_summary = extract_codeact_summary({})
    default_flags = compute_codeact_flags([], {})
    defaults: dict[str, Any] = {
        "codeact_retrieval_mode": "unknown",
        "codeact_chart_prep_mode": "unknown",
        "codeact_analysis_tags": "",
        **default_summary,
        **default_flags,
    }

    for col, val in defaults.items():
        if col not in out.columns:
            out[col] = val

    if out.empty:
        return out

    codeact_present = out.get("codeact_present")
    if not isinstance(codeact_present, pd.Series):
        codeact_present = pd.Series([False] * len(out), index=out.index, dtype=bool)
    mask = codeact_present.fillna(False).astype(bool)

    for idx in out.index[mask]:
        trace_id = _safe_str(out.at[idx, "trace_id"]).strip() if "trace_id" in out.columns else ""
        trace = traces_by_id.get(trace_id, {}) if trace_id else {}
        output_obj = trace.get("output") if isinstance(trace, dict) else {}
        output_obj = output_obj if isinstance(output_obj, dict) else {}

        timeline = extract_codeact_timeline(output_obj)
        decoded_code_blocks = [_safe_str(p.get("decoded")) for p in timeline if p.get("type") == "code_block"]

        summary = extract_codeact_summary(output_obj)
        flags = compute_codeact_flags(decoded_code_blocks, output_obj)
        retrieval_mode = classify_codeact_retrieval_mode(decoded_code_blocks, output_obj)
        chart_prep_mode = classify_codeact_chart_prep_mode(decoded_code_blocks)
        tags = extract_codeact_analysis_tags(decoded_code_blocks, output_obj)

        for col, val in summary.items():
            out.at[idx, col] = val
        for col, val in flags.items():
            out.at[idx, col] = val
        out.at[idx, "codeact_retrieval_mode"] = retrieval_mode
        out.at[idx, "codeact_chart_prep_mode"] = chart_prep_mode
        out.at[idx, "codeact_analysis_tags"] = tags

    return out
