"""Deterministic CodeAct QA/QC utilities."""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from typing import Any

import pandas as pd

from utils.codeact_utils import iter_decoded_codeact_parts


_SCORED_INTENTS = {"trend_over_time", "data_lookup"}
_ISO_DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_LONG_BLOB_RE = re.compile(r"\b[A-Za-z0-9+/]{24,}={0,2}\b")


def _to_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in {"1", "true", "yes", "y"}
    return bool(v)


def _to_str(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float) and pd.isna(v):
        return ""
    return str(v).strip()


def normalize_code_for_template(code: str) -> str:
    text = str(code or "").lower()
    text = re.sub(r"'''[\s\S]*?'''|\"\"\"[\s\S]*?\"\"\"", " <STR_BLOCK> ", text)
    text = re.sub(r"'(?:\\.|[^'\\])*'|\"(?:\\.|[^\"\\])*\"", " <STR> ", text)
    text = _ISO_DATE_RE.sub("<DATE>", text)
    text = re.sub(r"\b\d+(?:\.\d+)?\b", "<NUM>", text)
    text = _LONG_BLOB_RE.sub("<BLOB>", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s*([=,:()\[\]{}])\s*", r"\1", text)
    text = text.strip()
    return text


def compute_codeact_template_id(code_blocks: list[str], max_chars: int = 200_000) -> str:
    blocks = [str(b or "") for b in code_blocks if str(b or "").strip()]
    if not blocks:
        return ""
    combined = "\n\n---\n\n".join(blocks)
    normalized = normalize_code_for_template(combined)
    capped = normalized[: max(0, int(max_chars))]
    if not capped:
        return ""
    return hashlib.sha256(capped.encode("utf-8")).hexdigest()[:12]


def extract_code_param_signals(code_blocks: list[str]) -> dict:
    code = "\n".join(str(b or "") for b in code_blocks)
    lower = code.lower()

    key_patterns = {
        "start_date": r"\b(start_date|startdate)\b",
        "end_date": r"\b(end_date|enddate)\b",
        "dataset": r"\b(dataset|dataset_id|layer|layer_id|collection)\b",
        "aoi": r"\b(aoi|aoi_id|aoi_name|bbox|geometry|adm0|adm1|iso)\b",
        "bbox": r"\bbbox\b",
        "geometry": r"\bgeometry\b",
        "adm0": r"\badm0\b",
        "adm1": r"\badm1\b",
        "iso": r"\biso\b",
    }
    param_keys: set[str] = set()
    for key, pattern in key_patterns.items():
        if re.search(pattern, lower):
            param_keys.add(key)

    iso_dates_found = sorted(set(_ISO_DATE_RE.findall(code)))

    named_start = sorted(
        {
            d
            for d in re.findall(r"(?i)\b(?:start_date|startdate)\b\s*[:=]\s*['\"]?(\d{4}-\d{2}-\d{2})['\"]?", code)
        }
    )
    named_end = sorted(
        {
            d
            for d in re.findall(r"(?i)\b(?:end_date|enddate)\b\s*[:=]\s*['\"]?(\d{4}-\d{2}-\d{2})['\"]?", code)
        }
    )

    dataset_values = sorted(
        {
            val.strip()
            for _, val in re.findall(
                r"(?is)\b(dataset|dataset_id|layer|layer_id)\b\s*[:=]\s*['\"]([^'\"]{1,200})['\"]",
                code,
            )
            if val.strip()
        }
    )

    has_aoi_tokens = bool(re.search(r"(?i)\b(aoi|bbox|geometry|adm0|adm1|iso|country|region)\b", lower))
    codeact_has_analytics_host = "analytics.globalnaturewatch.org" in lower

    return {
        "param_keys": sorted(param_keys),
        "iso_dates_found": iso_dates_found,
        "start_dates_named": named_start,
        "end_dates_named": named_end,
        "dataset_values": dataset_values,
        "has_aoi_tokens": has_aoi_tokens,
        "codeact_has_analytics_host": codeact_has_analytics_host,
    }


def evaluate_param_consistency(row: pd.Series, signals: dict) -> dict:
    requires_time = _to_bool(row.get("requires_time_range", False))
    requires_dataset = _to_bool(row.get("requires_dataset", False))
    requires_aoi = _to_bool(row.get("requires_aoi", False))

    time_start = _to_str(row.get("time_start", ""))
    time_end = _to_str(row.get("time_end", ""))
    dataset_name = _to_str(row.get("dataset_name", ""))

    iso_dates = set(signals.get("iso_dates_found") or [])
    start_named = set(signals.get("start_dates_named") or [])
    end_named = set(signals.get("end_dates_named") or [])
    dataset_values = [str(v or "") for v in (signals.get("dataset_values") or [])]
    has_aoi_tokens = bool(signals.get("has_aoi_tokens", False))

    time_check = "unknown"
    dataset_check = "unknown"
    aoi_check = "unknown"
    reasons: list[str] = []

    if requires_time:
        if not time_start or not time_end:
            time_check = "unknown"
        else:
            has_any_time_signals = bool(iso_dates or start_named or end_named)
            matches_start = time_start in iso_dates or time_start in start_named
            matches_end = time_end in iso_dates or time_end in end_named
            if matches_start and matches_end:
                time_check = "ok"
            elif has_any_time_signals:
                time_check = "mismatch"
                reasons.append("time_mismatch")
            else:
                time_check = "missing"
                reasons.append("time_missing_in_code")

    if requires_dataset:
        if dataset_name:
            normalized_dataset = dataset_name.lower()
            if any(
                normalized_dataset == dv.lower()
                or normalized_dataset in dv.lower()
                or dv.lower() in normalized_dataset
                for dv in dataset_values
                if dv.strip()
            ):
                dataset_check = "ok"
            elif dataset_values:
                dataset_check = "unknown"
            elif any(k in (signals.get("param_keys") or []) for k in ["dataset"]):
                dataset_check = "unknown"
            else:
                dataset_check = "missing"
                reasons.append("dataset_missing_in_code")
        else:
            dataset_check = "unknown"

    if requires_aoi:
        if has_aoi_tokens:
            aoi_check = "ok"
        else:
            aoi_check = "missing"
            reasons.append("aoi_missing_in_code")

    issue = any(v in {"missing", "mismatch"} for v in [time_check, dataset_check, aoi_check])

    return {
        "codeact_time_check": time_check,
        "codeact_dataset_check": dataset_check,
        "codeact_aoi_check": aoi_check,
        "codeact_consistency_issue": bool(issue),
        "codeact_consistency_reason": "|".join(reasons),
        "codeact_param_keys_csv": ",".join(sorted(set(signals.get("param_keys") or []))),
        "codeact_iso_dates_csv": ",".join(sorted(set(signals.get("iso_dates_found") or []))),
    }


def add_codeact_qaqc_columns(derived_df: pd.DataFrame, traces_by_id: dict[str, dict]) -> pd.DataFrame:
    out = derived_df.copy()
    defaults = {
        "codeact_template_id": "",
        "codeact_param_keys_csv": "",
        "codeact_iso_dates_csv": "",
        "codeact_time_check": "unknown",
        "codeact_dataset_check": "unknown",
        "codeact_aoi_check": "unknown",
        "codeact_consistency_issue": False,
        "codeact_consistency_reason": "",
    }
    for col, val in defaults.items():
        if col not in out.columns:
            out[col] = val

    if out.empty:
        return out

    codeact_present = out.get("codeact_present")
    if not isinstance(codeact_present, pd.Series):
        codeact_present = pd.Series([False] * len(out), index=out.index, dtype=bool)
    codeact_present = codeact_present.fillna(False).astype(bool)

    for idx, row in out.iterrows():
        if not bool(codeact_present.get(idx, False)):
            for key, val in defaults.items():
                out.at[idx, key] = val
            continue

        trace_id = _to_str(row.get("trace_id", ""))
        trace = traces_by_id.get(trace_id, {})
        output_obj = trace.get("output", {}) if isinstance(trace, dict) else {}

        try:
            parts = iter_decoded_codeact_parts(output_obj)
        except Exception:
            parts = []

        code_blocks = [p.get("decoded", "") for p in parts if str(p.get("type", "")).lower() == "code_block"]
        template_id = compute_codeact_template_id(code_blocks)
        signals = extract_code_param_signals(code_blocks)
        consistency = evaluate_param_consistency(row, signals)

        out.at[idx, "codeact_template_id"] = template_id
        for key, val in consistency.items():
            out.at[idx, key] = val

    out["codeact_consistency_issue"] = out["codeact_consistency_issue"].fillna(False).astype(bool)
    return out


def build_codeact_template_rollups(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    codeact = df.copy()
    codeact_present = codeact.get("codeact_present", False)
    if not isinstance(codeact_present, pd.Series):
        codeact_present = pd.Series([False] * len(codeact), index=codeact.index, dtype=bool)
    codeact = codeact.loc[codeact_present.fillna(False).astype(bool)].copy()
    codeact = codeact.loc[codeact.get("codeact_template_id", "").fillna("").astype(str).str.strip() != ""].copy()
    if codeact.empty:
        return pd.DataFrame(), pd.DataFrame()

    total_codeact = len(codeact)

    if "timestamp" in codeact.columns:
        codeact["_ts"] = pd.to_datetime(codeact["timestamp"], utc=True, errors="coerce")
    else:
        codeact["_ts"] = pd.NaT

    trace_rows: list[dict[str, Any]] = []
    for row in codeact.itertuples(index=False):
        trace_rows.append(
            {
                "codeact_template_id": _to_str(getattr(row, "codeact_template_id", "")),
                "trace_id": _to_str(getattr(row, "trace_id", "")),
                "timestamp": _to_str(getattr(row, "timestamp", "")),
                "sessionId": _to_str(getattr(row, "sessionId", "")),
                "thread_id": _to_str(getattr(row, "thread_id", "")),
                "intent_primary": _to_str(getattr(row, "intent_primary", "")),
                "completion_state": _to_str(getattr(row, "completion_state", "")),
                "codeact_time_check": _to_str(getattr(row, "codeact_time_check", "unknown")),
                "codeact_dataset_check": _to_str(getattr(row, "codeact_dataset_check", "unknown")),
                "codeact_aoi_check": _to_str(getattr(row, "codeact_aoi_check", "unknown")),
                "codeact_consistency_issue": bool(getattr(row, "codeact_consistency_issue", False)),
                "codeact_consistency_reason": _to_str(getattr(row, "codeact_consistency_reason", "")),
            }
        )
    template_traces_df = pd.DataFrame(trace_rows)

    summaries: list[dict[str, Any]] = []
    for template_id, g in codeact.groupby("codeact_template_id", sort=False):
        intents = Counter(g.get("intent_primary", pd.Series(dtype=str)).fillna("").astype(str).tolist())
        intents_top3 = ", ".join([name for name, _ in intents.most_common(3) if name])

        scored = g[g.get("intent_primary", "").isin(_SCORED_INTENTS)] if "intent_primary" in g.columns else g.iloc[0:0]
        complete_answer_rate = (
            (scored.get("completion_state", "") == "complete_answer").mean() if len(scored) > 0 else 0.0
        )
        error_rate = (g.get("completion_state", "") == "error").mean() if "completion_state" in g.columns else 0.0
        needs_rate = (
            (g.get("completion_state", "") == "needs_user_input").mean() if "completion_state" in g.columns else 0.0
        )

        req_time = g[g.get("requires_time_range", False).fillna(False).astype(bool)] if "requires_time_range" in g.columns else g.iloc[0:0]
        time_issue_rate = (
            req_time.get("codeact_time_check", "").isin(["missing", "mismatch"]).mean() if len(req_time) else 0.0
        )

        req_dataset = g[g.get("requires_dataset", False).fillna(False).astype(bool)] if "requires_dataset" in g.columns else g.iloc[0:0]
        dataset_issue_rate = (
            (req_dataset.get("codeact_dataset_check", "") == "missing").mean() if len(req_dataset) else 0.0
        )

        req_aoi = g[g.get("requires_aoi", False).fillna(False).astype(bool)] if "requires_aoi" in g.columns else g.iloc[0:0]
        aoi_issue_rate = (req_aoi.get("codeact_aoi_check", "") == "missing").mean() if len(req_aoi) else 0.0

        analytics_rate = (
            g.get("codeact_uses_analytics_api", pd.Series([False] * len(g))).fillna(False).astype(bool).mean()
            if "codeact_uses_analytics_api" in g.columns
            else 0.0
        )

        sorted_group = g.sort_values(by=["_ts", "trace_id"], ascending=[False, False], kind="mergesort")
        representative_trace_id = _to_str(sorted_group.iloc[0].get("trace_id", "")) if not sorted_group.empty else ""

        summaries.append(
            {
                "codeact_template_id": template_id,
                "n_traces": int(len(g)),
                "share_of_codeact_traces": float(len(g) / max(1, total_codeact)),
                "intents_top3": intents_top3,
                "complete_answer_rate_scored_intents": float(complete_answer_rate),
                "error_rate": float(error_rate),
                "needs_user_input_rate": float(needs_rate),
                "time_issue_rate": float(time_issue_rate),
                "dataset_issue_rate": float(dataset_issue_rate),
                "aoi_issue_rate": float(aoi_issue_rate),
                "uses_analytics_api_rate": float(analytics_rate),
                "representative_trace_id": representative_trace_id,
            }
        )

    template_summary_df = pd.DataFrame(summaries).sort_values(
        by=["n_traces", "codeact_template_id"], ascending=[False, True], kind="mergesort"
    )

    return template_summary_df, template_traces_df
