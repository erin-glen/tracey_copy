"""Deterministic content/structural KPI analysis over in-memory Langfuse traces."""

from __future__ import annotations

import base64
import re
from collections import Counter
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from utils.trace_parsing import (
    current_human_prompt,
    current_turn_ai_message,
    normalize_trace_format,
    parse_trace_dt,
    slice_output_to_current_turn,
)

SCORED_INTENTS = {"trend_over_time", "data_lookup"}
POSITIVE_ACK_PATTERNS = [
    "thanks",
    "thank you",
    "great",
    "perfect",
    "awesome",
    "that helps",
    "sounds good",
]
NEGATIVE_ACK_PATTERNS = [
    "not what i asked",
    "doesn't answer",
    "wrong",
    "try again",
    "no",
    "huh",
]


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, dict):
        for key in ("text", "content", "value"):
            val = content.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
        return ""
    if isinstance(content, list):
        bits: list[str] = []
        for part in content:
            txt = _content_to_text(part)
            if txt:
                bits.append(txt)
        return "\n".join(bits).strip()
    return ""


def _message_role(msg: Any) -> str:
    if not isinstance(msg, dict):
        return ""
    return str(msg.get("type") or msg.get("role") or "").strip().lower()


def _is_assistant(msg: Any) -> bool:
    return _message_role(msg) in {"assistant", "ai"}


def _is_user(msg: Any) -> bool:
    return _message_role(msg) in {"user", "human"}


def _find_first_key(obj: Any, keys: tuple[str, ...]) -> Any:
    if isinstance(obj, dict):
        for key in keys:
            if key in obj and obj[key] not in (None, "", []):
                return obj[key]
        for value in obj.values():
            found = _find_first_key(value, keys)
            if found not in (None, "", []):
                return found
    elif isinstance(obj, list):
        for item in obj:
            found = _find_first_key(item, keys)
            if found not in (None, "", []):
                return found
    return None


def _extract_struct_flags(output_obj: dict[str, Any] | None) -> dict[str, Any]:
    output_obj = output_obj or {}
    aoi = _find_first_key(output_obj, ("aoi", "selected_aoi", "selectedAOI"))
    aoi_selected = isinstance(aoi, dict) and bool(
        str(aoi.get("name") or "").strip() or str(aoi.get("id") or "").strip()
    )
    aoi_candidates = bool(_find_first_key(output_obj, ("aoi_candidates", "candidates", "aois")))

    time_start = _find_first_key(output_obj, ("time_start", "start", "start_date", "from"))
    time_end = _find_first_key(output_obj, ("time_end", "end", "end_date", "to"))
    time_range_struct = bool(time_start and time_end)

    dataset_name = _extract_dataset_name(output_obj)
    dataset_struct = bool(dataset_name)

    citations = _find_first_key(output_obj, ("citations", "sources", "sourceUrls", "references"))
    citations_struct = bool(citations)

    aoi_name = ""
    aoi_type = ""
    if isinstance(aoi, dict):
        aoi_name = str(aoi.get("name") or aoi.get("title") or "").strip()
        aoi_type = str(aoi.get("type") or "").strip()

    return {
        "aoi_selected_struct": aoi_selected,
        "aoi_candidates_struct": aoi_candidates,
        "time_range_struct": time_range_struct,
        "dataset_struct": dataset_struct,
        "citations_struct": citations_struct,
        "dataset_name": dataset_name,
        "aoi_name": aoi_name,
        "aoi_type": aoi_type,
        "time_start": str(time_start or "").strip(),
        "time_end": str(time_end or "").strip(),
    }


def _extract_dataset_name(output_obj: dict[str, Any]) -> str:
    blocks = [
        _find_first_key(output_obj, ("dataset_info", "datasetInfo")),
        _find_first_key(output_obj, ("result", "data", "payload")),
        output_obj,
    ]
    key_candidates = (
        "dataset_name",
        "datasetName",
        "dataset",
        "datasets",
        "layer",
        "layers",
        "layer_name",
        "layerName",
        "collection",
        "name",
        "title",
    )
    for block in blocks:
        found = _find_first_key(block, key_candidates)
        if isinstance(found, str) and found.strip():
            return found.strip()
        if isinstance(found, list):
            for item in found:
                if isinstance(item, str) and item.strip():
                    return item.strip()
                if isinstance(item, dict):
                    name = _find_first_key(item, ("name", "title", "dataset_name", "layer_name"))
                    if isinstance(name, str) and name.strip():
                        return name.strip()
    return ""


def _text_flags(prompt: str, response: str) -> dict[str, bool]:
    p = (prompt or "").lower()
    r = (response or "").lower()
    combined = f"{p}\n{r}"
    return {
        "aoi_text": any(k in combined for k in ["aoi", "region", "country", "city", "brazil", "india"]),
        "time_text": any(k in combined for k in ["year", "month", "time", "from", "to", "between", "trend"]),
        "dataset_text": any(k in combined for k in ["dataset", "layer", "collection", "tree cover", "population"]),
        "citations_text": any(k in combined for k in ["source", "citation", "http://", "https://"]),
    }


def _classify_intent(prompt: str) -> tuple[str, str]:
    p = (prompt or "").lower()
    if any(k in p for k in ["trend", "over time", "change over", "yearly", "monthly"]):
        return "trend_over_time", ""
    if any(k in p for k in ["show", "lookup", "find", "what is", "tree cover", "dataset"]):
        return "data_lookup", ""
    if any(k in p for k in ["how", "can you", "capability", "what can you do"]):
        return "conceptual_or_capability", ""
    return "other", ""


def _infer_requires(intent: str, prompt: str) -> dict[str, bool]:
    p = (prompt or "").lower()
    requires_data = intent in SCORED_INTENTS
    requires_aoi = intent in SCORED_INTENTS or any(k in p for k in ["in ", "for ", "country", "region"])
    requires_time = intent == "trend_over_time" or any(k in p for k in ["over time", "between", "from", "to", "year"])
    requires_dataset = intent in SCORED_INTENTS or "dataset" in p or "tree cover" in p
    return {
        "requires_data": requires_data,
        "requires_aoi": requires_aoi,
        "requires_time_range": requires_time,
        "requires_dataset": requires_dataset,
    }


def _answer_type(response: str, response_missing: bool, output_json_ok: bool) -> str:
    t = (response or "").strip().lower()
    if response_missing:
        return "missing_output"
    if any(k in t for k in ["error", "exception", "failed", "unable"]):
        return "model_error"
    if any(k in t for k in ["no data", "not available", "could not find"]):
        return "no_data"
    if len(t) < 8:
        return "empty_or_short"
    if not output_json_ok:
        return "text_only"
    return "answer"


def _needs_user_input(response: str, requires: dict[str, bool], struct: dict[str, Any]) -> tuple[bool, str]:
    r = (response or "").lower()
    missing: list[str] = []
    if requires["requires_aoi"] and not struct["aoi_selected_struct"]:
        missing.append("aoi")
    if requires["requires_time_range"] and not struct["time_range_struct"]:
        missing.append("time")
    if requires["requires_dataset"] and not struct["dataset_struct"]:
        missing.append("dataset")
    asks_for_more = any(k in r for k in ["please specify", "can you clarify", "which", "what time range", "provide"]) or "?" in r
    if missing and asks_for_more:
        if len(missing) > 1:
            return True, "multiple_missing"
        only = missing[0]
        return True, f"missing_{only}"
    return False, ""


def _metric_sanity_fail(response: str) -> bool:
    r = (response or "").lower()
    has_pct = bool(re.search(r"\b\d+(?:\.\d+)?%", r))
    impossible_pct = bool(re.search(r"\b(1\d{2,}|\d{4,})%", r))
    return has_pct and impossible_pct


def _classify_dataset_family(dataset_name: str) -> str:
    d = (dataset_name or "").lower()
    if not d:
        return "unknown"
    if "tree" in d or "forest" in d:
        return "forest"
    if "climate" in d or "temperature" in d:
        return "climate"
    if "population" in d or "demograph" in d:
        return "population"
    return "other"


def _parse_time_window_days(start: str, end: str) -> float | None:
    if not start or not end:
        return None
    try:
        s = pd.to_datetime(start, utc=True, errors="coerce")
        e = pd.to_datetime(end, utc=True, errors="coerce")
        if pd.isna(s) or pd.isna(e):
            return None
        return float((e - s).days)
    except Exception:
        return None


def _extract_codeact(output_obj: Any) -> dict[str, Any]:
    parts = _find_first_key(output_obj, ("codeact_parts", "codeActParts", "parts"))
    if not isinstance(parts, list):
        return {
            "codeact_present": False,
            "codeact_parts_count": 0,
            "codeact_code_blocks_count": 0,
            "codeact_exec_outputs_count": 0,
            "codeact_uses_analytics_api": False,
            "codeact_decoded_chars_total": 0,
        }

    code_blocks = 0
    exec_outputs = 0
    uses_analytics_api = False
    decoded_chars_total = 0
    for part in parts:
        if not isinstance(part, dict):
            continue
        ptype = str(part.get("type") or "").lower()
        if "code" in ptype:
            code_blocks += 1
        if "exec" in ptype or "output" in ptype:
            exec_outputs += 1
        payload = str(part.get("content") or part.get("text") or "")
        if "analytics" in payload.lower() or "/v1/query" in payload.lower():
            uses_analytics_api = True
        b64 = part.get("base64")
        if isinstance(b64, str) and b64:
            try:
                decoded_chars_total += len(base64.b64decode(b64).decode("utf-8", errors="ignore"))
            except Exception:
                pass

    return {
        "codeact_present": True,
        "codeact_parts_count": len(parts),
        "codeact_code_blocks_count": code_blocks,
        "codeact_exec_outputs_count": exec_outputs,
        "codeact_uses_analytics_api": uses_analytics_api,
        "codeact_decoded_chars_total": decoded_chars_total,
    }


def _completion_state(
    intent: str,
    answer_type: str,
    needs_user_input: bool,
    struct: dict[str, Any],
    requires: dict[str, bool],
) -> tuple[str, bool, bool, str]:
    reasons: list[str] = []
    struct_good_trend = False
    struct_good_lookup = False

    if intent in SCORED_INTENTS:
        if requires["requires_aoi"] and not struct["aoi_selected_struct"]:
            reasons.append("missing_aoi_struct")
        if requires["requires_time_range"] and not struct["time_range_struct"]:
            reasons.append("missing_time_struct")
        if requires["requires_dataset"] and not struct["dataset_struct"]:
            reasons.append("missing_dataset_struct")

    if intent == "trend_over_time":
        struct_good_trend = len(reasons) == 0
    if intent == "data_lookup":
        struct_good_lookup = len(reasons) == 0

    if answer_type in {"missing_output", "model_error"}:
        return "error", struct_good_trend, struct_good_lookup, "|".join(reasons)
    if answer_type == "no_data":
        return "no_data", struct_good_trend, struct_good_lookup, "|".join(reasons)
    if needs_user_input:
        return "needs_user_input", struct_good_trend, struct_good_lookup, "|".join(reasons)
    if intent in SCORED_INTENTS and reasons:
        return "incomplete_answer", struct_good_trend, struct_good_lookup, "|".join(reasons)
    if answer_type in {"answer", "text_only", "empty_or_short"}:
        return "complete_answer", struct_good_trend, struct_good_lookup, "|".join(reasons)
    return "other", struct_good_trend, struct_good_lookup, "|".join(reasons)


def infer_conversation_outcomes(derived: pd.DataFrame) -> pd.Series:
    if derived.empty:
        return pd.Series(dtype="string")

    df = derived.copy()
    group_keys = []
    for row in df.itertuples(index=False):
        if getattr(row, "thread_id", ""):
            group_keys.append(f"thread:{row.thread_id}")
        elif getattr(row, "sessionId", ""):
            group_keys.append(f"session:{row.sessionId}")
        else:
            group_keys.append(f"trace:{row.trace_id}")
    df["_group"] = group_keys
    df = df.sort_values("timestamp", na_position="last")

    outcomes = pd.Series(["unknown"] * len(df), index=df.index, dtype="string")
    for _, g in df.groupby("_group", sort=False):
        prompts = g["prompt"].fillna("").astype(str).str.lower().tolist()
        for i, idx in enumerate(g.index):
            p = prompts[i]
            if any(k in p for k in POSITIVE_ACK_PATTERNS):
                outcomes.loc[idx] = "success"
            elif any(k in p for k in NEGATIVE_ACK_PATTERNS) or p in {"ok", "k", "?"}:
                outcomes.loc[idx] = "clarification_needed"
            elif i > 0 and p and p == prompts[i - 1]:
                outcomes.loc[idx] = "repeat_question"
            elif i < len(g) - 1:
                outcomes.loc[idx] = "clarification_needed"
            else:
                outcomes.loc[idx] = "unknown"
    return outcomes.sort_index()


def compute_derived_interactions(traces: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for raw in traces:
        trace = normalize_trace_format(raw)
        prompt = current_human_prompt(trace)
        if not prompt.strip():
            continue

        out_msgs = slice_output_to_current_turn(trace)
        response = current_turn_ai_message(trace)

        response_missing = True
        if isinstance(out_msgs, list):
            for m in reversed(out_msgs):
                if _is_assistant(m) and _content_to_text(m.get("content")):
                    response_missing = False
                    break
        if response_missing:
            response = ""

        output_json_ok = isinstance(trace.get("output"), dict)
        output_obj = trace.get("output") if isinstance(trace.get("output"), dict) else {}

        struct = _extract_struct_flags(output_obj)
        text_flags = _text_flags(prompt, response)
        intent_primary, intent_secondary = _classify_intent(prompt)
        requires = _infer_requires(intent_primary, prompt)
        answer_type = _answer_type(response, response_missing, output_json_ok)
        needs_ui, needs_reason = _needs_user_input(response, requires, struct)
        metric_fail = _metric_sanity_fail(response)
        completion_state, struct_good_trend, struct_good_lookup, struct_fail_reason = _completion_state(
            intent_primary, answer_type, needs_ui, struct, requires
        )
        dataset_name = struct.get("dataset_name", "")
        dataset_family = _classify_dataset_family(dataset_name)
        time_window_days = _parse_time_window_days(struct.get("time_start", ""), struct.get("time_end", ""))
        codeact = _extract_codeact(output_obj)

        metadata = trace.get("metadata") if isinstance(trace.get("metadata"), dict) else {}
        ts = parse_trace_dt(trace)
        timestamp = ts.astimezone(timezone.utc) if isinstance(ts, datetime) else None

        rows.append(
            {
                "trace_id": str(trace.get("id") or ""),
                "timestamp": timestamp,
                "sessionId": str(trace.get("sessionId") or ""),
                "thread_id": str(metadata.get("thread_id") or metadata.get("threadId") or ""),
                "userId": str(trace.get("userId") or ""),
                "level": str(trace.get("level") or ""),
                "latency": trace.get("latency"),
                "input_tokens": trace.get("inputTokens"),
                "output_tokens": trace.get("outputTokens"),
                "total_tokens": trace.get("totalTokens"),
                "prompt": prompt,
                "response": response,
                "response_missing": bool(response_missing),
                "output_json_ok": bool(output_json_ok),
                "intent_primary": intent_primary,
                "intent_secondary": intent_secondary,
                "complexity_bucket": "simple" if len(prompt.split()) < 12 else "complex",
                "geo_scope": "regional" if any(k in prompt.lower() for k in ["country", "region", "state", "city", "brazil"]) else "unspecified",
                **requires,
                **{k: struct[k] for k in ["aoi_selected_struct", "aoi_candidates_struct", "time_range_struct", "dataset_struct", "citations_struct"]},
                **text_flags,
                "dataset_name": dataset_name,
                "dataset_family": dataset_family,
                "dataset_identifiable": bool(dataset_name),
                "aoi_type": struct.get("aoi_type", ""),
                "aoi_name": struct.get("aoi_name", ""),
                "time_start": struct.get("time_start", ""),
                "time_end": struct.get("time_end", ""),
                "time_window_days": time_window_days,
                "answer_type": answer_type,
                "metric_sanity_fail": bool(metric_fail),
                "needs_user_input": bool(needs_ui),
                "needs_user_input_reason": needs_reason,
                "completion_state": completion_state,
                "struct_good_trend": bool(struct_good_trend),
                "struct_good_lookup": bool(struct_good_lookup),
                "struct_fail_reason": struct_fail_reason,
                "conversation_outcome": "unknown",
                **codeact,
            }
        )

    derived = pd.DataFrame(rows)
    if derived.empty:
        return derived
    derived["conversation_outcome"] = infer_conversation_outcomes(derived)
    return derived


def _pct(n: float, d: float) -> float:
    if not d:
        return 0.0
    return float(n) / float(d)


def summarize_content(derived: pd.DataFrame, timestamp_col: str = "timestamp") -> dict[str, Any]:
    rows = int(len(derived))
    unique_users = int(derived["userId"].replace("", pd.NA).dropna().nunique()) if "userId" in derived.columns else 0

    window = {"start": None, "end": None}
    if timestamp_col in derived.columns and rows:
        ts = pd.to_datetime(derived[timestamp_col], utc=True, errors="coerce").dropna()
        if len(ts):
            window = {"start": ts.min().isoformat(), "end": ts.max().isoformat()}

    answer_counts = derived["answer_type"].value_counts(dropna=False).to_dict() if rows else {}
    completion_counts = derived["completion_state"].value_counts(dropna=False).to_dict() if rows else {}
    reason_counts = derived["needs_user_input_reason"].replace("", pd.NA).dropna().value_counts().to_dict() if rows else {}
    has_citations = (derived["citations_struct"].fillna(False) | derived["citations_text"].fillna(False)) if rows else pd.Series(dtype=bool)

    scored = derived[derived["intent_primary"].isin(SCORED_INTENTS)] if rows else derived
    data_intents = derived[derived["requires_data"] == True] if rows else derived

    thread_col = derived["thread_id"].fillna("") if rows and "thread_id" in derived.columns else pd.Series(dtype=str)
    if rows:
        thread_key = thread_col.where(thread_col != "", derived["sessionId"].fillna(""))
        tmp = derived.assign(_thread=thread_key)
        tmp = tmp[tmp["_thread"] != ""].sort_values("timestamp", na_position="last")
        ended_nui = 0
        ended_err = 0
        total_threads = 0
        for _, g in tmp.groupby("_thread"):
            total_threads += 1
            last = g.iloc[-1]
            if str(last.get("completion_state")) == "needs_user_input":
                ended_nui += 1
            if str(last.get("completion_state")) == "error":
                ended_err += 1
    else:
        ended_nui = ended_err = total_threads = 0

    intent_summary: dict[str, dict[str, Any]] = {}
    for intent in sorted(SCORED_INTENTS):
        subset = derived[derived["intent_primary"] == intent] if rows else derived
        c = len(subset)
        intent_summary[intent] = {
            "count": int(c),
            "share_of_total": _pct(c, rows),
            "complete_answer_rate": _pct((subset["completion_state"] == "complete_answer").sum(), c),
            "needs_user_input_rate": _pct((subset["completion_state"] == "needs_user_input").sum(), c),
            "error_rate": _pct((subset["completion_state"] == "error").sum(), c),
            "structural_complete_rate": _pct(
                subset["struct_good_trend"].sum() if intent == "trend_over_time" else subset["struct_good_lookup"].sum(),
                c,
            ),
        }

    struct_subset = scored
    fail_reasons = Counter()
    if len(struct_subset):
        for val in struct_subset["struct_fail_reason"].fillna(""):
            for token in [x for x in str(val).split("|") if x.strip()]:
                fail_reasons[token] += 1

    dataset_family_summary: dict[str, dict[str, Any]] = {}
    for family, g in data_intents.groupby("dataset_family") if len(data_intents) else []:
        c = len(g)
        dataset_family_summary[str(family)] = {
            "count_data_intents": int(c),
            "complete_answer_rate_data_intents": _pct((g["completion_state"] == "complete_answer").sum(), c),
            "needs_user_input_rate_data_intents": _pct((g["completion_state"] == "needs_user_input").sum(), c),
            "error_rate_data_intents": _pct((g["completion_state"] == "error").sum(), c),
            "codeact_present_rate_data_intents": _pct(g["codeact_present"].fillna(False).sum(), c),
        }

    global_quality = {
        "answer_type_counts": answer_counts,
        "answer_type_percentages": {k: _pct(v, rows) for k, v in answer_counts.items()},
        "completion_state_counts": completion_counts,
        "completion_state_rates": {k: _pct(v, rows) for k, v in completion_counts.items()},
        "needs_user_input_reason_counts": reason_counts,
        "metric_sanity_fail_rate": _pct(derived["metric_sanity_fail"].fillna(False).sum(), rows),
        "has_citations_rate": _pct(has_citations.sum(), rows),
        "dataset_identifiable_rate_scored_intents": _pct(scored["dataset_identifiable"].fillna(False).sum(), len(scored)),
        "codeact_present_rate": _pct(derived["codeact_present"].fillna(False).sum(), rows),
        "codeact_present_rate_scored_intents": _pct(scored["codeact_present"].fillna(False).sum(), len(scored)),
        "threads_ended_after_needs_user_input_rate": _pct(ended_nui, total_threads),
        "threads_ended_after_error_rate": _pct(ended_err, total_threads),
    }

    kpis = {
        "complete_answer_rate_scored_intents": _pct((scored["completion_state"] == "complete_answer").sum(), len(scored)),
        "needs_user_input_rate_scored_intents": _pct((scored["completion_state"] == "needs_user_input").sum(), len(scored)),
        "error_rate_scored_intents": _pct((scored["completion_state"] == "error").sum(), len(scored)),
        "global_dataset_identifiable_rate_scored_intents": global_quality["dataset_identifiable_rate_scored_intents"],
        "global_citation_rate": global_quality["has_citations_rate"],
        "threads_ended_after_needs_user_input_rate": global_quality["threads_ended_after_needs_user_input_rate"],
    }

    return {
        "rows": rows,
        "unique_users": unique_users,
        "window_utc": window,
        "kpis": kpis,
        "global_quality": global_quality,
        "intent_summary": intent_summary,
        "struct_outcome_summary": {
            "total": int(len(struct_subset)),
            "complete": int((struct_subset["completion_state"] == "complete_answer").sum()) if len(struct_subset) else 0,
            "needs_user_input": int((struct_subset["completion_state"] == "needs_user_input").sum()) if len(struct_subset) else 0,
            "errors": int((struct_subset["completion_state"] == "error").sum()) if len(struct_subset) else 0,
            "no_data": int((struct_subset["completion_state"] == "no_data").sum()) if len(struct_subset) else 0,
            "failures_excluding_needs_user_input": int(
                (~struct_subset["completion_state"].isin(["complete_answer", "needs_user_input"])).sum()
            ) if len(struct_subset) else 0,
            "needs_user_input_reasons": reason_counts,
            "failure_reasons": dict(fail_reasons),
        },
        "dataset_family_summary": dataset_family_summary,
    }


def build_content_slices(derived: pd.DataFrame) -> pd.DataFrame:
    if derived.empty:
        return pd.DataFrame(
            columns=[
                "intent_primary",
                "complexity_bucket",
                "count",
                "complete_answer_rate",
                "needs_user_input_rate",
                "error_rate",
                "metric_sanity_fail_rate",
                "has_citations_rate",
            ]
        )

    df = derived.copy()
    df["has_citations"] = df["citations_struct"].fillna(False) | df["citations_text"].fillna(False)
    grouped = (
        df.groupby(["intent_primary", "complexity_bucket"], dropna=False)
        .agg(
            count=("trace_id", "count"),
            complete_answer_rate=("completion_state", lambda s: _pct((s == "complete_answer").sum(), len(s))),
            needs_user_input_rate=("completion_state", lambda s: _pct((s == "needs_user_input").sum(), len(s))),
            error_rate=("completion_state", lambda s: _pct((s == "error").sum(), len(s))),
            metric_sanity_fail_rate=("metric_sanity_fail", lambda s: _pct(s.fillna(False).sum(), len(s))),
            has_citations_rate=("has_citations", lambda s: _pct(s.fillna(False).sum(), len(s))),
        )
        .reset_index()
    )
    return grouped.sort_values(["intent_primary", "complexity_bucket"]).reset_index(drop=True)
