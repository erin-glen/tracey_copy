"""Deterministic sampling helpers for Human Eval."""

from __future__ import annotations

import hashlib
import random
from typing import Any

import pandas as pd

SAMPLE_PRESETS = {
    "random": {
        "label": "Random (existing behavior)",
        "description": "No deterministic KPI mask; sample from all candidates.",
    },
    "balanced_scored_intents": {
        "label": "Balanced scored intents",
        "description": "Sample evenly across trend_over_time and data_lookup intents.",
    },
    "trend_failures": {
        "label": "Trend failures",
        "description": "trend_over_time traces excluding complete answers and needs_user_input.",
    },
    "trend_no_citation": {
        "label": "Trend no citation",
        "description": "trend_over_time traces with no_citation structural failure token.",
    },
    "lookup_missing_dataset": {
        "label": "Lookup missing dataset",
        "description": "data_lookup traces with missing_dataset structural failure token.",
    },
    "model_errors": {
        "label": "Model errors",
        "description": "Rows with model_error, missing_output, or empty_or_short answer types.",
    },
    "needs_user_input": {
        "label": "Needs user input",
        "description": "Rows classified as needs_user_input.",
    },
    "codeact_examples": {
        "label": "CodeAct examples",
        "description": "Rows with codeact present in scored intents.",
    },
    "codeact_param_issues": {
        "label": "CodeAct parameter issues",
        "description": "Scored-intent CodeAct traces with deterministic parameter consistency issues.",
    },
}


def _str_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df.columns:
        return df[col].fillna("").astype(str)
    return pd.Series([""] * len(df), index=df.index, dtype=str)


def _contains_token(series: pd.Series, token: str) -> pd.Series:
    tok = str(token or "").strip().lower()
    if not tok:
        return pd.Series([False] * len(series), index=series.index, dtype=bool)

    def _match(value: Any) -> bool:
        parts = [p.strip().lower() for p in str(value or "").split("|") if p is not None]
        return tok in parts

    return series.apply(_match).astype(bool)


def compute_thread_key(df: pd.DataFrame) -> pd.Series:
    """Compute deterministic thread key: thread_id -> sessionId -> trace_id."""
    thread_id = _str_series(df, "thread_id").str.strip()
    session_id = _str_series(df, "sessionId").str.strip()
    trace_id = _str_series(df, "trace_id").str.strip()

    key = thread_id.where(thread_id != "", session_id)
    key = key.where(key != "", trace_id)
    return key.astype(str)


def build_preset_mask(df: pd.DataFrame, preset_id: str) -> pd.Series:
    """Return deterministic preset mask over derived interactions DataFrame."""
    intent = _str_series(df, "intent_primary")
    completion = _str_series(df, "completion_state")
    struct_fail = _str_series(df, "struct_fail_reason")
    answer_type = _str_series(df, "answer_type")
    codeact_present = df.get("codeact_present", False)
    if not isinstance(codeact_present, pd.Series):
        codeact_present = pd.Series([bool(codeact_present)] * len(df), index=df.index, dtype=bool)
    codeact_present = codeact_present.fillna(False).astype(bool)

    if preset_id == "balanced_scored_intents":
        return intent.isin(["trend_over_time", "data_lookup"])
    if preset_id == "trend_failures":
        return (intent == "trend_over_time") & (~completion.isin(["complete_answer", "needs_user_input"]))
    if preset_id == "trend_no_citation":
        return (
            (intent == "trend_over_time")
            & (completion != "needs_user_input")
            & _contains_token(struct_fail, "no_citation")
        )
    if preset_id == "lookup_missing_dataset":
        return (
            (intent == "data_lookup")
            & (completion != "needs_user_input")
            & _contains_token(struct_fail, "missing_dataset")
        )
    if preset_id == "model_errors":
        return answer_type.isin(["model_error", "missing_output", "empty_or_short"])
    if preset_id == "needs_user_input":
        return completion == "needs_user_input"
    if preset_id == "codeact_examples":
        return codeact_present & intent.isin(["trend_over_time", "data_lookup"])
    if preset_id == "codeact_param_issues":
        consistency_issue = df.get("codeact_consistency_issue", False)
        if not isinstance(consistency_issue, pd.Series):
            consistency_issue = pd.Series([bool(consistency_issue)] * len(df), index=df.index, dtype=bool)
        consistency_issue = consistency_issue.fillna(False).astype(bool)
        return codeact_present & intent.isin(["trend_over_time", "data_lookup"]) & consistency_issue

    return pd.Series([True] * len(df), index=df.index, dtype=bool)


def _norm_filter_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str) and value == "All":
        return None
    return value


def apply_candidate_filters(
    df: pd.DataFrame,
    *,
    intent: Any = None,
    completion_state: Any = None,
    needs_reason: Any = None,
    dataset_family: Any = None,
    answer_type: Any = None,
    codeact_present: Any = None,
    trace_ids_allowlist: Any = None,
) -> pd.DataFrame:
    """Apply optional deterministic filters to candidate rows."""
    out = df.copy()
    mask = pd.Series([True] * len(out), index=out.index, dtype=bool)

    intent = _norm_filter_value(intent)
    completion_state = _norm_filter_value(completion_state)
    needs_reason = _norm_filter_value(needs_reason)
    dataset_family = _norm_filter_value(dataset_family)
    answer_type = _norm_filter_value(answer_type)
    codeact_present = _norm_filter_value(codeact_present)

    if intent is not None:
        mask &= _str_series(out, "intent_primary") == str(intent)
    if completion_state is not None:
        mask &= _str_series(out, "completion_state") == str(completion_state)
    if needs_reason is not None:
        mask &= _str_series(out, "needs_user_input_reason") == str(needs_reason)
    if dataset_family is not None:
        mask &= _str_series(out, "dataset_family") == str(dataset_family)
    if answer_type is not None:
        mask &= _str_series(out, "answer_type") == str(answer_type)

    codeact_series = out.get("codeact_present")
    if not isinstance(codeact_series, pd.Series):
        codeact_series = pd.Series([False] * len(out), index=out.index, dtype=bool)
    codeact_series = codeact_series.fillna(False).astype(bool)

    if codeact_present == "Yes":
        mask &= codeact_series
    elif codeact_present == "No":
        mask &= ~codeact_series

    if trace_ids_allowlist is not None:
        allow = {str(t).strip() for t in trace_ids_allowlist if str(t).strip()}
        if allow:
            mask &= _str_series(out, "trace_id").isin(allow)
        else:
            mask &= pd.Series([False] * len(out), index=out.index, dtype=bool)

    return out.loc[mask].copy()


def _stable_hash(group_val: Any) -> int:
    digest = hashlib.md5(str(group_val).encode("utf-8")).hexdigest()[:8]
    return int(digest, 16)


def sample_trace_ids(
    df: pd.DataFrame,
    n: int,
    seed: int,
    *,
    stratify_col: str | None = None,
    one_per_thread: bool = False,
) -> list[str]:
    """Deterministically sample trace IDs with optional balanced stratification."""
    if n <= 0 or df.empty:
        return []

    out = df.copy()
    trace_ids = _str_series(out, "trace_id").str.strip()
    out = out.loc[trace_ids != ""].copy()
    if out.empty:
        return []

    out["trace_id"] = _str_series(out, "trace_id").str.strip()

    if one_per_thread:
        out["_thread_key"] = compute_thread_key(out)
        out = out.sort_values("trace_id", kind="mergesort").drop_duplicates(subset=["_thread_key"], keep="first")

    out = out.sort_values("trace_id", kind="mergesort")
    n = min(int(n), len(out))
    if n <= 0:
        return []

    if not stratify_col or stratify_col not in out.columns:
        ids = out["trace_id"].tolist()
        rng = random.Random(int(seed))
        rng.shuffle(ids)
        return ids[:n]

    groups: dict[str, list[str]] = {}
    for group_val, group_df in out.groupby(stratify_col, dropna=False, sort=False):
        gname = str(group_val)
        ids = sorted(group_df["trace_id"].astype(str).tolist())
        grng = random.Random(int(seed) + _stable_hash(gname))
        grng.shuffle(ids)
        groups[gname] = ids

    group_names = sorted(groups.keys())
    if not group_names:
        return []

    base = n // len(group_names)
    remainder = n % len(group_names)

    selected: list[str] = []
    selected_set: set[str] = set()
    deficits = 0

    for i, name in enumerate(group_names):
        want = base + (1 if i < remainder else 0)
        pool = groups[name]
        take = min(want, len(pool))
        picked = pool[:take]
        selected.extend(picked)
        selected_set.update(picked)
        deficits += max(0, want - take)

    if deficits > 0:
        remaining_ids = [tid for tid in out["trace_id"].tolist() if tid not in selected_set]
        rrng = random.Random(int(seed) + 99991)
        rrng.shuffle(remaining_ids)
        fill = remaining_ids[:deficits]
        selected.extend(fill)
        selected_set.update(fill)

    return selected[:n]
