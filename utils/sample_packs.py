"""Deterministic QA sample pack builders."""

from __future__ import annotations

import base64
import re
from typing import Any

import pandas as pd

from utils.eval_sampling import build_preset_mask


SAMPLE_PACKS = {
    "trend_failures": {
        "label": "Trend failures (exclude needs_user_input)",
        "description": "trend_over_time where completion_state is not complete_answer and not needs_user_input.",
        "preset_id": "trend_failures",
    },
    "trend_no_citation": {
        "label": "Trend missing citations",
        "description": "trend_over_time where struct_fail_reason includes no_citation (excluding needs_user_input).",
        "preset_id": "trend_no_citation",
    },
    "lookup_missing_dataset": {
        "label": "Lookup missing dataset",
        "description": "data_lookup where struct_fail_reason includes missing_dataset (excluding needs_user_input).",
        "preset_id": "lookup_missing_dataset",
    },
    "model_errors": {
        "label": "Model / output errors",
        "description": "answer_type in {model_error, missing_output, empty_or_short}.",
        "preset_id": "model_errors",
    },
    "needs_user_input": {
        "label": "Needs user input",
        "description": "completion_state == needs_user_input.",
        "preset_id": "needs_user_input",
    },
    "codeact_examples": {
        "label": "CodeAct examples (scored intents)",
        "description": "codeact_present==True for trend_over_time or data_lookup.",
        "preset_id": "codeact_examples",
    },
    "codeact_param_issues": {
        "label": "CodeAct parameter issues",
        "description": "codeact_present==True in scored intents with codeact_consistency_issue==True.",
        "preset_id": "codeact_param_issues",
    },
}

DEFAULT_SAMPLE_COLS = [
    "timestamp",
    "trace_id",
    "trace_version",
    "trace_release",
    "thread_id",
    "sessionId",
    "userId",
    "intent_primary",
    "dataset_name",
    "dataset_family",
    "dataset_struct",
    "aoi_name",
    "time_start",
    "time_end",
    "completion_state",
    "needs_user_input_reason",
    "struct_fail_reason",
    "answer_type",
    "codeact_present",
    "codeact_code_blocks_count",
    "codeact_uses_analytics_api",
    "codeact_consistency_issue",
    "codeact_consistency_reason",
    "codeact_code_snippet",
    "codeact_exec_snippet",
    "prompt",
    "response",
]


def sanitize_one_line(s: str) -> str:
    txt = str(s or "")
    txt = txt.replace("\r\n", " \u23ce ").replace("\n", " \u23ce ").replace("\r", " \u23ce ")
    txt = re.sub(r"\s+", " ", txt)
    return txt.strip()


def truncate(s: str, n: int) -> str:
    txt = str(s or "")
    if n <= 0:
        return ""
    if len(txt) <= n:
        return txt
    if n <= 3:
        return "." * n
    return txt[: n - 3] + "..."


def stable_sort(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "timestamp" in out.columns:
        by = ["timestamp"]
        ascending = [False]
        if "trace_id" in out.columns:
            by.append("trace_id")
            ascending.append(True)
        return out.sort_values(by=by, ascending=ascending, kind="mergesort")
    if "trace_id" in out.columns:
        return out.sort_values(by=["trace_id"], ascending=[True], kind="mergesort")
    return out


def build_sample_pack_df(
    derived: pd.DataFrame,
    pack_id: str,
    *,
    max_rows: int = 200,
    prompt_max_chars: int = 4000,
    response_max_chars: int = 4000,
) -> pd.DataFrame:
    if pack_id not in SAMPLE_PACKS:
        raise ValueError(f"Unknown sample pack: {pack_id}")

    preset_id = SAMPLE_PACKS[pack_id]["preset_id"]
    mask = build_preset_mask(derived, preset_id)
    df = derived.loc[mask].copy()
    df = stable_sort(df)
    df = df.head(int(max_rows)).copy()

    if "prompt" in df.columns:
        df["prompt"] = df["prompt"].apply(lambda x: truncate(sanitize_one_line(str(x or "")), int(prompt_max_chars)))
    if "response" in df.columns:
        df["response"] = df["response"].apply(
            lambda x: truncate(sanitize_one_line(str(x or "")), int(response_max_chars))
        )

    cols = [c for c in DEFAULT_SAMPLE_COLS if c in df.columns]
    return df.loc[:, cols].copy()


def _decode_maybe_base64(content: Any) -> str:
    raw = str(content or "")
    candidate = raw.strip()
    if not candidate:
        return ""

    looks_base64 = bool(re.fullmatch(r"[A-Za-z0-9+/=\s]+", candidate))
    if looks_base64:
        try:
            decoded = base64.b64decode(candidate, validate=False)
            text = decoded.decode("utf-8", errors="replace")
            printable = sum(1 for ch in text if ch.isprintable() or ch in "\n\r\t")
            ratio = printable / max(1, len(text))
            if ratio >= 0.85:
                return text
        except Exception:
            pass
    return raw


def add_codeact_snippets_for_pack(
    pack_df: pd.DataFrame,
    trace_by_id: dict[str, dict],
    *,
    max_chars: int = 500,
) -> pd.DataFrame:
    df = pack_df.copy()
    code_snippets: list[str] = []
    exec_snippets: list[str] = []

    trace_ids = df.get("trace_id", pd.Series([""] * len(df), index=df.index)).fillna("").astype(str)
    for trace_id in trace_ids:
        trace = trace_by_id.get(trace_id, {})
        output_obj = trace.get("output") if isinstance(trace, dict) else {}
        parts = output_obj.get("codeact_parts", []) if isinstance(output_obj, dict) else []

        code_txt = ""
        exec_txt = ""
        if isinstance(parts, list):
            for part in parts:
                if not isinstance(part, dict):
                    continue
                ptype = str(part.get("type") or "")
                content = _decode_maybe_base64(part.get("content", ""))
                if ptype == "code_block" and not code_txt:
                    code_txt = truncate(sanitize_one_line(content), int(max_chars))
                elif ptype == "execution_output" and not exec_txt:
                    exec_txt = truncate(sanitize_one_line(content), int(max_chars))
                if code_txt and exec_txt:
                    break

        code_snippets.append(code_txt)
        exec_snippets.append(exec_txt)

    df["codeact_code_snippet"] = code_snippets
    df["codeact_exec_snippet"] = exec_snippets
    return df
