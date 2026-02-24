"""Classification specification tests for Content KPIs.

These tests are intentionally written as an executable *spec by example*.
They document and lock in the deterministic rules that produce the derived
labels used by the Content KPIs dashboard:

  - intent_primary (scored intents)
  - requires_* flags (what the turn *needs* structurally)
  - struct flags (what the tool output actually contains)
  - answer_type (coarse response/trace outcome)
  - needs_user_input (+ needs_user_input_reason)
  - completion_state (+ struct_fail_reason)

If you change the classification logic in utils/content_kpis.py, you should:

  1) update these tests to reflect the new spec, and
  2) update the glossary/spec documentation so researchers can reproduce results.

The goal is not to test every regex edge case; it's to enforce the *priority
order* and the highest-impact decision rules.
"""

from __future__ import annotations

from datetime import datetime, timezone
import importlib.util
import sys
import types
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import pytest


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec for {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


# NOTE: The repo's `utils/__init__.py` imports Streamlit, which makes "pure-data" tests
# fail in minimal environments. For executable specs we instead load the needed modules
# directly from file paths and register them under a lightweight `utils.*` namespace.
ROOT = Path(__file__).resolve().parents[1]
if "utils" not in sys.modules:
    pkg = types.ModuleType("utils")
    pkg.__path__ = [str(ROOT / "utils")]
    sys.modules["utils"] = pkg

_load_module("utils.trace_parsing", ROOT / "utils" / "trace_parsing.py")
_load_module("utils.codeact_utils", ROOT / "utils" / "codeact_utils.py")
ck = _load_module("utils.content_kpis", ROOT / "utils" / "content_kpis.py")


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def make_trace(
    *,
    trace_id: str,
    prompt: str,
    response: str,
    output_obj: Dict[str, Any] | None = None,
    created_at: str | None = None,
    session_id: str = "sess-1",
    user_id: str = "user-1",
    thread_id: str = "",
    level: str = "",
    error_count: int | None = None,
) -> Dict[str, Any]:
    """Build a minimal Langfuse-like trace dict accepted by compute_derived_interactions()."""
    out = dict(output_obj or {})
    out.setdefault("messages", [{"type": "assistant", "content": response}])

    meta: Dict[str, Any] = {}
    if thread_id:
        meta["thread_id"] = thread_id

    return {
        "id": trace_id,
        "createdAt": created_at or _iso(datetime(2026, 2, 1, tzinfo=timezone.utc)),
        "sessionId": session_id,
        "userId": user_id,
        "level": level,
        "errorCount": error_count,
        "metadata": meta,
        "input": {"messages": [{"type": "user", "content": prompt}]},
        "output": out,
    }


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        ({"response": "anything", "response_missing": True, "output_json_ok": True}, "missing_output"),
        (
            {
                "response": "This is a sufficiently long response to avoid empty_or_short.",
                "response_missing": False,
                "output_json_ok": True,
                "level": "ERROR",
            },
            "model_error",
        ),
        ({"response": "Too short", "response_missing": False, "output_json_ok": True}, "empty_or_short"),
        (
            {
                "response": "No data available for this region in GNW at the moment.",
                "response_missing": False,
                "output_json_ok": True,
            },
            "no_data",
        ),
        (
            {
                "response": "As an AI language model, I can't access real time systems.",
                "response_missing": False,
                "output_json_ok": True,
            },
            "fallback",
        ),
        (
            {
                "response": "Tree cover loss in Brazil in 2021 was approximately 1.2M ha.",
                "response_missing": False,
                "output_json_ok": True,
            },
            "answer",
        ),
    ],
)
def test_answer_type_priority_order(kwargs: dict[str, Any], expected: str) -> None:
    assert ck._answer_type(**kwargs) == expected


def test_needs_user_input_can_trigger_from_missing_required_struct_even_without_explicit_ask() -> None:
    requires = {
        "requires_data": True,
        "requires_aoi": False,
        "requires_time_range": False,
        "requires_dataset": True,
    }
    struct = {
        "aoi_selected_struct": True,
        "time_range_struct": True,
        "dataset_struct": False,
        "aoi_candidates_struct": False,
        "aoi_options_unique_count": 0,
    }
    needs, reason = ck._needs_user_input(response="Here are your results.", requires=requires, struct=struct)
    assert needs is True
    assert reason == "missing_dataset"


def test_needs_user_input_detects_aoi_disambiguation_even_if_an_aoi_is_selected() -> None:
    requires = {
        "requires_data": True,
        "requires_aoi": True,
        "requires_time_range": True,
        "requires_dataset": True,
    }
    struct = {
        "aoi_selected_struct": True,
        "aoi_candidates_struct": True,
        "aoi_options_unique_count": 3,
        "time_range_struct": True,
        "dataset_struct": True,
    }
    needs, reason = ck._needs_user_input(
        response="I found multiple location matches. Which one did you mean?",
        requires=requires,
        struct=struct,
    )
    assert needs is True
    assert reason == "missing_aoi"


def test_needs_user_input_reason_multiple_missing_when_more_than_one_field_missing() -> None:
    requires = {
        "requires_data": True,
        "requires_aoi": False,
        "requires_time_range": True,
        "requires_dataset": True,
    }
    struct = {
        "aoi_selected_struct": True,
        "time_range_struct": False,
        "dataset_struct": False,
        "aoi_candidates_struct": False,
        "aoi_options_unique_count": 0,
    }
    needs, reason = ck._needs_user_input(response="Here are your results.", requires=requires, struct=struct)
    assert needs is True
    assert reason == "multiple_missing"


def test_needs_user_input_prefers_explicit_ask_reason_over_missing_required_reason() -> None:
    requires = {
        "requires_data": True,
        "requires_aoi": False,
        "requires_time_range": False,
        "requires_dataset": True,
    }
    struct = {
        "aoi_selected_struct": False,
        "time_range_struct": True,
        "dataset_struct": False,
        "aoi_candidates_struct": False,
        "aoi_options_unique_count": 0,
    }
    needs, reason = ck._needs_user_input(response="Which location should I use?", requires=requires, struct=struct)
    assert needs is True
    assert reason == "missing_aoi"


def test_completion_state_priority_error_beats_needs_user_input() -> None:
    struct = {"aoi_selected_struct": False, "time_range_struct": False, "dataset_struct": False}
    requires = {
        "requires_data": True,
        "requires_aoi": True,
        "requires_time_range": True,
        "requires_dataset": True,
    }
    state, _, _, reason = ck._completion_state("trend_over_time", "model_error", True, struct, requires, has_citations=False)
    assert state == "error"
    assert reason == "model_error"


def test_completion_state_priority_needs_user_input_beats_no_data() -> None:
    struct = {"aoi_selected_struct": True, "time_range_struct": True, "dataset_struct": False}
    requires = {
        "requires_data": True,
        "requires_aoi": False,
        "requires_time_range": False,
        "requires_dataset": True,
    }
    state, _, _, reason = ck._completion_state("data_lookup", "no_data", True, struct, requires, has_citations=False)
    assert state == "needs_user_input"
    assert reason in {"missing_dataset", "multiple_missing", ""}


def test_completion_state_trend_requires_citations_or_marks_incomplete() -> None:
    struct = {"aoi_selected_struct": True, "time_range_struct": True, "dataset_struct": True}
    requires = {
        "requires_data": True,
        "requires_aoi": True,
        "requires_time_range": True,
        "requires_dataset": True,
    }
    state, good_trend, good_lookup, reason = ck._completion_state(
        "trend_over_time", "answer", False, struct, requires, has_citations=False
    )
    assert state == "incomplete_answer"
    assert "no_citation" in reason.split("|")
    assert good_trend is False
    assert good_lookup is False


def test_completion_state_data_lookup_does_not_require_citations_for_complete() -> None:
    struct = {"aoi_selected_struct": True, "time_range_struct": True, "dataset_struct": True}
    requires = {
        "requires_data": True,
        "requires_aoi": True,
        "requires_time_range": True,
        "requires_dataset": True,
    }
    state, good_trend, good_lookup, reason = ck._completion_state(
        "data_lookup", "answer", False, struct, requires, has_citations=False
    )
    assert state == "complete_answer"
    assert good_lookup is True
    assert good_trend is False
    assert reason == ""


def _single(derived: pd.DataFrame) -> pd.Series:
    assert len(derived) == 1
    return derived.iloc[0]


def test_derived_interactions_trend_complete_answer_when_struct_fields_and_citations_present() -> None:
    trace = make_trace(
        trace_id="t-trend-ok",
        prompt="Show tree cover loss trend over time in Brazil from 2020 to 2022.",
        response="Tree cover loss declined from 2020 to 2022.",
        output_obj={
            "selected_aoi": {"name": "Brazil", "gadm_id": "BRA"},
            "start_date": "2020-01-01",
            "end_date": "2022-12-31",
            "dataset": {"dataset_name": "Tree cover loss", "citation": "Hansen et al. 2013"},
            "citation": "Hansen et al. 2013",
            "raw_data": [{"year": 2020, "value": 1.0}],
        },
    )

    derived = ck.compute_derived_interactions([trace])
    row = _single(derived)
    assert row["intent_primary"] == "trend_over_time"
    assert row["completion_state"] == "complete_answer"
    assert bool(row["struct_good_trend"]) is True
    assert str(row["struct_fail_reason"]) == ""


def test_derived_interactions_trend_incomplete_when_missing_citations() -> None:
    trace = make_trace(
        trace_id="t-trend-no-cite",
        prompt="Show tree cover loss trend over time in Brazil from 2020 to 2022.",
        response="Tree cover loss declined from 2020 to 2022.",
        output_obj={
            "selected_aoi": {"name": "Brazil", "gadm_id": "BRA"},
            "start_date": "2020-01-01",
            "end_date": "2022-12-31",
            "dataset": {"dataset_name": "Tree cover loss"},
            "raw_data": [{"year": 2020, "value": 1.0}],
        },
    )
    derived = ck.compute_derived_interactions([trace])
    row = _single(derived)
    assert row["intent_primary"] == "trend_over_time"
    assert row["completion_state"] == "incomplete_answer"
    assert "no_citation" in str(row["struct_fail_reason"]).split("|")


def test_derived_interactions_data_lookup_can_be_complete_without_citations() -> None:
    trace = make_trace(
        trace_id="t-lookup-ok",
        prompt="How much tree cover loss occurred in Brazil in 2021?",
        response="Tree cover loss in 2021 was approximately 1.2M ha.",
        output_obj={
            "selected_aoi": {"name": "Brazil", "gadm_id": "BRA"},
            "start_date": "2021-01-01",
            "end_date": "2021-12-31",
            "dataset": {"dataset_name": "Tree cover loss"},
            "raw_data": [{"year": 2021, "value": 1.2}],
        },
    )
    derived = ck.compute_derived_interactions([trace])
    row = _single(derived)
    assert row["intent_primary"] == "data_lookup"
    assert row["completion_state"] == "complete_answer"
    assert bool(row["struct_good_lookup"]) is True
    assert bool(row["citations_text"]) is False
    assert bool(row["citations_struct"]) is False


def test_derived_interactions_needs_user_input_when_dataset_missing_and_assistant_asks() -> None:
    trace = make_trace(
        trace_id="t-lookup-need-ds",
        prompt="How much tree cover loss occurred in Brazil in 2021?",
        response="Which dataset/layer should I use?",
        output_obj={
            "selected_aoi": {"name": "Brazil", "gadm_id": "BRA"},
            "start_date": "2021-01-01",
            "end_date": "2021-12-31",
        },
    )
    derived = ck.compute_derived_interactions([trace])
    row = _single(derived)
    assert row["completion_state"] == "needs_user_input"
    assert row["needs_user_input_reason"] in {"missing_dataset", "multiple_missing"}


def test_threads_ended_after_needs_user_input_rate_uses_last_turn_per_thread() -> None:
    t1 = make_trace(
        trace_id="thr1-1",
        session_id="sess-a",
        thread_id="thread-1",
        created_at="2026-02-01T00:00:00Z",
        prompt="How much tree cover loss occurred in Brazil in 2021?",
        response="Tree cover loss in 2021 was approximately 1.2M ha.",
        output_obj={
            "selected_aoi": {"name": "Brazil"},
            "start_date": "2021-01-01",
            "end_date": "2021-12-31",
            "dataset": {"dataset_name": "Tree cover loss"},
            "raw_data": [{"year": 2021, "value": 1.2}],
        },
    )
    t2 = make_trace(
        trace_id="thr1-2",
        session_id="sess-a",
        thread_id="thread-1",
        created_at="2026-02-01T00:01:00Z",
        prompt="How much tree cover loss occurred in Brazil in 2021?",
        response="Which dataset/layer should I use?",
        output_obj={
            "selected_aoi": {"name": "Brazil"},
            "start_date": "2021-01-01",
            "end_date": "2021-12-31",
        },
    )
    t3 = make_trace(
        trace_id="thr2-1",
        session_id="sess-b",
        thread_id="thread-2",
        created_at="2026-02-01T00:02:00Z",
        prompt="How much tree cover loss occurred in Peru in 2021?",
        response="Tree cover loss in 2021 was approximately 0.3M ha.",
        output_obj={
            "selected_aoi": {"name": "Peru"},
            "start_date": "2021-01-01",
            "end_date": "2021-12-31",
            "dataset": {"dataset_name": "Tree cover loss"},
            "raw_data": [{"year": 2021, "value": 0.3}],
        },
    )

    derived = ck.compute_derived_interactions([t1, t2, t3])
    summary = ck.summarize_content(derived)
    assert summary["global_quality"]["threads_total"] == 2
    assert summary["global_quality"]["threads_ended_after_needs_user_input"] == 1
    assert summary["global_quality"]["threads_ended_after_needs_user_input_rate"] == pytest.approx(0.5)
