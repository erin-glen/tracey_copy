import importlib.util
import pathlib
import sys
import types
import unittest

import pandas as pd

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load_module(mod_name: str, rel_path: str):
    spec = importlib.util.spec_from_file_location(mod_name, REPO_ROOT / rel_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module


if "utils" not in sys.modules:
    utils_pkg = types.ModuleType("utils")
    utils_pkg.__path__ = [str(REPO_ROOT / "utils")]
    sys.modules["utils"] = utils_pkg

_load_module("utils.trace_parsing", "utils/trace_parsing.py")
content_kpis = _load_module("utils.content_kpis", "utils/content_kpis.py")
build_thread_summary = content_kpis.build_thread_summary


class TestThreadSummary(unittest.TestCase):
    def test_thread_grouping_fallback_priority(self):
        df = pd.DataFrame(
            [
                {
                    "timestamp": "2026-01-01T00:00:00Z",
                    "thread_id": "t1",
                    "sessionId": "s1",
                    "trace_id": "a",
                    "completion_state": "needs_user_input",
                    "intent_primary": "data_lookup",
                },
                {
                    "timestamp": "2026-01-01T00:01:00Z",
                    "thread_id": "t1",
                    "sessionId": "s1",
                    "trace_id": "b",
                    "completion_state": "complete_answer",
                    "intent_primary": "data_lookup",
                },
                {
                    "timestamp": "2026-01-01T00:02:00Z",
                    "thread_id": "",
                    "sessionId": "s2",
                    "trace_id": "c",
                    "completion_state": "error",
                    "intent_primary": "data_lookup",
                },
            ]
        )

        out = build_thread_summary(df)
        self.assertEqual(len(out), 2)

        t1 = out[out["thread_key"] == "t1"].iloc[0]
        self.assertEqual(int(t1["n_turns"]), 2)
        self.assertTrue(bool(t1["ever_complete_answer"]))
        self.assertFalse(bool(t1["ended_after_needs_user_input"]))

        s2 = out[out["thread_key"] == "s2"].iloc[0]
        self.assertTrue(bool(s2["ended_after_error"]))
        self.assertTrue(bool(s2["ever_error"]))

    def test_needs_user_input_reasons_and_last_reason(self):
        df = pd.DataFrame(
            [
                {
                    "timestamp": "2026-01-01T00:00:00Z",
                    "thread_id": "t3",
                    "sessionId": "s3",
                    "trace_id": "a1",
                    "completion_state": "complete_answer",
                    "needs_user_input_reason": "",
                    "intent_primary": "trend_over_time",
                },
                {
                    "timestamp": "2026-01-01T00:01:00Z",
                    "thread_id": "t3",
                    "sessionId": "s3",
                    "trace_id": "a2",
                    "completion_state": "needs_user_input",
                    "needs_user_input_reason": "missing_time",
                    "intent_primary": "trend_over_time",
                },
            ]
        )

        out = build_thread_summary(df)
        row = out.iloc[0]
        self.assertTrue(bool(row["ended_after_needs_user_input"]))
        self.assertEqual(row["last_needs_user_input_reason"], "missing_time")
        self.assertIn("missing_time", row["needs_user_input_reasons"])

    def test_dataset_families_seen_ordered_unique(self):
        df = pd.DataFrame(
            [
                {
                    "timestamp": "2026-01-01T00:00:00Z",
                    "thread_id": "t9",
                    "sessionId": "s9",
                    "trace_id": "x1",
                    "completion_state": "complete_answer",
                    "dataset_family": "tree_cover_loss",
                    "intent_primary": "data_lookup",
                },
                {
                    "timestamp": "2026-01-01T00:01:00Z",
                    "thread_id": "t9",
                    "sessionId": "s9",
                    "trace_id": "x2",
                    "completion_state": "complete_answer",
                    "dataset_family": "tree_cover_loss",
                    "intent_primary": "data_lookup",
                },
                {
                    "timestamp": "2026-01-01T00:02:00Z",
                    "thread_id": "t9",
                    "sessionId": "s9",
                    "trace_id": "x3",
                    "completion_state": "complete_answer",
                    "dataset_family": "alerts",
                    "intent_primary": "data_lookup",
                },
            ]
        )

        out = build_thread_summary(df)
        row = out.iloc[0]
        self.assertEqual(int(row["dataset_families_seen_count"]), 2)
        self.assertTrue(str(row["dataset_families_seen"]).startswith("tree_cover_loss,alerts"))


if __name__ == "__main__":
    unittest.main()
