import unittest
import importlib.util
import sys
import types
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_utils_module(module_name: str):
    """Load utils.<module_name> without executing utils/__init__.py (Streamlit dependency)."""
    if "utils" not in sys.modules:
        utils_pkg = types.ModuleType("utils")
        utils_pkg.__path__ = [str(REPO_ROOT / "utils")]
        sys.modules["utils"] = utils_pkg

    full_name = f"utils.{module_name}"
    if full_name in sys.modules:
        return sys.modules[full_name]

    path = REPO_ROOT / "utils" / f"{module_name}.py"
    spec = importlib.util.spec_from_file_location(full_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load {full_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[full_name] = module
    spec.loader.exec_module(module)
    return module


eval_sampling = _load_utils_module("eval_sampling")
build_preset_mask = eval_sampling.build_preset_mask
sample_trace_ids = eval_sampling.sample_trace_ids


class TestEvalSampling(unittest.TestCase):
    def test_preset_masks(self):
        df = pd.DataFrame(
            [
                {
                    "trace_id": "t1",
                    "intent_primary": "trend_over_time",
                    "completion_state": "error",
                    "struct_fail_reason": "no_citation|missing_dataset",
                    "answer_type": "answer",
                    "codeact_present": False,
                },
                {
                    "trace_id": "t2",
                    "intent_primary": "data_lookup",
                    "completion_state": "error",
                    "struct_fail_reason": "missing_dataset",
                    "answer_type": "model_error",
                    "codeact_present": False,
                },
                {
                    "trace_id": "t3",
                    "intent_primary": "other",
                    "completion_state": "complete_answer",
                    "struct_fail_reason": "",
                    "answer_type": "missing_output",
                    "codeact_present": False,
                },
                {
                    "trace_id": "t4",
                    "intent_primary": "other",
                    "completion_state": "complete_answer",
                    "struct_fail_reason": "",
                    "answer_type": "empty_or_short",
                    "codeact_present": False,
                },
            ]
        )

        trend_no_citation = df.loc[build_preset_mask(df, "trend_no_citation"), "trace_id"].tolist()
        self.assertEqual(trend_no_citation, ["t1"])

        lookup_missing_dataset = df.loc[build_preset_mask(df, "lookup_missing_dataset"), "trace_id"].tolist()
        self.assertEqual(lookup_missing_dataset, ["t2"])

        model_errors = sorted(df.loc[build_preset_mask(df, "model_errors"), "trace_id"].tolist())
        self.assertEqual(model_errors, ["t2", "t3", "t4"])

        df2 = pd.DataFrame(
            [
                {"trace_id": "c1", "intent_primary": "trend_over_time", "codeact_present": True, "codeact_consistency_issue": True},
                {"trace_id": "c2", "intent_primary": "data_lookup", "codeact_present": True, "codeact_consistency_issue": False},
                {"trace_id": "c3", "intent_primary": "other", "codeact_present": True, "codeact_consistency_issue": True},
            ]
        )
        codeact_param_issues = df2.loc[build_preset_mask(df2, "codeact_param_issues"), "trace_id"].tolist()
        self.assertEqual(codeact_param_issues, ["c1"])

    def test_deterministic_sampling(self):
        df = pd.DataFrame([{"trace_id": f"t{i}", "intent_primary": "A"} for i in range(20)])
        a = sample_trace_ids(df, n=8, seed=42)
        b = sample_trace_ids(df, n=8, seed=42)
        self.assertEqual(a, b)

    def test_stratified_balanced_sampling(self):
        df = pd.DataFrame(
            [{"trace_id": f"a{i}", "intent_primary": "A"} for i in range(5)]
            + [{"trace_id": f"b{i}", "intent_primary": "B"} for i in range(5)]
        )
        sampled = sample_trace_ids(df, n=4, seed=7, stratify_col="intent_primary")
        sampled_df = df[df["trace_id"].isin(sampled)]
        counts = sampled_df["intent_primary"].value_counts().to_dict()
        self.assertEqual(counts.get("A"), 2)
        self.assertEqual(counts.get("B"), 2)

    def test_one_per_thread(self):
        df = pd.DataFrame(
            [
                {"trace_id": "t1", "thread_id": "th1", "sessionId": "s1", "intent_primary": "A"},
                {"trace_id": "t2", "thread_id": "th1", "sessionId": "s1", "intent_primary": "A"},
                {"trace_id": "t3", "thread_id": "th2", "sessionId": "s2", "intent_primary": "A"},
            ]
        )
        sampled = sample_trace_ids(df, n=3, seed=0, one_per_thread=True)
        self.assertLessEqual(len({t for t in sampled if t in {"t1", "t2"}}), 1)


if __name__ == "__main__":
    unittest.main()
