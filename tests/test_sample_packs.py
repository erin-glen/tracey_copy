import base64
import unittest
import importlib.util
import sys
import types
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_utils_module(module_name: str):
    """Load a module from utils/... without importing utils/__init__.py."""
    if "utils" not in sys.modules:
        pkg = types.ModuleType("utils")
        pkg.__path__ = [str(REPO_ROOT / "utils")]
        sys.modules["utils"] = pkg

    full_name = f"utils.{module_name}"
    if full_name in sys.modules:
        return sys.modules[full_name]

    path = REPO_ROOT / "utils" / f"{module_name}.py"
    spec = importlib.util.spec_from_file_location(full_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load {full_name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[full_name] = mod
    spec.loader.exec_module(mod)
    return mod


_sample_packs = _load_utils_module("sample_packs")
add_codeact_snippets_for_pack = _sample_packs.add_codeact_snippets_for_pack
build_sample_pack_df = _sample_packs.build_sample_pack_df


class TestSamplePacks(unittest.TestCase):
    def test_pack_selection_logic(self):
        df = pd.DataFrame(
            [
                {
                    "trace_id": "t1",
                    "timestamp": "2024-01-01T00:00:00Z",
                    "intent_primary": "trend_over_time",
                    "completion_state": "failed",
                    "struct_fail_reason": "no_citation",
                    "answer_type": "answer",
                },
                {
                    "trace_id": "t2",
                    "timestamp": "2024-01-02T00:00:00Z",
                    "intent_primary": "data_lookup",
                    "completion_state": "failed",
                    "struct_fail_reason": "missing_dataset",
                    "answer_type": "answer",
                },
                {
                    "trace_id": "t3",
                    "timestamp": "2024-01-03T00:00:00Z",
                    "intent_primary": "other",
                    "completion_state": "complete_answer",
                    "struct_fail_reason": "",
                    "answer_type": "model_error",
                },
                {
                    "trace_id": "t4",
                    "timestamp": "2024-01-04T00:00:00Z",
                    "intent_primary": "other",
                    "completion_state": "complete_answer",
                    "struct_fail_reason": "",
                    "answer_type": "missing_output",
                },
                {
                    "trace_id": "t5",
                    "timestamp": "2024-01-05T00:00:00Z",
                    "intent_primary": "other",
                    "completion_state": "complete_answer",
                    "struct_fail_reason": "",
                    "answer_type": "empty_or_short",
                },
                {
                    "trace_id": "t6",
                    "timestamp": "2024-01-06T00:00:00Z",
                    "intent_primary": "other",
                    "completion_state": "needs_user_input",
                    "struct_fail_reason": "",
                    "answer_type": "answer",
                },
            ]
        )

        trend_no_citation = build_sample_pack_df(df, "trend_no_citation")
        self.assertTrue((trend_no_citation["struct_fail_reason"].str.contains("no_citation")).all())

        lookup_missing_dataset = build_sample_pack_df(df, "lookup_missing_dataset")
        self.assertTrue((lookup_missing_dataset["struct_fail_reason"].str.contains("missing_dataset")).all())

        model_errors = build_sample_pack_df(df, "model_errors")
        self.assertTrue(model_errors["answer_type"].isin({"model_error", "missing_output", "empty_or_short"}).all())

        needs_user_input = build_sample_pack_df(df, "needs_user_input")
        self.assertTrue((needs_user_input["completion_state"] == "needs_user_input").all())

    def test_deterministic_ordering_and_cap(self):
        df = pd.DataFrame(
            [
                {
                    "trace_id": f"t{i}",
                    "timestamp": f"2024-01-{i:02d}T00:00:00Z",
                    "intent_primary": "trend_over_time",
                    "completion_state": "failed",
                    "struct_fail_reason": "",
                }
                for i in range(1, 11)
            ]
        )
        shuffled = df.sample(frac=1.0, random_state=7).reset_index(drop=True)

        out = build_sample_pack_df(shuffled, "trend_failures", max_rows=3)
        self.assertEqual(len(out), 3)
        self.assertEqual(out["timestamp"].tolist(), sorted(out["timestamp"].tolist(), reverse=True))

    def test_codeact_snippet_extraction(self):
        code_b64 = base64.b64encode("print('hi')".encode("utf-8")).decode("utf-8")
        exec_b64 = base64.b64encode("hi".encode("utf-8")).decode("utf-8")

        pack_df = pd.DataFrame([{"trace_id": "t-code"}])
        trace_by_id = {
            "t-code": {
                "output": {
                    "codeact_parts": [
                        {"type": "code_block", "content": code_b64},
                        {"type": "execution_output", "content": exec_b64},
                    ]
                }
            }
        }

        out = add_codeact_snippets_for_pack(pack_df, trace_by_id)
        self.assertIn("print", out.loc[0, "codeact_code_snippet"])
        self.assertIn("hi", out.loc[0, "codeact_exec_snippet"])


if __name__ == "__main__":
    unittest.main()
