import importlib.util
import pathlib
import sys
import types
import unittest

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
compute_derived_interactions = content_kpis.compute_derived_interactions


class TestContentKPIs(unittest.TestCase):
    def test_keeps_missing_outputs(self):
        traces = [
            {
                "id": "t1",
                "timestamp": "2026-01-01T00:00:00Z",
                "sessionId": "s1",
                "userId": "u1",
                "input": {"messages": [{"role": "user", "content": "Show me tree cover loss in Brazil"}]},
                "output": {},
                "metadata": {"thread_id": "th1"},
            },
            {
                "id": "t2",
                "timestamp": "2026-01-01T00:01:00Z",
                "sessionId": "s2",
                "userId": "u2",
                "input": {"messages": [{"role": "user", "content": "How are you?"}]},
                "output": {"messages": [{"role": "assistant", "content": "I can help with maps."}]},
                "metadata": {"thread_id": "th2"},
            },
        ]

        derived = compute_derived_interactions(traces)
        self.assertEqual(len(derived), 2)

        row = derived.loc[derived["trace_id"] == "t1"].iloc[0]
        self.assertTrue(bool(row["response_missing"]))
        self.assertTrue(
            row["completion_state"] == "error"
            or row["answer_type"] in {"missing_output", "model_error", "empty_or_short"}
        )

    def test_scored_intent_completeness(self):
        traces = [
            {
                "id": "t3",
                "timestamp": "2026-01-02T00:00:00Z",
                "sessionId": "s3",
                "userId": "u3",
                "input": {"messages": [{"role": "user", "content": "Show me tree cover loss in Brazil"}]},
                "output": {
                    "messages": [
                        {"role": "assistant", "content": "Tree cover loss in Brazil is available from the dataset."}
                    ],
                    "result": {
                        "aoi": {"name": "Brazil", "type": "country"},
                        "dataset_name": "Tree Cover Loss",
                    },
                },
                "metadata": {"thread_id": "th3"},
            }
        ]

        derived = compute_derived_interactions(traces)
        row = derived.iloc[0]
        self.assertEqual(row["intent_primary"], "data_lookup")
        self.assertEqual(row["completion_state"], "complete_answer")
        self.assertTrue(bool(row["struct_good_lookup"]))

    def test_needs_user_input_reason_gating(self):
        traces = [
            {
                "id": "t4",
                "timestamp": "2026-01-03T00:00:00Z",
                "sessionId": "s4",
                "userId": "u4",
                "input": {"messages": [{"role": "user", "content": "Show me trend over time for tree cover loss in Brazil"}]},
                "output": {
                    "messages": [
                        {"role": "assistant", "content": "Please specify the time range for this trend analysis."}
                    ],
                    "result": {
                        "aoi": {"name": "Brazil", "type": "country"},
                        "dataset_name": "Tree Cover Loss",
                    },
                },
                "metadata": {"thread_id": "th4"},
            }
        ]

        derived = compute_derived_interactions(traces)
        row = derived.iloc[0]
        self.assertEqual(row["intent_primary"], "trend_over_time")
        self.assertEqual(row["completion_state"], "needs_user_input")
        self.assertIn(row["needs_user_input_reason"], {"missing_time", "multiple_missing"})

    def test_dataset_dict_extraction_and_citation_struct(self):
        """Regression: GNW outputs often store dataset metadata under output['dataset'] as a dict."""
        traces = [
            {
                "id": "t5",
                "timestamp": "2026-01-04T00:00:00Z",
                "sessionId": "s5",
                "userId": "u5",
                "input": {
                    "messages": [{"role": "user", "content": "Show me grassland extent in Montana"}]
                },
                "output": {
                    "messages": [
                        {
                            "role": "assistant",
                            "content": "Here is the grassland extent for Montana. Would you like a map?",
                        }
                    ],
                    "aoi": {"name": "Montana", "type": "state"},
                    "dataset": {
                        "dataset_name": "Global natural/semi-natural grassland extent",
                        "citation": "Some citation text",
                    },
                    "start_date": "2020-01-01",
                    "end_date": "2020-12-31",
                },
                "metadata": {"thread_id": "th5"},
            }
        ]

        derived = compute_derived_interactions(traces)
        row = derived.iloc[0]
        self.assertEqual(row["dataset_name"], "Global natural/semi-natural grassland extent")
        self.assertTrue(bool(row["dataset_struct"]))
        self.assertTrue(bool(row["citations_struct"]))
        self.assertTrue(bool(row["dataset_has_citation"]))

    def test_model_error_not_triggered_by_word_error(self):
        """Regression: answers mentioning 'errors' shouldn't be classified as model_error."""
        traces = [
            {
                "id": "t6",
                "timestamp": "2026-01-05T00:00:00Z",
                "sessionId": "s6",
                "userId": "u6",
                "input": {
                    "messages": [{"role": "user", "content": "Show me tree cover loss in Brazil"}]
                },
                "output": {
                    "messages": [
                        {
                            "role": "assistant",
                            "content": "These numbers can have classification errors; here is the estimate...",
                        }
                    ],
                    "result": {
                        "aoi": {"name": "Brazil", "type": "country"},
                        "dataset_name": "Tree Cover Loss",
                    },
                },
                "metadata": {"thread_id": "th6"},
            }
        ]

        row = compute_derived_interactions(traces).iloc[0]
        self.assertNotEqual(row["answer_type"], "model_error")
        self.assertNotEqual(row["completion_state"], "error")

    def test_polite_follow_up_question_does_not_trigger_needs_user_input(self):
        """Regression: trailing friendly questions shouldn't flip into needs_user_input."""
        traces = [
            {
                "id": "t7",
                "timestamp": "2026-01-06T00:00:00Z",
                "sessionId": "s7",
                "userId": "u7",
                "input": {
                    "messages": [{"role": "user", "content": "Show me tree cover loss in Brazil"}]
                },
                "output": {
                    "messages": [
                        {
                            "role": "assistant",
                            "content": "Tree cover loss in Brazil is available from the dataset. Would you like a chart?",
                        }
                    ],
                    "result": {
                        "aoi": {"name": "Brazil", "type": "country"},
                        "dataset_name": "Tree Cover Loss",
                    },
                },
                "metadata": {"thread_id": "th7"},
            }
        ]

        row = compute_derived_interactions(traces).iloc[0]
        self.assertEqual(row["completion_state"], "complete_answer")
        self.assertFalse(bool(row["needs_user_input"]))

    def test_capability_prompt_does_not_require_aoi(self):
        traces = [
            {
                "id": "t8",
                "timestamp": "2026-01-07T00:00:00Z",
                "sessionId": "s8",
                "userId": "u8",
                "input": {"messages": [{"role": "user", "content": "What can you do for me?"}]},
                "output": {
                    "messages": [
                        {
                            "role": "assistant",
                            "content": "I can help you explore GNW datasets and visualize results.",
                        }
                    ]
                },
                "metadata": {"thread_id": "th8"},
            }
        ]

        row = compute_derived_interactions(traces).iloc[0]
        self.assertEqual(row["intent_primary"], "conceptual_or_capability")
        self.assertFalse(bool(row["requires_aoi"]))


if __name__ == "__main__":
    unittest.main()
