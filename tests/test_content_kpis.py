import base64
import importlib.util
import pandas as pd
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

    def test_raw_data_dict_is_detected_as_analysis_executed(self):
        traces = [
            {
                "id": "t9",
                "timestamp": "2026-01-08T00:00:00Z",
                "sessionId": "s9",
                "userId": "u9",
                "input": {"messages": [{"role": "user", "content": "Show me tree cover loss in Brazil"}]},
                "output": {
                    "messages": [{"role": "assistant", "content": "Here are the results."}],
                    "result": {
                        "aoi": {"name": "Brazil", "type": "country"},
                        "dataset_name": "Tree Cover Loss",
                    },
                    # NOTE: GNW export shape: raw_data is often a dict of query-id -> rows
                    "raw_data": {"q1": {"0": {"value": 1}, "1": {"value": 2}}},
                    "charts_data": [],
                },
                "metadata": {"thread_id": "th9"},
            }
        ]

        row = compute_derived_interactions(traces).iloc[0]
        self.assertGreater(int(row.get("raw_data_len", 0)), 0)
        self.assertEqual(int(row["raw_data_len"]), 2)
        self.assertTrue(bool(row["analysis_executed"]))

    def test_codeact_parts_are_counted(self):
        code = "print('hi')"
        code_b64 = base64.b64encode(code.encode("utf-8")).decode("utf-8")
        traces = [
            {
                "id": "t10",
                "timestamp": "2026-01-09T00:00:00Z",
                "sessionId": "s10",
                "userId": "u10",
                "input": {"messages": [{"role": "user", "content": "Show me tree cover loss in Brazil"}]},
                "output": {
                    "messages": [{"role": "assistant", "content": "Here are the results."}],
                    "result": {
                        "aoi": {"name": "Brazil", "type": "country"},
                        "dataset_name": "Tree Cover Loss",
                    },
                    "codeact_parts": [{"type": "code", "content": code_b64}],
                },
                "metadata": {"thread_id": "th10"},
            }
        ]

        row = compute_derived_interactions(traces).iloc[0]
        self.assertTrue(bool(row["codeact_present"]))
        self.assertEqual(int(row["codeact_parts_count"]), 1)
        self.assertEqual(int(row["codeact_code_blocks_count"]), 1)
        self.assertGreaterEqual(int(row["codeact_decoded_chars_total"]), len(code))

    def test_needs_user_input_detects_plural_location_disambiguation(self):
        traces = [
            {
                "id": "t11",
                "timestamp": "2026-01-10T00:00:00Z",
                "sessionId": "s11",
                "userId": "u11",
                "input": {
                    "messages": [
                        {
                            "role": "user",
                            "content": "Show land cover changes in Monterrey over the last 10 years",
                        }
                    ]
                },
                "output": {
                    "messages": [
                        {
                            "role": "assistant",
                            "content": "I found two locations named \"Monterrey\". Could you please specify which one you are interested in?",
                        }
                    ],
                    "result": {
                        # Intentionally omit AOI selection, but keep dataset+time to isolate the AOI disambig
                        "dataset_name": "Land Cover",
                        "time_start": "2014-01-01",
                        "time_end": "2024-01-01",
                    },
                },
                "metadata": {"thread_id": "th11"},
            }
        ]

        row = compute_derived_interactions(traces).iloc[0]
        self.assertTrue(bool(row["needs_user_input"]))
        self.assertEqual(row["completion_state"], "needs_user_input")
        self.assertIn("missing_aoi", str(row["needs_user_input_reason"]))

    def test_metric_sanity_does_not_flag_confidence_percent(self):
        traces = [
            {
                "id": "t12",
                "timestamp": "2026-01-11T00:00:00Z",
                "sessionId": "s12",
                "userId": "u12",
                "input": {"messages": [{"role": "user", "content": "Show me tree cover loss in Brazil"}]},
                "output": {
                    "messages": [{"role": "assistant", "content": "I'm 100% sure this is correct."}],
                    "result": {
                        "aoi": {"name": "Brazil", "type": "country"},
                        "dataset_name": "Tree Cover Loss",
                    },
                },
                "metadata": {"thread_id": "th12"},
            }
        ]

        row = compute_derived_interactions(traces).iloc[0]
        self.assertNotEqual(row["answer_type"], "model_error")
        self.assertNotEqual(row["completion_state"], "error")

    def test_flux_query_classified_as_data_lookup_without_question_mark(self):
        """Regression: noun-phrase carbon/flux queries should not fall into intent_primary='other'."""
        traces = [
            {
                "id": "t13",
                "timestamp": "2026-01-12T00:00:00Z",
                "sessionId": "s13",
                "userId": "u13",
                "input": {"messages": [{"role": "user", "content": "Forest greenhouse gas net flux for Winchester"}]},
                "output": {
                    "messages": [{"role": "assistant", "content": "Here are the net flux results."}],
                    "result": {
                        "aoi": {"name": "Winchester", "type": "city"},
                        "dataset_name": "Forest GHG Net Flux",
                    },
                    "raw_data": {"q1": {"0": {"value": 1}}},
                },
                "metadata": {"thread_id": "th13"},
            }
        ]

        row = compute_derived_interactions(traces).iloc[0]
        self.assertEqual(row["intent_primary"], "data_lookup")
        self.assertTrue(bool(row["analysis_executed"]))

    def test_confidence_alerts_prompt_is_not_conceptual(self):
        """Regression: 'confidence' used as an alert filter should remain a data intent."""
        traces = [
            {
                "id": "t14",
                "timestamp": "2026-01-13T00:00:00Z",
                "sessionId": "s14",
                "userId": "u14",
                "input": {"messages": [{"role": "user", "content": "Show me high confidence GLAD alerts in Brazil"}]},
                "output": {
                    "messages": [{"role": "assistant", "content": "Here are the alerts."}],
                    "result": {
                        "aoi": {"name": "Brazil", "type": "country"},
                        "dataset_name": "GLAD Alerts",
                    },
                },
                "metadata": {"thread_id": "th14"},
            }
        ]

        row = compute_derived_interactions(traces).iloc[0]
        self.assertEqual(row["intent_primary"], "data_lookup")
        self.assertNotEqual(row["intent_primary"], "conceptual_or_capability")

    def test_needs_user_input_detected_even_when_requires_aoi_is_false(self):
        """Regression: if the assistant asks for a missing location, mark needs_user_input even if requires_aoi was False."""
        traces = [
            {
                "id": "t15",
                "timestamp": "2026-01-14T00:00:00Z",
                "sessionId": "s15",
                "userId": "u15",
                "input": {"messages": [{"role": "user", "content": "Forest greenhouse gas net flux for Winchester"}]},
                "output": {
                    "messages": [
                        {
                            "role": "assistant",
                            "content": "I found multiple locations named Winchester. Could you please specify which location you mean?",
                        }
                    ],
                    "result": {
                        "dataset_name": "Forest GHG Net Flux",
                    },
                },
                "metadata": {"thread_id": "th15"},
            }
        ]

        row = compute_derived_interactions(traces).iloc[0]
        self.assertTrue(bool(row["needs_user_input"]))
        self.assertEqual(row["completion_state"], "needs_user_input")
        self.assertIn("missing_aoi", str(row["needs_user_input_reason"]))

    def test_no_data_detects_global_scope_not_supported(self):
        """Regression: global/continental scope limitations should be classified as no_data/unsupported (not answer)."""
        traces = [
            {
                "id": "t16",
                "timestamp": "2026-01-15T00:00:00Z",
                "sessionId": "s16",
                "userId": "u16",
                "input": {"messages": [{"role": "user", "content": "Show me tree cover loss globally"}]},
                "output": {
                    "messages": [
                        {
                            "role": "assistant",
                            "content": "Sorry — I cannot currently support requests for the entire world at once. Please select a country or region.",
                        }
                    ],
                    "result": {},
                },
                "metadata": {"thread_id": "th16"},
            }
        ]

        row = compute_derived_interactions(traces).iloc[0]
        self.assertEqual(row["answer_type"], "no_data")
        # Narrowing global scope is a blocking clarification even though the message contains no-data/unsupported language.
        self.assertEqual(row["completion_state"], "needs_user_input")
        self.assertIn("missing_aoi", str(row["needs_user_input_reason"]))



    def test_aoi_options_do_not_count_as_selected_aoi(self):
        traces = [
            {
                "id": "tA",
                "timestamp": "2026-01-01T00:00:00Z",
                "sessionId": "sA",
                "userId": "uA",
                "input": {"messages": [{"role": "user", "content": "Tree cover loss in Winchester"}]},
                "output": {
                    "aoi_options": [
                        {"aoi": {"name": "Winchester, Virginia, United States", "gadm_id": "USA.47.1_1"}},
                        {"aoi": {"name": "Winchester, Manitoba, Canada", "gadm_id": "CAN.3.4_1"}},
                        # duplicate option should not affect unique-count logic
                        {"aoi": {"name": "Winchester, Virginia, United States", "gadm_id": "USA.47.1_1"}},
                    ]
                },
                "metadata": {"thread_id": "thA"},
            }
        ]

        df = compute_derived_interactions(traces)
        self.assertEqual(len(df), 1)
        row = df.iloc[0].to_dict()

        self.assertTrue(row.get("aoi_candidates_struct"))
        # Candidate options should not be treated as a selected AOI.
        self.assertFalse(row.get("aoi_selected_struct"))

    def test_time_terms_do_not_match_english_data_word(self):
        # "data" in English should not be interpreted as a *date/time* request,
        # but it CAN be a dataset-selection question.
        response = "What specific environmental data would you like to explore?"
        requires = {"requires_aoi": False, "requires_time_range": False, "requires_dataset": False}
        struct = {
            "aoi_selected_struct": True,
            "time_range_struct": False,
            "dataset_struct": False,
            "aoi_candidates_struct": False,
            "aoi_options_unique_count": 0,
        }
        needs, reason = content_kpis._needs_user_input(response, requires, struct)
        self.assertTrue(needs)
        self.assertEqual(reason, "missing_dataset")

    def test_no_data_detects_no_access_phrase(self):
        traces = [
            {
                "id": "t_no_access",
                "timestamp": "2025-01-01T00:00:00Z",
                "sessionId": "s1",
                "thread_id": "th1",
                "userId": "u1",
                "level": "info",
                "errorCount": 0,
                "latency": 1.0,
                "input_tokens": 10,
                "output_tokens": 20,
                "input": {"messages": [{"type": "human", "content": "Do we have soil quality data?"}]},
                "output": {
                    "messages": [
                        {
                            "role": "assistant",
                            "content": "I do not have access to soil quality data in my available datasets.",
                        }
                    ]
                },
            }
        ]
        out = compute_derived_interactions(traces)
        self.assertEqual(out.loc[0, "answer_type"], "no_data")
        self.assertEqual(out.loc[0, "completion_state"], "no_data")

    def test_dataset_family_is_null_when_missing(self):
        traces = [
            {
                "id": "t_no_dataset",
                "timestamp": "2025-01-01T00:00:00Z",
                "sessionId": "s1",
                "thread_id": "th1",
                "userId": "u1",
                "level": "info",
                "errorCount": 0,
                "latency": 1.0,
                "input_tokens": 10,
                "output_tokens": 20,
                "input": {"messages": [{"type": "human", "content": "Ghana"}]},
                "output": {"messages": [{"role": "assistant", "content": "I have selected Ghana."}]},
            }
        ]
        out = compute_derived_interactions(traces)
        self.assertTrue(pd.isna(out.loc[0, "dataset_family"]))

    def test_needs_user_input_detects_too_many_candidates_even_with_aoi_selected(self):
        response = (
            "I found 41 protected areas in Loreto, which is too many to analyze at once. "
            "Could you please specify which protected area you mean?"
        )
        requires = {"requires_aoi": True, "requires_time_range": False, "requires_dataset": False}
        struct = {
            "aoi_selected_struct": True,
            "time_range_struct": True,
            "dataset_struct": True,
            "aoi_candidates_struct": False,
            "aoi_options_unique_count": 0,
        }
        needs, reason = content_kpis._needs_user_input(response, requires, struct)
        self.assertTrue(needs)
        self.assertEqual(reason, "missing_aoi")

    def test_needs_user_input_detects_missing_time_with_eg_abbrev(self):
        response = "**Time Period**: What years would you like to analyze (e.g., 2015-2023)?"
        requires = {"requires_aoi": False, "requires_time_range": False, "requires_dataset": False}
        struct = {
            "aoi_selected_struct": True,
            "time_range_struct": False,
            "dataset_struct": True,
            "aoi_candidates_struct": False,
            "aoi_options_unique_count": 0,
        }
        needs, reason = content_kpis._needs_user_input(response, requires, struct)
        self.assertTrue(needs)
        self.assertEqual(reason, "missing_time")

    def test_needs_user_input_detects_data_colon_as_dataset_request(self):
        response = "Could you please specify:\n- Data: Tree cover loss\n- Time range: 2015-2023"
        requires = {"requires_aoi": False, "requires_time_range": False, "requires_dataset": False}
        struct = {
            "aoi_selected_struct": True,
            "time_range_struct": False,
            "dataset_struct": False,
            "aoi_candidates_struct": False,
            "aoi_options_unique_count": 0,
        }
        needs, reason = content_kpis._needs_user_input(response, requires, struct)
        self.assertTrue(needs)
        self.assertEqual(reason, "multiple_missing")

    def test_aoi_disambig_does_not_trigger_on_year_ranges(self):
        response = (
            "I see you've selected the Forest greenhouse gas net flux dataset (2001-2024) "
            "for Ondo State, Nigeria. Here are the results."
        )
        requires = {"requires_aoi": True, "requires_time_range": True, "requires_dataset": True}
        struct = {
            "aoi_selected_struct": True,
            "time_range_struct": True,
            "dataset_struct": True,
            "aoi_candidates_struct": False,
            "aoi_options_unique_count": 0,
        }
        needs, reason = content_kpis._needs_user_input(response, requires, struct)
        self.assertFalse(needs)
        self.assertEqual(reason, "")

    def test_which_of_these_followup_not_needs_user_input_without_candidates(self):
        response = "Which of these would you like to see next?"
        requires = {"requires_aoi": True, "requires_time_range": False, "requires_dataset": False}
        struct = {
            "aoi_selected_struct": True,
            "time_range_struct": True,
            "dataset_struct": True,
            "aoi_candidates_struct": False,
            "aoi_options_unique_count": 0,
        }
        needs, reason = content_kpis._needs_user_input(response, requires, struct)
        self.assertFalse(needs)
        self.assertEqual(reason, "")

if __name__ == "__main__":
    unittest.main()
