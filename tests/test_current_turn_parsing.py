import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "utils"))

from trace_parsing import (  # noqa: E402
    current_human_prompt,
    current_turn_ai_message,
    trace_used_tools,
    extract_usage_metadata,
)


class TestCurrentTurnParsing(unittest.TestCase):
    def test_history_contamination_removed(self):
        trace = {
            "input": {
                "messages": [
                    {"type": "human", "content": "first prompt"},
                    {"type": "human", "content": "second prompt"},
                ]
            },
            "output": {
                "messages": [
                    {"type": "human", "content": "first prompt"},
                    {
                        "type": "ai",
                        "content": "thinking",
                        "tool_calls": [{"id": "tc1", "name": "pull_data", "args": {}}],
                        "usage_metadata": {"input_tokens": 10, "output_tokens": 20},
                    },
                    {"type": "tool", "tool_call_id": "tc1", "name": "pull_data", "content": "ok"},
                    {"type": "human", "content": "second prompt"},
                    {
                        "type": "ai",
                        "content": "second answer",
                        "usage_metadata": {"input_tokens": 5, "output_tokens": 7},
                    },
                ]
            },
        }

        self.assertEqual(current_human_prompt(trace), "second prompt")
        self.assertEqual(current_turn_ai_message(trace), "second answer")
        self.assertFalse(trace_used_tools(trace))
        usage = extract_usage_metadata(trace)
        self.assertEqual(usage["total_output_tokens"], 7)

    def test_role_schema_and_parts_content_supported(self):
        trace = {
            "input": {
                "messages": [
                    {"role": "user", "content": [{"text": "older prompt"}]},
                    {"role": "user", "content": [{"content": "current prompt"}]},
                ]
            },
            "output": {
                "messages": [
                    {"role": "user", "content": [{"value": "older prompt"}]},
                    {"role": "assistant", "content": [{"text": "older answer"}]},
                    {"role": "user", "content": [{"text": "current prompt"}]},
                    {"role": "assistant", "content": [{"content": "current answer"}]},
                ]
            },
        }

        self.assertEqual(current_human_prompt(trace), "current prompt")
        self.assertEqual(current_turn_ai_message(trace), "current answer")


if __name__ == "__main__":
    unittest.main()
