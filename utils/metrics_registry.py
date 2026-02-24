"""Metric + page documentation registry.

This registry is the single source of truth for:
  1) inline KPI tooltips (popover / expander help)
  2) the Metrics Glossary page
  3) per-page "How to read this" sections

It is intentionally **static** (plain dicts) so definitions can be edited
without touching UI code.

Notes on provenance language
----------------------------
* "Raw" = comes directly from a Langfuse trace field (e.g., latency, totalCost).
* "Derived" = computed deterministically by this repo (heuristics, parsing, joins).
* "External" = requires additional Langfuse API calls beyond the loaded trace window.
"""

from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# Metric documentation
# ---------------------------------------------------------------------------

METRICS: dict[str, dict[str, Any]] = {
    # -------------------------
    # Analytics (trace-level)
    # -------------------------
    "total_traces": {
        "name": "Total traces",
        "category": "Analytics · Volume",
        "definition": "Count of traces loaded into the current session (≈ prompts).",
        "population": 'Loaded traces in the current session',
        "numerator": 'Number of traces loaded',
        "denominator": 'N/A',
        "unit": 'count',
        "method": 'deterministic',
        "code_refs": ['tabs.analytics.render'],
        "formula": "`total_traces = len(traces_loaded)`",
        "provenance": "Derived from the in-memory dataset you fetched via the sidebar (`stats_traces`).",
        "caveats": [
            "A multi-turn conversation typically produces multiple traces (one per user turn).",
            "If you re-fetch with different filters, this number changes for all pages.",
        ],
        "used_in": ["Analytics"],
    },
    "unique_threads": {
        "name": "Unique threads",
        "category": "Analytics · Volume",
        "definition": "Distinct conversation threads in the loaded traces (thread_id → sessionId/session_id → trace_id fallback).",
        "population": 'Loaded traces',
        "numerator": 'Distinct thread_key values',
        "denominator": 'N/A',
        "unit": 'count',
        "method": 'deterministic',
        "code_refs": ['utils.content_kpis.compute_thread_key', 'tabs.analytics.render', 'tabs.session_urls.render'],
        "formula": "`unique_threads = nunique(thread_key)`",
        "provenance": "Derived by `utils.content_kpis.compute_thread_key()` from identifiers on each row.",
        "caveats": [
            "If `sessionId`/`thread_id` are missing, the fallback uses `trace_id` (effectively counting turns as threads).",
        ],
        "used_in": ["Analytics"],
    },
    "unique_users": {
        "name": "Unique users",
        "category": "Analytics · Volume",
        "definition": "Distinct user IDs that generated at least one trace in the loaded window.",
        "population": 'Loaded traces with a user identifier',
        "numerator": 'Distinct user IDs (after trimming blanks and excluding internal users)',
        "denominator": 'N/A',
        "unit": 'count',
        "method": 'deterministic',
        "code_refs": ['tabs.analytics.render'],
        "formula": "`unique_users = nunique(user_id)` (after dropping blanks and 'machine' users)",
        "provenance": "Derived from trace `userId` (or metadata user_id/userId) after normalization.",
        "caveats": [
            "Users with missing IDs are excluded.",
            "Heuristically filters out IDs containing the string 'machine'.",
        ],
        "used_in": ["Analytics"],
    },
    "success_rate": {
        "name": "Success rate",
        "category": "Analytics · Outcomes",
        "definition": "Share of traces classified as a successful answer.",
        "population": 'Loaded traces',
        "numerator": "Traces with outcome == 'ANSWER'",
        "denominator": 'Total loaded traces',
        "unit": '%',
        "method": 'heuristic',
        "code_refs": ['utils.trace_parsing.classify_outcome', 'tabs.analytics.render'],
        "formula": "`success_rate = mean(outcome == 'ANSWER')`",
        "provenance": "Outcome is derived by `utils.trace_parsing.classify_outcome()` using final answer text + tool usage.",
        "caveats": [
            "This is a heuristic, not a human judgment of correctness.",
            "Tool usage is evaluated in the *active turn window* (last human prompt → corresponding AI response), not across the entire trace.",
            "Traces with valid-looking text but no tool calls in the active window may be counted as DEFER instead of ANSWER.",
        ],
        "used_in": ["Analytics"],
    },
    "defer_rate": {
        "name": "Defer rate",
        "category": "Analytics · Outcomes",
        "definition": "Share of traces where the assistant responded but did not use tools in the active turn.",
        "population": 'Loaded traces',
        "numerator": "Traces with outcome == 'DEFER'",
        "denominator": 'Total loaded traces',
        "unit": '%',
        "method": 'heuristic',
        "code_refs": ['utils.trace_parsing.classify_outcome', 'tabs.analytics.render'],
        "formula": "`defer_rate = mean(outcome == 'DEFER')`",
        "provenance": "Derived by `classify_outcome()` when answer text is non-empty, not a soft-error, and `active_turn_used_tools(trace)` is False.",
        "caveats": [
            "A DEFER is not always bad: some queries are conceptual and don't require data pulls.",
            "If tool-call logging is incomplete, defers can be overcounted.",
        ],
        "used_in": ["Analytics"],
    },
    "soft_error_rate": {
        "name": "Soft error rate",
        "category": "Analytics · Outcomes",
        "definition": "Share of traces where the answer text *looks like* an error/apology response.",
        "population": 'Loaded traces',
        "numerator": "Traces with outcome == 'SOFT_ERROR'",
        "denominator": 'Total loaded traces',
        "unit": '%',
        "method": 'heuristic',
        "code_refs": ['utils.trace_parsing.classify_outcome', 'utils.trace_parsing.looks_like_error_answer'],
        "formula": "`soft_error_rate = mean(outcome == 'SOFT_ERROR')`",
        "provenance": "Derived by `classify_outcome()` via `looks_like_error_answer(answer)` string heuristics.",
        "caveats": [
            "False positives can occur (e.g., user asks about 'error' as a concept).",
            "False negatives can occur if errors are phrased unusually.",
        ],
        "used_in": ["Analytics"],
    },
    "error_rate": {
        "name": "Error rate",
        "category": "Analytics · Outcomes",
        "definition": "Share of traces classified as ERROR (hard error in the active turn, or an AI turn with an empty extracted answer).",
        "population": 'Loaded traces',
        "numerator": "Traces with outcome == 'ERROR'",
        "denominator": 'Total loaded traces',
        "unit": '%',
        "method": 'heuristic',
        "code_refs": ['utils.trace_parsing.classify_outcome'],
        "formula": "`error_rate = mean(outcome == 'ERROR')`",
        "provenance": "Derived by `classify_outcome()` using active-turn error status + whether an AI message exists for the active turn.",
        "caveats": [
            "This is a heuristic bucket and may include transient tool/runtime failures as well as empty model outputs.",
            "Does not capture errors where the model returns text but it is still unusable.",
        ],
        "used_in": ["Analytics"],
    },
    "empty_rate": {
        "name": "Empty rate",
        "category": "Analytics · Outcomes",
        "definition": "Share of traces classified as EMPTY (no AI answer message found for the active turn).",
        "population": "Loaded traces",
        "numerator": "Traces with outcome == 'EMPTY'",
        "denominator": "Total loaded traces",
        "unit": "%",
        "method": "heuristic",
        "code_refs": [
            "utils.trace_parsing.classify_outcome",
            "utils.trace_parsing.active_turn_has_ai_message",
            "tabs.analytics.render",
        ],
        "formula": "`empty_rate = mean(outcome == 'EMPTY')`",
        "provenance": "Derived by `classify_outcome()` when there is no AI message in the active turn window.",
        "caveats": [
            "Often indicates timeouts, dropped requests, or logging gaps rather than a model refusal.",
            "If traces are truncated, EMPTY can be overcounted.",
        ],
        "used_in": ["Analytics"],
    },
    "user_visible_error_rate": {
        "name": "User-visible error rate",
        "category": "Analytics · Outcomes",
        "definition": "Share of traces where the user saw an error/empty response (SOFT_ERROR, ERROR, or EMPTY).",
        "population": "Loaded traces",
        "numerator": "Traces where outcome ∈ {SOFT_ERROR, ERROR, EMPTY}",
        "denominator": "Total loaded traces",
        "unit": "%",
        "method": "heuristic",
        "code_refs": [
            "utils.trace_parsing.classify_outcome",
            "tabs.analytics.render",
        ],
        "formula": "`user_visible_error_rate = mean(outcome in {'SOFT_ERROR','ERROR','EMPTY'})`",
        "provenance": "Derived from the `outcome` label computed per trace in Analytics.",
        "caveats": [
            "This is a heuristic; it does not require human evaluation.",
            "EMPTY indicates no AI message was found for the active turn; this can be caused by timeouts or logging gaps.",
        ],
        "used_in": ["Analytics"],
    },
    "internal_tool_error_rate": {
        "name": "Internal tool error rate",
        "category": "Analytics · Outcomes",
        "definition": "Share of traces with any internal tool/API error (even if the agent recovered).",
        "population": "Loaded traces",
        "numerator": "Traces where has_internal_error is True",
        "denominator": "Total loaded traces",
        "unit": "%",
        "method": "deterministic",
        "code_refs": [
            "utils.trace_parsing.trace_has_internal_error",
            "utils.trace_parsing.extract_tool_calls_and_results",
            "tabs.analytics.render",
        ],
        "formula": "`internal_tool_error_rate = mean(has_internal_error == True)`",
        "provenance": "Derived from tool call statuses inside the trace output.",
        "caveats": [
            "Internal errors can be masked by agent recovery (the user may still see a successful answer).",
            "If tool call results/statuses are missing in traces, internal errors can be undercounted.",
        ],
        "used_in": ["Analytics"],
    },
    "tool_failure_rate": {
        "name": "Tool failure rate",
        "category": "Analytics · Outcomes",
        "definition": "Among tool-using traces, share where at least one tool call failed.",
        "population": "Traces where the agent used tools in the active turn",
        "numerator": "Tool-using traces with has_internal_error == True",
        "denominator": "Tool-using traces",
        "unit": "%",
        "method": "deterministic",
        "code_refs": [
            "utils.trace_parsing.active_turn_used_tools",
            "utils.trace_parsing.trace_has_internal_error",
            "tabs.analytics.render",
        ],
        "formula": "`tool_failure_rate = mean(has_internal_error == True)` over traces where `used_tools_active_turn == True`",
        "provenance": "Derived by combining active-turn tool usage with internal tool/API error detection.",
        "caveats": [
            "This focuses on reliability of tool calls, not overall user-visible success.",
            "If a trace has multiple tool calls, any one failing counts as a failure.",
        ],
        "used_in": ["Analytics"],
    },
    "mean_cost": {
        "name": "Mean cost",
        "category": "Analytics · Performance",
        "definition": "Average LLM cost per trace (in USD), over traces with cost recorded.",
        "population": 'Loaded traces with totalCost recorded',
        "numerator": 'Mean of total_cost across traces with cost',
        "denominator": 'Traces with cost recorded',
        "unit": 'USD',
        "method": 'raw_langfuse',
        "code_refs": ['tabs.analytics.render'],
        "formula": "`mean_cost = mean(total_cost)`",
        "provenance": "Raw from trace field `totalCost` (normalized to float).",
        "caveats": [
            "If `totalCost` is missing for some traces, they are excluded from the mean.",
        ],
        "used_in": ["Analytics"],
    },
    "median_cost": {
        "name": "Median cost",
        "category": "Analytics · Performance",
        "definition": "Median LLM cost per trace (in USD), over traces with cost recorded.",
        "population": "Loaded traces with totalCost recorded",
        "numerator": "Median of total_cost",
        "denominator": "Traces with cost recorded",
        "unit": "USD",
        "method": "raw_langfuse",
        "code_refs": [
            "tabs.analytics.render",
        ],
        "formula": "`median_cost = median(total_cost)`",
        "provenance": "Raw from trace field `totalCost` (normalized to float) and aggregated as median.",
        "caveats": [
            "More robust to outliers than mean_cost.",
            "If `totalCost` is missing for some traces, they are excluded.",
        ],
        "used_in": ["Analytics"],
    },
    "p95_cost": {
        "name": "P95 cost",
        "category": "Analytics · Performance",
        "definition": "95th percentile cost per trace (USD).",
        "population": 'Loaded traces with totalCost recorded',
        "numerator": '95th percentile of total_cost',
        "denominator": 'Traces with cost recorded',
        "unit": 'USD',
        "method": 'raw_langfuse',
        "code_refs": ['tabs.analytics.render'],
        "formula": "`p95_cost = quantile(total_cost, 0.95)`",
        "provenance": "Raw from trace field `totalCost` (normalized to float).",
        "caveats": [
            "Sensitive to missing values and small sample sizes.",
        ],
        "used_in": ["Analytics"],
    },
    "mean_latency": {
        "name": "Mean latency",
        "category": "Analytics · Performance",
        "definition": "Average end-to-end latency per trace (seconds), over traces with latency recorded.",
        "population": 'Loaded traces with latency recorded',
        "numerator": 'Mean of latency_seconds across traces with latency',
        "denominator": 'Traces with latency recorded',
        "unit": 'seconds',
        "method": 'raw_langfuse',
        "code_refs": ['tabs.analytics.render'],
        "formula": "`mean_latency = mean(latency_seconds)`",
        "provenance": "Raw from trace field `latency` (normalized to float seconds).",
        "caveats": [
            "Latency can include tool time + model time; interpretation depends on how Langfuse records it.",
        ],
        "used_in": ["Analytics"],
    },
    "p95_latency": {
        "name": "P95 latency",
        "category": "Analytics · Performance",
        "definition": "95th percentile latency per trace (seconds).",
        "population": 'Loaded traces with latency recorded',
        "numerator": '95th percentile of latency_seconds',
        "denominator": 'Traces with latency recorded',
        "unit": 'seconds',
        "method": 'raw_langfuse',
        "code_refs": ['tabs.analytics.render'],
        "formula": "`p95_latency = quantile(latency_seconds, 0.95)`",
        "provenance": "Raw from trace field `latency` (normalized to float seconds).",
        "caveats": [
            "Sensitive to outliers and missing values.",
        ],
        "used_in": ["Analytics"],
    },
    "user_days": {
        "name": "User-days",
        "category": "Analytics · Engagement",
        "definition": "Count of distinct (user, day) pairs in the loaded traces.",
        "population": 'Loaded traces with parsed timestamp + user_id',
        "numerator": 'Distinct (user_id, date) pairs',
        "denominator": 'N/A',
        "unit": 'count',
        "method": 'deterministic',
        "code_refs": ['tabs.analytics.render'],
        "formula": "Group by (`date`, `user_id`) then count rows.",
        "provenance": "Derived from parsed timestamp → `date` and normalized `user_id`.",
        "caveats": [
            "Requires both timestamp parsing and user_id availability.",
        ],
        "used_in": ["Analytics"],
    },
    "mean_prompts_per_user_day": {
        "name": "Mean prompts/user/day",
        "category": "Analytics · Engagement",
        "definition": "Average number of prompts a user sends per active day.",
        "population": 'User-days in loaded traces',
        "numerator": 'Mean prompts per (user_id, date)',
        "denominator": 'User-days',
        "unit": 'prompts/user/day',
        "method": 'deterministic',
        "code_refs": ['tabs.analytics.render'],
        "formula": "Compute prompts per (user, day), then take the mean.",
        "provenance": "Derived from traces grouped by (`date`, `user_id`).",
        "caveats": [
            "Highly sensitive to power users; consider median + p95 alongside mean.",
        ],
        "used_in": ["Analytics"],
    },
    "p95_prompts_per_user_day": {
        "name": "P95 prompts/user/day",
        "category": "Analytics · Engagement",
        "definition": "95th percentile of prompts per user-day (how intense the top users are).",
        "population": 'User-days in loaded traces',
        "numerator": '95th percentile prompts per (user_id, date)',
        "denominator": 'User-days',
        "unit": 'prompts/user/day',
        "method": 'deterministic',
        "code_refs": ['tabs.analytics.render'],
        "formula": "Compute prompts per (user, day), then take the 95th percentile.",
        "provenance": "Derived from traces grouped by (`date`, `user_id`).",
        "caveats": [
            "Needs enough user-days for this percentile to be stable.",
        ],
        "used_in": ["Analytics"],
    },

    # -------------------------
    # Content KPIs (deterministic)
    # -------------------------
    "complete_answer_rate_scored_intents": {
        "name": "Complete (scored)",
        "category": "Content KPIs · Deterministic",
        "definition": "Among *scored intents*, share of turns classified as `complete_answer`.",
        "spec_refs": ["content_kpis_classification"],
        "population": 'Turns in scored intents',
        "numerator": "Turns where completion_state == 'complete_answer'",
        "denominator": 'Turns in scored intents',
        "unit": '%',
        "method": 'deterministic',
        "code_refs": ['utils.content_kpis.compute_derived_interactions', 'utils.content_kpis.summarize_content'],
        "formula": "`(# turns where completion_state == 'complete_answer') / (# turns in scored intents)`",
        "provenance": "Derived by `utils.content_kpis.compute_derived_interactions()` then summarized by `summarize_content()`.",
        "caveats": [
            "'Scored intents' are currently `trend_over_time` and `data_lookup`.",
            "Completion is structural/heuristic, not correctness.",
        ],
        "used_in": ["Content KPIs", "Thread QA"],
    },
    "needs_user_input_rate_scored_intents": {
        "name": "Needs input (scored)",
        "category": "Content KPIs · Deterministic",
        "definition": "Among *scored intents*, share of turns where the assistant asks for missing required info (AOI/time/dataset).",
        "spec_refs": ["content_kpis_classification"],
        "population": 'Turns in scored intents',
        "numerator": "Turns where completion_state == 'needs_user_input'",
        "denominator": 'Turns in scored intents',
        "unit": '%',
        "method": 'deterministic',
        "code_refs": ['utils.content_kpis.compute_derived_interactions', 'utils.content_kpis.summarize_content'],
        "formula": "`(# turns where completion_state == 'needs_user_input') / (# turns in scored intents)`",
        "provenance": "Derived deterministically via missing-structure checks + response-text heuristics.",
        "caveats": [
            "Some follow-up questions are legitimate; this metric is best read alongside intent/context.",
        ],
        "used_in": ["Content KPIs", "Thread QA"],
    },
    "error_rate_scored_intents": {
        "name": "Errors (scored)",
        "category": "Content KPIs · Deterministic",
        "definition": "Among *scored intents*, share of turns classified as `error` (missing/failed output).",
        "spec_refs": ["content_kpis_classification"],
        "population": 'Turns in scored intents',
        "numerator": "Turns where completion_state == 'error'",
        "denominator": 'Turns in scored intents',
        "unit": '%',
        "method": 'deterministic',
        "code_refs": ['utils.content_kpis.compute_derived_interactions', 'utils.content_kpis.summarize_content'],
        "formula": "`(# turns where completion_state == 'error') / (# turns in scored intents)`",
        "provenance": "Derived from trace output parsing (`output_json_ok`) + response presence checks.",
        "caveats": [
            "This is separate from Analytics ERROR/Soft-error rates; different heuristics.",
        ],
        "used_in": ["Content KPIs", "Thread QA"],
    },
    "global_dataset_identifiable_rate_scored_intents": {
        "name": "Dataset identifiable (scored)",
        "category": "Content KPIs · Deterministic",
        "definition": "Among *scored intents*, share of turns where a dataset/layer can be identified in the structured output.",
        "spec_refs": ["content_kpis_classification"],
        "population": 'Turns in scored intents',
        "numerator": 'Scored turns where dataset_struct is True',
        "denominator": 'Turns in scored intents',
        "unit": '%',
        "method": 'deterministic',
        "code_refs": ['utils.content_kpis.compute_derived_interactions', 'utils.content_kpis.summarize_content'],
        "formula": "`(# scored turns with dataset_struct == True) / (# scored turns)`",
        "provenance": "Dataset name is extracted from tool output JSON via key-heuristics (e.g., `datasetName`, `layer`, `collection`).",
        "caveats": [
            "Heuristic extraction can miss datasets if schema changes.",
            "A dataset can be mentioned in text but absent from output JSON; this metric focuses on structured identifiability.",
        ],
        "used_in": ["Content KPIs"],
    },
    "citations_shown_rate_scored_intents": {
    "name": "Citations shown (scored)",
    "category": "Content KPIs · Global",
    "spec_refs": ["content_kpis_classification"],
    "definition": (
        "Share of scored data interactions where the assistant included a user-visible citation in its response "
        "(e.g., URL/DOI or an explicit 'Source:' reference)."
    ),
    "population": "Turns in scored intents",
    "numerator": "Scored turns where citations_text is True",
    "denominator": "Turns in scored intents",
    "unit": "%",
    "method": "deterministic",
    "code_refs": ["utils.content_kpis.compute_derived_interactions", "utils.content_kpis.summarize_content"],
    "formula": "citations_shown_rate_scored_intents = mean(citations_text == True) over scored intents",
    "provenance": "Derived from assistant response text; does not rely on structured tool metadata.",
    "caveats": [
        "Heuristic detection — can miss citations that are not URL/DOI/'Source:' style.",
        "Not meaningful for non-data intents; computed on scored intents only.",
    ],
    "used_in": ["Content KPIs"],
    },
"citation_metadata_present_rate_scored_intents": {
    "name": "Citation metadata present (scored)",
    "category": "Content KPIs · Global",
    "spec_refs": ["content_kpis_classification"],
    "definition": (
        "Share of scored data interactions where the tool output included structured citation metadata "
        "(e.g., dataset citation fields), regardless of whether it was rendered in the assistant's text."
    ),
    "population": "Turns in scored intents",
    "numerator": "Scored turns where citations_struct or dataset_has_citation is True",
    "denominator": "Turns in scored intents",
    "unit": "%",
    "method": "deterministic",
    "code_refs": ["utils.content_kpis.compute_derived_interactions", "utils.content_kpis.summarize_content"],
    "formula": "citation_metadata_present_rate_scored_intents = mean(citations_struct OR dataset_has_citation) over scored intents",
    "provenance": "Derived from structured tool output fields (e.g., dataset citation metadata).",
    "caveats": [
        "High values do not guarantee the citation was shown to the user in the response text.",
        "Not meaningful for non-data intents; computed on scored intents only.",
    ],
    "used_in": ["Content KPIs"],
},
    "threads_ended_after_needs_user_input_rate": {
        "name": "Threads ending in needs-input",
        "category": "Content KPIs · Deterministic",
        "definition": "Share of conversation threads whose *last* turn was classified as `needs_user_input`.",
        "spec_refs": ["content_kpis_classification"],
        "population": 'Conversation threads in the loaded traces',
        "numerator": 'Threads whose last turn is needs_user_input',
        "denominator": 'Total threads',
        "unit": '%',
        "method": 'deterministic',
        "code_refs": ['utils.content_kpis.build_thread_summary', 'utils.content_kpis.compute_thread_key'],
        "formula": "For each thread: take last turn → count if needs_user_input; divide by total threads.",
        "provenance": "Derived by grouping derived interactions by thread key (thread_id → sessionId → trace_id fallback).",
        "caveats": [
            "This is a churn proxy; a thread can end for many reasons (user satisfied, user left, etc.).",
            "Requires correct thread grouping; missing IDs can affect results.",
        ],
        "used_in": ["Content KPIs", "Thread QA"],
    },

    # -------------------------
    # Thread QA
    # -------------------------
    "threads_total": {
        "name": "Threads",
        "category": "Thread QA · Rollups",
        "definition": "Total number of thread summaries computed from the loaded traces.",
        "population": 'Conversation threads in the loaded traces',
        "numerator": 'Distinct thread keys',
        "denominator": 'N/A',
        "unit": 'count',
        "method": 'deterministic',
        "code_refs": ['utils.content_kpis.build_thread_summary'],
        "formula": "Group derived interactions by `thread_key` and count groups.",
        "provenance": "Derived by `utils.content_kpis.build_thread_summary()`.",
        "caveats": [
            "Thread grouping uses `compute_thread_key()`; if identifiers are missing, grouping may be imperfect.",
        ],
        "used_in": ["Thread QA"],
    },
    "threads_ended_after_needs_input": {
        "name": "Ended after needs input",
        "category": "Thread QA · Rollups",
        "definition": "Count of threads whose last turn is `needs_user_input`.",
        "population": 'Conversation threads in the loaded traces',
        "numerator": "Threads where last_completion_state == 'needs_user_input'",
        "denominator": 'N/A',
        "unit": 'count',
        "method": 'deterministic',
        "code_refs": ['utils.content_kpis.build_thread_summary'],
        "formula": "`sum(last_completion_state == 'needs_user_input')`",
        "provenance": "Derived by `build_thread_summary()` using the last turn per thread.",
        "caveats": [
            "This does not mean the user is unhappy; it signals the agent asked for missing info.",
        ],
        "used_in": ["Thread QA"],
    },
    "threads_ended_after_error": {
        "name": "Ended after error",
        "category": "Thread QA · Rollups",
        "definition": "Count of threads whose last turn is `error`.",
        "population": 'Conversation threads in the loaded traces',
        "numerator": "Threads where last_completion_state == 'error'",
        "denominator": 'N/A',
        "unit": 'count',
        "method": 'deterministic',
        "code_refs": ['utils.content_kpis.build_thread_summary'],
        "formula": "`sum(last_completion_state == 'error')`",
        "provenance": "Derived by `build_thread_summary()` using the last turn per thread.",
        "caveats": [
            "Represents 'last observed turn' in the loaded window — partial windows can misclassify endings.",
        ],
        "used_in": ["Thread QA"],
    },
    "threads_never_complete": {
        "name": "Never complete",
        "category": "Thread QA · Rollups",
        "definition": "Count of threads that never contain a `complete_answer` turn.",
        "population": 'Conversation threads in the loaded traces',
        "numerator": 'Threads with no complete_answer turns',
        "denominator": 'N/A',
        "unit": 'count',
        "method": 'deterministic',
        "code_refs": ['utils.content_kpis.build_thread_summary'],
        "formula": "`sum(ever_complete_answer == False)`",
        "provenance": "Derived by `build_thread_summary()` via any-turn checks within each thread.",
        "caveats": [
            "If the selected date range cuts threads mid-way, you may see false 'never complete' threads.",
        ],
        "used_in": ["Thread QA"],
    },
    "median_turns_per_thread": {
        "name": "Median turns/thread",
        "category": "Thread QA · Rollups",
        "definition": "Median number of turns (traces) per thread in the loaded data.",
        "population": 'Conversation threads in the loaded traces',
        "numerator": 'Median number of turns per thread',
        "denominator": 'N/A',
        "unit": 'turns/thread',
        "method": 'deterministic',
        "code_refs": ['utils.content_kpis.build_thread_summary'],
        "formula": "`median(n_turns)` over thread summaries",
        "provenance": "Derived by `build_thread_summary()`.",
        "caveats": [
            "Depends on how traces are grouped into threads.",
        ],
        "used_in": ["Thread QA"],
    },

    # -------------------------
    # Eval Insights (human evaluation)
    # -------------------------
    "queue_completed_items": {
        "name": "Completed (queue)",
        "category": "Eval Insights · Queue",
        "definition": "Number of items in the selected Langfuse annotation queue with status == COMPLETED.",
        "population": 'Items in the selected annotation queue',
        "numerator": 'Completed queue items',
        "denominator": 'Total queue items',
        "unit": 'count',
        "method": 'human_eval',
        "code_refs": ['tabs.eval_insights._calculate_metrics'],
        "formula": "`completed_items = count(queue_item.status == 'COMPLETED')`",
        "provenance": "External: derived from the Langfuse annotation queue item list.",
        "caveats": [
            "Queue completion is a workflow/throughput metric; it is not a quality metric.",
        ],
        "used_in": ["Eval Insights"],
    },
    "queue_pending_items": {
        "name": "Pending (queue)",
        "category": "Eval Insights · Queue",
        "definition": "Number of queue items not yet evaluated (status != COMPLETED).",
        "population": 'Items in the selected annotation queue',
        "numerator": 'Pending queue items',
        "denominator": 'Total queue items',
        "unit": 'count',
        "method": 'human_eval',
        "code_refs": ['tabs.eval_insights._calculate_metrics'],
        "formula": "`pending_items = total_items - completed_items`",
        "provenance": "External: derived from the Langfuse annotation queue item list.",
        "caveats": [
            "Pending includes any non-COMPLETED statuses (e.g., OPEN/IN_PROGRESS), depending on Langfuse queue behavior.",
        ],
        "used_in": ["Eval Insights"],
    },
    "queue_completion_rate": {
        "name": "Queue completion rate",
        "category": "Eval Insights · Queue",
        "definition": "Percent of queue items that have been evaluated.",
        "population": 'Items in the selected annotation queue',
        "numerator": 'Completed queue items',
        "denominator": 'Total queue items',
        "unit": '%',
        "method": 'human_eval',
        "code_refs": ['tabs.eval_insights._calculate_metrics'],
        "formula": "`completion_rate = completed_items / total_items`",
        "provenance": "External: computed from Langfuse queue item list (status-based).",
        "caveats": [
            "This reflects reviewer throughput, not quality.",
        ],
        "used_in": ["Eval Insights"],
    },
    "eval_accuracy": {
        "name": "Accuracy",
        "category": "Eval Insights · Outcomes",
        "definition": "Percent of scored items marked Pass, excluding Unsure.",
        "population": 'Scored items in the selected eval queue',
        "numerator": 'Pass scores (excluding unsure)',
        "denominator": 'Pass + Fail scores',
        "unit": '%',
        "method": 'human_eval',
        "code_refs": ['tabs.eval_insights._calculate_metrics'],
        "formula": "`accuracy = pass / (pass + fail)`",
        "provenance": "Derived from score values normalized into pass/fail/unsure buckets.",
        "caveats": [
            "Interpretation depends on your rubric and scoring schema.",
        ],
        "used_in": ["Eval Insights"],
    },
    "eval_pass_rate": {
        "name": "Pass rate",
        "category": "Eval Insights · Outcomes",
        "definition": "Percent of scored items marked Pass.",
        "population": 'Scored items in the selected eval queue',
        "numerator": 'Pass scores',
        "denominator": 'Total scores',
        "unit": '%',
        "method": 'human_eval',
        "code_refs": ['tabs.eval_insights._calculate_metrics'],
        "formula": "`pass_rate = pass / total_scores`",
        "provenance": "Derived from score values.",
        "caveats": [
            "If your rubric includes 'unsure', pass rate may look lower even if accuracy is high.",
        ],
        "used_in": ["Eval Insights"],
    },
    "eval_fail_rate": {
        "name": "Fail rate",
        "category": "Eval Insights · Outcomes",
        "definition": "Percent of scored items marked Fail.",
        "population": 'Scored items in the selected eval queue',
        "numerator": 'Fail scores',
        "denominator": 'Total scores',
        "unit": '%',
        "method": 'human_eval',
        "code_refs": ['tabs.eval_insights._calculate_metrics'],
        "formula": "`fail_rate = fail / total_scores`",
        "provenance": "Derived from score values.",
        "caveats": [
            "Depending on your scoring schema, '0'/'false'/'no' are treated as fail.",
        ],
        "used_in": ["Eval Insights"],
    },
    "eval_unsure_rate": {
        "name": "Unsure rate",
        "category": "Eval Insights · Outcomes",
        "definition": "Percent of scored items marked Unsure.",
        "population": 'Scored items in the selected eval queue',
        "numerator": 'Unsure scores',
        "denominator": 'Total scores',
        "unit": '%',
        "method": 'human_eval',
        "code_refs": ['tabs.eval_insights._calculate_metrics'],
        "formula": "`unsure_rate = unsure / total_scores`",
        "provenance": "Derived from score values.",
        "caveats": [
            "High unsure rate can indicate an unclear rubric or insufficient context in eval UI.",
        ],
        "used_in": ["Eval Insights"],
    },

    "first_evaluation_date": {
        "name": "First evaluation",
        "category": "Eval Insights · Timeline",
        "definition": "Earliest evaluation timestamp/date observed in the selected queue's score records.",
        "population": 'Scores in the selected eval queue',
        "numerator": 'Earliest evaluation timestamp',
        "denominator": 'N/A',
        "unit": 'date',
        "method": 'human_eval',
        "code_refs": ['tabs.eval_insights._build_scores_dataframe'],
        "formula": "`min(score.timestamp)` (or `created_at`) after parsing timestamps",
        "provenance": "External: derived from Langfuse score objects returned by `fetch_scores_by_queue()`.",
        "caveats": [
            "If timestamps are missing or unparseable, this may be blank or incorrect.",
        ],
        "used_in": ["Eval Insights"],
    },
    "last_evaluation_date": {
        "name": "Last evaluation",
        "category": "Eval Insights · Timeline",
        "definition": "Most recent evaluation timestamp/date observed in the selected queue's score records.",
        "population": 'Scores in the selected eval queue',
        "numerator": 'Most recent evaluation timestamp',
        "denominator": 'N/A',
        "unit": 'date',
        "method": 'human_eval',
        "code_refs": ['tabs.eval_insights._build_scores_dataframe'],
        "formula": "`max(score.timestamp)` (or `created_at`) after parsing timestamps",
        "provenance": "External: derived from Langfuse score objects returned by `fetch_scores_by_queue()`.",
        "caveats": [
            "If you are looking at a partial set of scores (filters/limits), this can be misleading.",
        ],
        "used_in": ["Eval Insights"],
    },
    "avg_evaluations_per_day": {
        "name": "Avg per day",
        "category": "Eval Insights · Timeline",
        "definition": "Average number of evaluations per day across the observed evaluation window.",
        "population": 'Scores in the selected eval queue',
        "numerator": 'Total scores',
        "denominator": 'Days in observed window',
        "unit": 'scores/day',
        "method": 'human_eval',
        "code_refs": ['tabs.eval_insights._render_temporal_insights'],
        "formula": "`len(scores) / days_active`, where `days_active = (max_dt - min_dt).days + 1`",
        "provenance": "Derived from parsed timestamps for the loaded score records.",
        "caveats": [
            "This is averaged over the observed window, not just days with activity.",
        ],
        "used_in": ["Eval Insights"],
    },

    # -------------------------
    # QA Samples
    # -------------------------
    "qa_pack_candidates_uncapped": {
        "name": "Candidates (uncapped)",
        "category": "QA Samples · Sampling",
        "definition": "Number of traces that match the pack definition before applying the row cap.",
        "population": 'Loaded traces',
        "numerator": 'Traces matching the sample pack mask (before cap)',
        "denominator": 'N/A',
        "unit": 'count',
        "method": 'deterministic',
        "code_refs": ['utils.eval_sampling.build_preset_mask'],
        "formula": "`uncapped = sum(mask_for_pack)`",
        "provenance": "Derived deterministically from `utils.eval_sampling.build_preset_mask()`.",
        "caveats": [
            "Candidates are computed from derived interactions; if derived parsing changes, the candidate set can shift.",
        ],
        "used_in": ["QA Sample Packs"],
    },
    "qa_pack_rows_in_export": {
        "name": "Rows in export",
        "category": "QA Samples · Sampling",
        "definition": "Number of rows included in the exported CSV for the selected sample pack.",
        "population": 'Loaded traces',
        "numerator": 'Rows included in exported pack',
        "denominator": 'N/A',
        "unit": 'count',
        "method": 'deterministic',
        "code_refs": ['tabs.qa_samples.render'],
        "formula": "`rows = len(pack_df)`",
        "provenance": "Derived after applying max_rows and pack-specific selection logic.",
        "caveats": [
            "Can be lower than the cap if there are not enough candidates.",
        ],
        "used_in": ["QA Sample Packs"],
    },

    # -------------------------
    # CodeAct Templates
    # -------------------------
    "codeact_traces": {
        "name": "CodeAct traces",
        "category": "CodeAct · Templates",
        "definition": "Number of turns where CodeAct content was detected in the trace output.",
        "population": 'Loaded traces',
        "numerator": 'Turns where CodeAct content is detected',
        "denominator": 'N/A',
        "unit": 'count',
        "method": 'deterministic',
        "code_refs": ['utils.codeact_qaqc.add_codeact_qaqc_columns'],
        "formula": "`n_codeact = count(codeact_present == True)`",
        "provenance": "Derived by `utils.codeact_qaqc.add_codeact_qaqc_columns()`.",
        "caveats": [
            "Detection depends on how CodeAct is encoded in outputs; schema changes may affect this.",
        ],
        "used_in": ["CodeAct Templates"],
    },
    "codeact_templates": {
        "name": "Templates",
        "category": "CodeAct · Templates",
        "definition": "Number of unique CodeAct templates (clusters) inferred from code structure.",
        "population": 'CodeAct turns in the loaded traces',
        "numerator": 'Unique template clusters',
        "denominator": 'N/A',
        "unit": 'count',
        "method": 'deterministic',
        "code_refs": ['utils.codeact_qaqc.build_codeact_template_rollups'],
        "formula": "`n_templates = nunique(codeact_template_id)`",
        "provenance": "Derived by `utils.codeact_qaqc.build_codeact_template_rollups()`.",
        "caveats": [
            "Clustering is deterministic but heuristic; small code changes can produce a new template ID.",
        ],
        "used_in": ["CodeAct Templates"],
    },
    "codeact_consistency_issue_rate": {
        "name": "Consistency issue rate",
        "category": "CodeAct · QA",
        "definition": "Among CodeAct turns, share with a parameter-consistency issue (AOI/time/dataset mismatch signals).",
        "population": 'CodeAct turns in the loaded traces',
        "numerator": 'CodeAct turns with a consistency issue signal',
        "denominator": 'CodeAct turns',
        "unit": '%',
        "method": 'deterministic',
        "code_refs": ['utils.codeact_qaqc.add_codeact_qaqc_columns'],
        "formula": "`mean(codeact_consistency_issue == True)` over CodeAct turns",
        "provenance": "Derived by deterministic checks in `utils.codeact_qaqc`.",
        "caveats": [
            "This is a 'signal' metric; it can produce false positives and is best used for sampling/drilldown.",
        ],
        "used_in": ["CodeAct Templates"],
    },
}


# ---------------------------------------------------------------------------
# Classification specs (executable documentation)
# ---------------------------------------------------------------------------

# These are human-readable specs for *derived labels* used by KPI computations.
# They are meant to be paired with spec-by-example tests in `tests/`.

CLASSIFICATION_SPECS: dict[str, dict[str, Any]] = {
    "content_kpis_classification": {
        "name": "Content KPIs: deterministic classification spec",
        "applies_to_metrics": [
            "complete_answer_rate_scored_intents",
            "needs_user_input_rate_scored_intents",
            "error_rate_scored_intents",
            "global_dataset_identifiable_rate_scored_intents",
            "citations_shown_rate_scored_intents",
            "citation_metadata_present_rate_scored_intents",
            "threads_ended_after_needs_user_input_rate",
        ],
        "markdown": """
This spec documents the deterministic rules used to derive **Content KPI** labels.

The implementation lives in `utils/content_kpis.py` and is enforced by the executable
spec tests in `tests/test_content_kpis_classification_spec.py`.

### Derived table
Content KPIs are computed from a per-turn derived table produced by:

- `compute_derived_interactions(traces)` → one row per prompt-bearing turn
- `summarize_content(derived)` → KPI rates and denominators

### Scored intents
The headline Content KPIs are restricted to:

`SCORED_INTENTS = {"trend_over_time", "data_lookup"}`

Intent is inferred from the prompt text, **but can be re-cast** when the system clearly
executed analysis (raw_data/charts) and returned a structured dataset:

- if `intent_primary ∈ {other, parameter_refinement}` AND `analysis_executed == True` AND `dataset_struct == True`,
  then `intent_primary` is set to `trend_over_time` (if prompt looks like a trend request) else `data_lookup`.

### Structural flags (`*_struct`)
Structural flags are extracted from tool output JSON (`output`):

- `aoi_selected_struct`: True if output includes a selected AOI under `selected_aoi`/`selectedAOI`/`aoi`
  with a non-empty name/id/gadm_id/etc.
- `time_range_struct`: True if both `start_date` and `end_date` (or camelCase variants) are present.
- `dataset_struct`: True if a dataset/layer name can be extracted from dataset/layer/collection containers.
- `citations_struct`: True if output contains `citation`/`citations`/`sources`/`sourceUrls`/`references`.
- `dataset_has_citation`: True if the dataset object contains a non-empty `citation(s)` string.

### `answer_type` (coarse response outcome)
`answer_type = _answer_type(response, response_missing, output_json_ok, level, errorCount)`

Priority order:

1) `missing_output` if the assistant response for the turn is missing.
2) `model_error` if any of:
   - Langfuse `level == ERROR`,
   - invalid output JSON with `level == ERROR`,
   - `errorCount > 0`,
   - strong system-failure markers in the text (traceback/timeout/service unavailable/etc.).
3) `empty_or_short` if stripped response length `< 20` characters.
4) `no_data` if the response matches explicit no-results / unsupported-coverage patterns.
5) `fallback` for meta disclaimers (e.g., "as an AI...", "can't access real time").
6) otherwise `answer`.

**Correction:** if `answer_type == no_data` but `analysis_executed == True`, the label is corrected to `answer`
and `no_data_with_analysis=True` is set for QA filtering.

### `needs_user_input`
`needs_user_input, reason = _needs_user_input(response, requires, struct)`

This is a **conservative, deterministic** signal. It triggers when either:

- required fields are missing according to inferred `requires_*` flags **OR**
- the response explicitly asks for missing AOI/time/dataset (regex-gated by domain terms), including AOI disambiguation.

Reason codes:

- `missing_aoi`, `missing_time`, `missing_dataset`, `multiple_missing`

If the assistant explicitly asks, that asked-field reason is preferred over the inferred missing-required reason.

### `completion_state` (the label used by Content KPIs)
`completion_state, struct_good_trend, struct_good_lookup, struct_fail_reason = _completion_state(...)`

Priority order:

1) `error` if `answer_type ∈ {missing_output, model_error, empty_or_short}`.
   - `struct_fail_reason` becomes the specific `answer_type` token.
2) `needs_user_input` if `needs_user_input == True`.
3) `no_data` if `answer_type == no_data`.
4) `incomplete_answer` if intent is scored and any structural reasons apply:
   - missing required AOI/time/dataset (`missing_aoi|missing_time|missing_dataset`)
   - plus for `trend_over_time` with `requires_data == True`: `no_citation` if no citations and not needs_user_input
5) `complete_answer` if `answer_type ∈ {answer, text_only}` and none of the above.
6) otherwise `other`.

`struct_fail_reason` is a `|`-joined list of failure tokens.

### Citations used for completion
The citation presence used in `completion_state` is:

`citations_any = citations_struct OR citations_text OR dataset_has_citation`

**Note:** `citations_text` currently scans **prompt + response** together (`combined = f"{prompt}\n{response}"`).
This can credit user-provided citations; interpret "Citations shown" accordingly.

### Thread grouping
Thread-level Content KPIs group by `compute_thread_key()` priority:

`thread_id → sessionId/session_id → trace_id → row index fallback`.
""",
        "code_refs": [
            "utils.content_kpis.compute_derived_interactions",
            "utils.content_kpis._extract_struct_flags",
            "utils.content_kpis._infer_requires",
            "utils.content_kpis._answer_type",
            "utils.content_kpis._needs_user_input",
            "utils.content_kpis._completion_state",
            "utils.content_kpis.compute_thread_key",
        ],
        "test_refs": [
            "tests/test_content_kpis_classification_spec.py",
        ],
    }
}


# ---------------------------------------------------------------------------
# Page-level documentation
# ---------------------------------------------------------------------------

PAGES: dict[str, dict[str, Any]] = {
    "analytics": {
        "title": "📊 Trace Analytics",
        "what": [
            "A high-level report over the *currently loaded* traces: volume, outcomes, performance, language, and tool usage.",
            "Use this page for weekly/monthly reporting and to spot regressions (latency/cost/error spikes).",
        ],
        "data": [
            "Input dataset = `st.session_state.stats_traces` (fetched from the sidebar).",
            "Outcome labels (ANSWER/DEFER/SOFT_ERROR/ERROR) are **heuristics** from `utils.trace_parsing.classify_outcome()`.",
            "Cost/latency come directly from Langfuse trace fields when present.",
            "New vs Returning user metrics require an *additional* all-time scan (button on this page).",
        ],
        "key_metrics": [
            "total_traces",
            "unique_threads",
            "unique_users",
            "success_rate",
            "error_rate",
            "empty_rate",
            "user_visible_error_rate",
            "internal_tool_error_rate",
            "tool_failure_rate",
            "mean_latency",
            "mean_cost",
            "median_cost",
        ],
        "pitfalls": [
            "A 'trace' ≠ a 'conversation'. A single thread can include many traces (turns).",
            "Heuristic outcomes are useful for trends, but don't substitute for human eval on correctness.",
            "If your date window cuts through ongoing threads, thread-level interpretations can be misleading.",
        ],
    },
    "content_kpis": {
        "title": "🧱 Content KPIs (Deterministic)",
        "what": [
            "Deterministic structural KPIs computed from trace text + tool output JSON (no LLM scoring).",
            "Use this page to track missing-parameter friction (AOI/time/dataset), citation presence, and structural completeness.",
        ],
        "data": [
            "Input dataset = loaded traces (same session).",
            "Turns are reduced to a derived row-level table via `compute_derived_interactions()`.",
            "'Scored intents' are currently: `trend_over_time`, `data_lookup`.",
        ],
        "key_metrics": [
            "complete_answer_rate_scored_intents",
            "needs_user_input_rate_scored_intents",
            "error_rate_scored_intents",
            "global_dataset_identifiable_rate_scored_intents",
            "citations_shown_rate_scored_intents",
            "citation_metadata_present_rate_scored_intents",
            "threads_ended_after_needs_user_input_rate",
        ],
        "pitfalls": [
            "These metrics are structural and heuristic — they do not measure factual correctness.",
            "Dataset/citation detection uses key + text heuristics; schema changes can affect rates.",
        ],
    },
    "thread_qa": {
        "title": "🧵 Thread QA",
        "what": [
            "Thread-level rollups computed deterministically from derived interactions.",
            "Use this page to find threads that ended with missing-info requests or errors, and to drill into their turns.",
        ],
        "data": [
            "Thread summaries come from `build_thread_summary(derived)`.",
            "Thread grouping uses `compute_thread_key()` (thread_id → sessionId → trace_id fallback).",
        ],
        "key_metrics": [
            "threads_total",
            "threads_ended_after_needs_input",
            "threads_ended_after_error",
            "threads_never_complete",
            "median_turns_per_thread",
        ],
        "pitfalls": [
            "If the selected date range captures only part of a conversation, 'ended after X' reflects the last turn in-window.",
        ],
    },
    "eval_insights": {
        "title": "📈 Eval Insights",
        "what": [
            "Dashboard of human evaluation results pulled from Langfuse annotation queues.",
            "Use this page to track reviewer throughput and rubric outcomes over time.",
        ],
        "data": [
            "External: pulls queue items + scores from Langfuse APIs.",
            "Outcome buckets (pass/fail/unsure) are inferred from score values.",
        ],
        "key_metrics": [
            "queue_completion_rate",
            "eval_accuracy",
            "eval_pass_rate",
            "eval_fail_rate",
            "eval_unsure_rate",
        ],
        "pitfalls": [
            "Interpretation depends on which score config/rubric you select.",
        ],
    },
    "qa_samples": {
        "title": "📦 QA Sample Packs",
        "what": [
            "Deterministic sample packs that match pipeline exports (useful for consistent QA slices).",
        ],
        "data": [
            "Sampling masks come from `utils.eval_sampling` over derived interactions.",
        ],
        "key_metrics": [
            "qa_pack_candidates_uncapped",
            "qa_pack_rows_in_export",
        ],
        "pitfalls": [
            "If you change derived-interaction logic, pack membership can change.",
        ],
    },
    "codeact_templates": {
        "title": "🧩 CodeAct Templates",
        "what": [
            "Deterministic clustering of CodeAct code blocks into templates, plus parameter-consistency QA signals.",
        ],
        "data": [
            "Requires CodeAct content to be present in trace outputs.",
            "Adds derived columns via `utils.codeact_qaqc.add_codeact_qaqc_columns()`.",
        ],
        "key_metrics": [
            "codeact_traces",
            "codeact_templates",
            "codeact_consistency_issue_rate",
        ],
        "pitfalls": [
            "Consistency checks are heuristic signals; use drilldown to confirm.",
        ],
    },
}
