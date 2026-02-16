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
        "definition": "Distinct conversation sessions (sessionId) in the loaded traces.",
        "formula": "`unique_threads = nunique(session_id)`",
        "provenance": "Derived from the trace field `sessionId` after normalization (`normalize_trace_format`).",
        "caveats": [
            "If `sessionId` is missing in traces, this will undercount threads.",
        ],
        "used_in": ["Analytics"],
    },
    "unique_users": {
        "name": "Unique users",
        "category": "Analytics · Volume",
        "definition": "Distinct user IDs that generated at least one trace in the loaded window.",
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
        "formula": "`success_rate = mean(outcome == 'ANSWER')`",
        "provenance": "Outcome is derived by `utils.trace_parsing.classify_outcome()` using final answer text + tool usage.",
        "caveats": [
            "This is a heuristic, not a human judgment of correctness.",
            "Traces with valid-looking text but no tool calls may be counted as DEFER instead of ANSWER.",
        ],
        "used_in": ["Analytics"],
    },
    "defer_rate": {
        "name": "Defer rate",
        "category": "Analytics · Outcomes",
        "definition": "Share of traces where the assistant responded but did not use tools.",
        "formula": "`defer_rate = mean(outcome == 'DEFER')`",
        "provenance": "Derived by `classify_outcome()` when answer text is non-empty and `trace_used_tools(trace)` is False.",
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
        "definition": "Share of traces with an empty final answer.",
        "formula": "`error_rate = mean(outcome == 'ERROR')`",
        "provenance": "Derived by `classify_outcome()` when the final answer text is empty/whitespace.",
        "caveats": [
            "Does not capture errors where the model returns text but it is still unusable.",
        ],
        "used_in": ["Analytics"],
    },
    "mean_cost": {
        "name": "Mean cost",
        "category": "Analytics · Performance",
        "definition": "Average LLM cost per trace (in USD), over traces with cost recorded.",
        "formula": "`mean_cost = mean(total_cost)`",
        "provenance": "Raw from trace field `totalCost` (normalized to float).",
        "caveats": [
            "If `totalCost` is missing for some traces, they are excluded from the mean.",
        ],
        "used_in": ["Analytics"],
    },
    "p95_cost": {
        "name": "P95 cost",
        "category": "Analytics · Performance",
        "definition": "95th percentile cost per trace (USD).",
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
        "definition": (
            "Share of scored data interactions where the assistant included a user-visible citation in its response "
            "(e.g., URL/DOI or an explicit 'Source:' reference)."
        ),
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
        "definition": (
            "Share of scored data interactions where the tool output included structured citation metadata "
            "(e.g., dataset citation fields), regardless of whether it was rendered in the assistant's text."
        ),
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
        "definition": "Number of items in the selected Langfuse annotation queue that have at least one score.",
        "formula": "`completed_items = total_items - pending_items` (based on items with score data)",
        "provenance": "External: requires fetching queue items + their scores via Langfuse APIs.",
        "caveats": [
            "Exact definition depends on which score(s) you load (some queues track multiple rubrics).",
        ],
        "used_in": ["Eval Insights"],
    },
    "queue_pending_items": {
        "name": "Pending (queue)",
        "category": "Eval Insights · Queue",
        "definition": "Number of queue items not yet evaluated (no score recorded).",
        "formula": "`pending_items = total_items - completed_items`",
        "provenance": "External: derived from queue item list + score presence.",
        "caveats": [
            "If you change which score name/config you analyze, pending vs completed can change.",
        ],
        "used_in": ["Eval Insights"],
    },
    "queue_completion_rate": {
        "name": "Queue completion rate",
        "category": "Eval Insights · Queue",
        "definition": "Percent of queue items that have been evaluated.",
        "formula": "`completion_rate = completed_items / total_items`",
        "provenance": "External: computed from Langfuse queue item list.",
        "caveats": [
            "This reflects reviewer throughput, not quality.",
        ],
        "used_in": ["Eval Insights"],
    },
    "eval_accuracy": {
        "name": "Accuracy",
        "category": "Eval Insights · Outcomes",
        "definition": "Percent of scored items marked Pass, excluding Unsure.",
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
        "formula": "`mean(codeact_consistency_issue == True)` over CodeAct turns",
        "provenance": "Derived by deterministic checks in `utils.codeact_qaqc`.",
        "caveats": [
            "This is a 'signal' metric; it can produce false positives and is best used for sampling/drilldown.",
        ],
        "used_in": ["CodeAct Templates"],
    },
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
            "mean_latency",
            "mean_cost",
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
