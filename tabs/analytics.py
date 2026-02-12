"""Trace Analytics Reports tab."""
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st

from utils import (
    normalize_trace_format,
    parse_trace_dt,
    first_human_prompt,
    final_ai_message,
    classify_outcome,
    active_turn_prompt,
    active_turn_answer,
    extract_trace_context,
    extract_tool_calls_and_results,
    extract_tool_flow,
    extract_usage_metadata,
    trace_has_internal_error,
    as_float,
    csv_bytes_any,
    daily_volume_chart,
    daily_outcome_chart,
    daily_cost_chart,
    daily_latency_chart,
    outcome_pie_chart,
    language_bar_chart,
    latency_histogram,
    cost_histogram,
    category_pie_chart,
    tool_success_rate_chart,
    tool_calls_vs_latency_chart,
    tool_flow_sankey_data,
    reasoning_tokens_histogram,
    classify_user_segments,
    build_first_seen_lookup,
    build_daily_user_segments,
    compute_engaged_users,
    UserSegments,
)


def render(
    base_url: str,
    base_thread_url: str,
    gemini_api_key: str,
    use_date_filter: bool,
    start_date,
    end_date,
    envs: list[str],
    stats_max_traces: int,
) -> None:
    """Render the Trace Analytics Reports tab."""

    def _format_report_date(d: Any) -> str:
        try:
            if isinstance(d, datetime):
                dt = d
            else:
                dt = datetime(d.year, d.month, d.day)
            day = int(dt.day)
            if 11 <= (day % 100) <= 13:
                suffix = "th"
            else:
                suffix = {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")
            return dt.strftime("%a ") + f"{day}{suffix} " + dt.strftime("%b")
        except Exception:
            return str(d)

    start_date_label = _format_report_date(start_date)
    end_date_label = _format_report_date(end_date)
    st.subheader("📊 Trace Analytics")
    st.caption(
        "Explore aggregate volume, outcomes, latency, cost, languages, tool usage, and errors across the currently loaded traces. "
        "Use filters and exports to share a report with others."
    )

    st.markdown(
        """
<style>
div[data-testid="stMetric"] * { line-height: 1.1; }
div[data-testid="stMetric"] [data-testid="stMetricLabel"] { font-size: 0.75rem; }
div[data-testid="stMetric"] [data-testid="stMetricValue"] { font-size: 1.0rem; }
div[data-testid="stMetric"] [data-testid="stMetricDelta"] { font-size: 0.75rem; }
</style>
        """,
        unsafe_allow_html=True,
    )

    if "stats_traces" not in st.session_state:
        st.session_state.stats_traces = []

    if "analytics_user_first_seen" not in st.session_state:
        st.session_state.analytics_user_first_seen = None
    if "analytics_user_first_seen_debug" not in st.session_state:
        st.session_state.analytics_user_first_seen_debug = {}

    traces: list[dict[str, Any]] = st.session_state.stats_traces
    if not traces:
        st.info(
            "This tab gives you a high-level view of the currently loaded trace dataset: volumes, outcomes, latency, cost, "
            "and usage patterns.\n\n"
            "1. Use the sidebar **🚀 Fetch traces** button to load a dataset once.\n"
            "2. Switch between tabs to explore different views of the **same** traces without re-fetching."
        )
        return

    raw_traces = st.session_state.get("stats_traces_raw", [])
    n_raw = len(raw_traces)
    n_filtered = len(traces)
    n_excluded = n_raw - n_filtered
    exclude_internal = bool(st.session_state.get("_shadow_exclude_internal", True))
    if exclude_internal and n_excluded > 0:
        st.info(
            f"**{n_raw:,}** raw traces loaded · **{n_excluded:,}** internal-user traces excluded · "
            f"**{n_filtered:,}** traces used for analytics (_{n_filtered/n_raw:,.1%} of raw traces_)"
        )

    normed = [normalize_trace_format(t) for t in traces]

    rows: list[dict[str, Any]] = []
    for n in normed:
        prompt = active_turn_prompt(n) or first_human_prompt(n)
        answer = active_turn_answer(n) or final_ai_message(n)
        dt = parse_trace_dt(n)
        outcome = classify_outcome(n, answer or "")

        ctx = extract_trace_context(n)
        usage = extract_usage_metadata(n)
        has_internal_err = trace_has_internal_error(n)

        rows.append(
            {
                "trace_id": n.get("id"),
                "timestamp": dt,
                "date": dt.date() if dt else None,
                "environment": n.get("environment"),
                "session_id": n.get("sessionId"),
                "user_id": n.get("userId") or (n.get("metadata") or {}).get("user_id") or (n.get("metadata") or {}).get("userId"),
                "latency_seconds": as_float(n.get("latency")),
                "total_cost": as_float(n.get("totalCost")),
                "outcome": outcome,
                "prompt": prompt,
                "answer": answer,
                "aoi_name": ctx.get("aoi_name", ""),
                "aoi_type": ctx.get("aoi_type", ""),
                "datasets_analysed": ", ".join(ctx.get("datasets_analysed", [])),
                "tool_call_count": usage.get("tool_call_count", 0),
                "total_input_tokens": usage.get("total_input_tokens", 0),
                "total_output_tokens": usage.get("total_output_tokens", 0),
                "total_reasoning_tokens": usage.get("total_reasoning_tokens", 0),
                "reasoning_ratio": usage.get("reasoning_ratio", 0.0),
                "has_internal_error": has_internal_err,
            }
        )

    df = pd.DataFrame(rows)
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp", na_position="last")

    if "timestamp" in df.columns:
        ts_parsed = df["timestamp"].dropna()
        if len(df) and not len(ts_parsed):
            st.warning("No timestamps could be parsed from the fetched traces; daily charts may be incomplete.")
        elif len(ts_parsed):
            st.caption(
                f"Timestamp coverage: {len(ts_parsed):,}/{len(df):,} parsed. "
                f"Range: {ts_parsed.min()} → {ts_parsed.max()}"
            )

    if "prompt" in df.columns:
        df["prompt_len_chars"] = df["prompt"].fillna("").astype("string").map(lambda x: len(str(x)))
        df["prompt_len_words"] = (
            df["prompt"]
            .fillna("")
            .astype("string")
            .map(lambda x: len([w for w in str(x).strip().split() if w]))
        )

    try:
        from langid.langid import LanguageIdentifier, model as langid_model

        lang_identifier = LanguageIdentifier.from_modelstring(langid_model, norm_probs=True)

        def _lang_id(x: Any) -> tuple[str, float]:
            if not isinstance(x, str) or len(x.strip()) < 3:
                return ("unknown", 0.0)
            return lang_identifier.classify(x)

        if "prompt" in df.columns and len(df):
            df[["lang_query", "lang_query_conf"]] = (
                df["prompt"].map(_lang_id).apply(pd.Series)
            )
    except Exception:
        pass

    total_traces = int(len(df))
    if "user_id" in df.columns:
        _users_series = df["user_id"].dropna().astype(str).map(lambda x: x.strip())
        _users_series = _users_series.loc[lambda s: s.ne("")]
        unique_users = int(_users_series.nunique())
    else:
        unique_users = 0
    unique_threads = int(df["session_id"].dropna().nunique()) if "session_id" in df.columns else 0

    user_first_seen_df: pd.DataFrame | None = None
    has_user_first_seen = False
    segments = UserSegments()

    if isinstance(st.session_state.get("analytics_user_first_seen"), pd.DataFrame):
        user_first_seen_df = st.session_state.get("analytics_user_first_seen")

    if user_first_seen_df is not None and len(user_first_seen_df):
        has_user_first_seen = True
        segments = classify_user_segments(df, user_first_seen_df, start_date, end_date)

    user_first_seen_total_users = int(len(user_first_seen_df)) if user_first_seen_df is not None else 0
    user_first_seen_new_users = len(segments.new_users)
    user_first_seen_returning_users = len(segments.returning_users)
    user_first_seen_unknown_users = len(segments.unknown_users)
    user_first_seen_filled_from_window = segments.filled_from_window
    engaged_users_total = len(segments.engaged_users)
    not_engaged_users_total = len(segments.not_engaged_users)

    util_user_days = 0
    util_mean_prompts = 0.0
    util_median_prompts = 0.0
    util_p95_prompts = 0.0
    user_day_counts: pd.DataFrame | None = None
    if "date" in df.columns and "user_id" in df.columns:
        base_prompts = df.dropna(subset=["date", "user_id"]).copy()
        base_prompts = base_prompts[base_prompts["user_id"].astype(str).str.strip().ne("")]
        if len(base_prompts):
            user_day_counts = (
                base_prompts.groupby(["date", "user_id"], dropna=True)
                .agg(prompts=("trace_id", "count"))
                .reset_index()
            )
            util_user_days = int(len(user_day_counts))
            s_util = pd.to_numeric(user_day_counts["prompts"], errors="coerce").dropna()
            if len(s_util):
                util_mean_prompts = float(s_util.mean())
                util_median_prompts = float(s_util.median())
                util_p95_prompts = float(s_util.quantile(0.95))

    if total_traces:
        success_rate = float((df["outcome"] == "ANSWER").mean())
        defer_rate = float((df["outcome"] == "DEFER").mean())
        soft_error_rate = float((df["outcome"] == "SOFT_ERROR").mean())
        error_rate = float((df["outcome"] == "ERROR").mean())
        empty_rate = float((df["outcome"] == "EMPTY").mean())
    else:
        success_rate = defer_rate = soft_error_rate = error_rate = empty_rate = 0.0

    cost_s = df["total_cost"].dropna() if "total_cost" in df.columns else pd.Series(dtype=float)
    lat_s = df["latency_seconds"].dropna() if "latency_seconds" in df.columns else pd.Series(dtype=float)

    mean_cost = float(cost_s.mean()) if len(cost_s) else 0.0
    median_cost = float(cost_s.median()) if len(cost_s) else 0.0
    p95_cost = float(cost_s.quantile(0.95)) if len(cost_s) else 0.0
    avg_latency = float(lat_s.mean()) if len(lat_s) else 0.0
    p95_latency = float(lat_s.quantile(0.95)) if len(lat_s) else 0.0

    st.markdown(
        f"### Summary Statistics ({(end_date - start_date).days + 1} days: {start_date_label} to {end_date_label})"
    )

    if not has_user_first_seen:
        st.info("Use the sidebar **👥 Fetch users** button to enable New vs Returning user metrics.")

    user_first_seen_coverage_debug: dict[str, Any] | None = None
    if user_first_seen_df is not None and len(user_first_seen_df) and "user_id" in df.columns:
        try:
            _active_users_dbg = (
                df["user_id"]
                .dropna()
                .astype(str)
                .map(lambda x: x.strip())
                .loc[lambda s: s.ne("")]
            )
            _active_users_set_dbg = set(_active_users_dbg.unique())

            _first_seen_dbg = user_first_seen_df.copy()
            _first_seen_dbg = _first_seen_dbg.dropna(subset=["user_id", "first_seen"])
            _first_seen_dbg["user_id"] = _first_seen_dbg["user_id"].astype(str).map(lambda x: x.strip())
            _first_seen_dbg = _first_seen_dbg[_first_seen_dbg["user_id"].ne("")]
            _first_seen_dbg = _first_seen_dbg[_first_seen_dbg["user_id"].isin(_active_users_set_dbg)]

            _matched = int(_first_seen_dbg["user_id"].nunique())
            _unknown = int(len(_active_users_set_dbg) - len(set(_first_seen_dbg["user_id"].tolist())))
            _min_fs = pd.to_datetime(_first_seen_dbg["first_seen"], errors="coerce", utc=True).min()
            _max_fs = pd.to_datetime(_first_seen_dbg["first_seen"], errors="coerce", utc=True).max()

            user_first_seen_coverage_debug = {
                "active_users_in_loaded_traces": int(len(_active_users_set_dbg)),
                "active_users_with_first_seen": _matched,
                "active_users_missing_first_seen": _unknown,
                "first_seen_min": str(_min_fs) if pd.notna(_min_fs) else None,
                "first_seen_max": str(_max_fs) if pd.notna(_max_fs) else None,
            }
        except Exception:
            user_first_seen_coverage_debug = None

    debug_df = st.session_state.get("analytics_user_first_seen_debug")
    if isinstance(debug_df, dict) and debug_df:
        with st.expander("User fetch debug", expanded=False):
            if isinstance(user_first_seen_coverage_debug, dict) and user_first_seen_coverage_debug:
                st.write(user_first_seen_coverage_debug)
            st.json(debug_df)
    if has_user_first_seen:
        user_lines = (
            f"• Total users (all time): {user_first_seen_total_users:,}\n"
            "**Acquisition**\n"
            f"• New users (since {start_date_label}): {user_first_seen_new_users:,}\n"
            f"• Returning users (since {start_date_label}): {user_first_seen_returning_users:,}\n"
            "**Engagement**\n"
            f"• Engaged users (since {start_date_label}): {engaged_users_total:,}\n"
            f"• Not engaged users (since {start_date_label}): {not_engaged_users_total:,}\n"
        )
    else:
        user_lines = ""

    summary_text = f"""📊 *GNW Trace Analytics Report*
📅 {start_date_label} → {end_date_label} ({(end_date - start_date).days + 1} days)

**Volume**
• Total traces (prompts): {total_traces:,}
• Unique threads (convos): {unique_threads:,}
• Unique users: {unique_users:,}
{user_lines}
**Outcomes**
• Success rate: {success_rate:.1%}
• Defer rate: {defer_rate:.1%}
• Soft error rate: {soft_error_rate:.1%}
• Error rate: {error_rate:.1%}
• Empty rate: {empty_rate:.1%}

**Performance**
• Mean cost: ${mean_cost:.4f}
• Median cost: ${median_cost:.4f}
• p95 cost: ${p95_cost:.4f}
• Mean latency: {avg_latency:.2f}s
• p95 latency: {p95_latency:.2f}s

**Prompt utilisation**
• User-days: {util_user_days:,}
• Mean prompts/user/day: {util_mean_prompts:.2f}
• Median prompts/user/day: {util_median_prompts:.0f}
• p95 prompts/user/day: {util_p95_prompts:.0f}"""

    with st.expander("📋 Copy summary for Slack", expanded=False):
        st.code(summary_text, language=None)

    summary_rows = [
        {"Section": "Volume", "Metric": "Total traces", "Value": f"{total_traces:,}", "Description": "Total number of prompts in the period"},
        {"Section": "Volume", "Metric": "Unique threads", "Value": f"{unique_threads:,}", "Description": "Distinct conversation sessions (multi-turn chats)"},
        {"Section": "Volume", "Metric": "Unique users", "Value": f"{unique_users:,}", "Description": "Distinct user IDs that made at least one prompt"},
    ]
    if has_user_first_seen:
        summary_rows.extend(
            [
                {"Section": "Volume", "Metric": "Total users (all time)", "Value": f"{user_first_seen_total_users:,}", "Description": "Distinct user IDs seen since launch (Sept 17th 2025) in Langfuse"},
                {"Section": "Acquisition", "Metric": f"New users (since {start_date_label})", "Value": f"{user_first_seen_new_users:,}", "Description": f"Users whose first-ever trace was on or after {start_date_label}"},
                {"Section": "Acquisition", "Metric": f"Returning users (since {start_date_label})", "Value": f"{user_first_seen_returning_users:,}", "Description": f"Users whose first-ever trace was before {start_date_label}"},
                {"Section": "Engagement", "Metric": f"Engaged users (since {start_date_label})", "Value": f"{engaged_users_total:,}", "Description": "Users with ≥2 sessions each having ≥2 prompts (new or returning)"},
                {"Section": "Engagement", "Metric": f"Not engaged users (since {start_date_label})", "Value": f"{not_engaged_users_total:,}", "Description": "Users not meeting the engagement threshold"},
            ]
        )

    summary_rows.extend(
        [
            {"Section": "Outcomes", "Metric": "Success rate", "Value": f"{success_rate:.1%}", "Description": "% of traces that returned a valid answer"},
            {"Section": "Outcomes", "Metric": "Defer rate", "Value": f"{defer_rate:.1%}", "Description": "% of traces classified as DEFER (final answer is non-empty/non-error, but the trace shows no tool usage)"},
            {"Section": "Outcomes", "Metric": "Soft error rate", "Value": f"{soft_error_rate:.1%}", "Description": "% of traces classified as SOFT_ERROR (final answer text looks like an error/apology via heuristic matching)"},
            {"Section": "Outcomes", "Metric": "Error rate", "Value": f"{error_rate:.1%}", "Description": "% of traces classified as ERROR (trace has an AI message, but the final extracted answer is empty)"},
            {"Section": "Outcomes", "Metric": "Empty rate", "Value": f"{empty_rate:.1%}", "Description": "% of traces classified as EMPTY (no AI answer message found in output)"},
            {"Section": "Performance", "Metric": "Mean cost", "Value": f"${mean_cost:.4f}", "Description": "Average LLM cost per trace"},
            {"Section": "Performance", "Metric": "Median cost", "Value": f"${median_cost:.4f}", "Description": "Middle value of cost distribution (less sensitive to outliers)"},
            {"Section": "Performance", "Metric": "p95 cost", "Value": f"${p95_cost:.4f}", "Description": "95th percentile cost (only 5% of traces cost more)"},
            {"Section": "Performance", "Metric": "Mean latency", "Value": f"{avg_latency:.2f}s", "Description": "Mean time from request to response"},
            {"Section": "Performance", "Metric": "P95 latency", "Value": f"{p95_latency:.2f}s", "Description": "95th percentile latency (worst-case for most users)"},
            {"Section": "Engagement", "Metric": "User-days", "Value": f"{util_user_days:,}", "Description": "Total user × day combinations (one user on 3 days = 3)"},
            {"Section": "Engagement", "Metric": "Mean prompts/user/day", "Value": f"{util_mean_prompts:.2f}", "Description": "Average number of prompts a user sends per active day"},
            {"Section": "Engagement", "Metric": "Median prompts/user/day", "Value": f"{util_median_prompts:.0f}", "Description": "Typical prompts per user per day (less skewed by power users)"},
            {"Section": "Engagement", "Metric": "p95 prompts/user/day", "Value": f"{util_p95_prompts:.0f}", "Description": "Top 5% of users send this many prompts or more per day"},
        ]
    )

    summary_tbl = pd.DataFrame(summary_rows)
    st.dataframe(summary_tbl, width="stretch", hide_index=True)

    report_rows = [
        {
            **{k: ("" if pd.isna(v) else v) for k, v in r.items()},
            "url": f"{base_thread_url.rstrip('/')}/{r.get('session_id')}" if r.get("session_id") else "",
        }
        for r in df.to_dict("records")
    ]
    report_csv_bytes = csv_bytes_any(report_rows)

    st.download_button(
        label="Download Report Data `.csv`",
        data=report_csv_bytes,
        file_name="stats_report_rows.csv",
        mime="text/csv",
        key="analytics_report_csv",
    )

    with st.expander("Report Data"):
        st.dataframe(df, width="stretch")

    st.markdown("### Prompt utilisation", help="How intensively users are engaging with the system. Higher prompts/user/day suggests stickier product usage.")
    if "date" not in df.columns or "user_id" not in df.columns:
        st.info("Prompt utilisation requires both date and user_id.")
    else:
        base_prompts = df.dropna(subset=["date", "user_id"]).copy()
        base_prompts = base_prompts[base_prompts["user_id"].astype(str).str.strip().ne("")]
        if len(base_prompts):
            user_day_counts = (
                base_prompts.groupby(["date", "user_id"], dropna=True)
                .agg(prompts=("trace_id", "count"))
                .reset_index()
            )
            s = pd.to_numeric(user_day_counts["prompts"], errors="coerce").dropna()
            if len(s):

                left, right = st.columns(2)

                with left:
                    st.markdown(
                        "#### Prompts per user per day (distribution)",
                        help="Histogram of how many prompts users send on an active day. Helps spot power users and typical usage.",
                    )
                    hist = (
                        alt.Chart(user_day_counts)
                        .mark_bar()
                        .encode(
                            x=alt.X(
                                "prompts:Q",
                                bin=alt.Bin(maxbins=30),
                                title="Prompts per user per day",
                            ),
                            y=alt.Y("count():Q", title="User-days"),
                            tooltip=[
                                alt.Tooltip("prompts:Q", title="Prompts/user/day", bin=True),
                                alt.Tooltip("count():Q", title="User-days"),
                            ],
                        )
                        .properties(height=260)
                    )
                    st.altair_chart(hist, width="stretch")

                with right:
                    st.markdown(
                        "#### Prompts per user per day (daily)",
                        help="Daily mean/median/p95 prompts per active user. Good for monitoring engagement and limiting behaviour.",
                    )
                    daily_user_prompt = (
                        user_day_counts.groupby("date", dropna=True)
                        .agg(
                            mean_prompts_per_user=("prompts", "mean"),
                            median_prompts_per_user=("prompts", "median"),
                            p95_prompts_per_user=(
                                "prompts",
                                lambda x: float(pd.to_numeric(x, errors="coerce").dropna().quantile(0.95))
                                if len(pd.to_numeric(x, errors="coerce").dropna())
                                else 0.0,
                            ),
                        )
                        .reset_index()
                        .sort_values("date")
                    )

                    daily_long = daily_user_prompt.melt(
                        id_vars=["date"],
                        value_vars=[
                            "mean_prompts_per_user",
                            "median_prompts_per_user",
                            "p95_prompts_per_user",
                        ],
                        var_name="metric",
                        value_name="value",
                    )
                    daily_long["metric"] = daily_long["metric"].replace(
                        {
                            "mean_prompts_per_user": "Mean",
                            "median_prompts_per_user": "Median",
                            "p95_prompts_per_user": "p95",
                        }
                    )

                    line = (
                        alt.Chart(daily_long)
                        .mark_line(point=True)
                        .encode(
                            x=alt.X("date:T", title="Date"),
                            y=alt.Y("value:Q", title="Prompts per user"),
                            color=alt.Color("metric:N", title="Metric", sort=["Mean", "Median", "p95"]),
                            tooltip=[
                                alt.Tooltip("date:T", title="Date"),
                                alt.Tooltip("metric:N", title="Metric"),
                                alt.Tooltip("value:Q", title="Prompts/user", format=".2f"),
                            ],
                        )
                        .properties(height=260)
                    )

                    prompt_limit = 10
                    limit_rule = (
                        alt.Chart(pd.DataFrame({"prompt_limit": [prompt_limit]}))
                        .mark_rule(color="#e5484d", strokeDash=[6, 4])
                        .encode(y=alt.Y("prompt_limit:Q"))
                    )

                    st.altair_chart(line + limit_rule, width="stretch")
            else:
                st.info("No prompt utilisation data available.")
        else:
            st.info("No prompt utilisation data available.")

    if "date" in df.columns and df["date"].notna().any():
        base_daily = df.dropna(subset=["date"]).copy()
        base_daily["date"] = pd.to_datetime(base_daily["date"], utc=True).dt.date

        def _q(s: pd.Series, q: float) -> float:
            try:
                return float(pd.to_numeric(s, errors="coerce").dropna().quantile(q))
            except Exception:
                return 0.0

        daily_metrics = (
            base_daily.groupby("date", dropna=True)
            .agg(
                traces=("trace_id", "count"),
                unique_users=("user_id", "nunique"),
                unique_threads=("session_id", "nunique"),
                success_rate=("outcome", lambda x: float((x == "ANSWER").mean())),
                defer_rate=("outcome", lambda x: float((x == "DEFER").mean())),
                soft_error_rate=("outcome", lambda x: float((x == "SOFT_ERROR").mean())),
                error_rate=("outcome", lambda x: float((x == "ERROR").mean())),
                empty_rate=("outcome", lambda x: float((x == "EMPTY").mean())),
                mean_cost=("total_cost", lambda x: float(pd.to_numeric(x, errors="coerce").dropna().mean()) if len(pd.to_numeric(x, errors="coerce").dropna()) else 0.0),
                p95_cost=("total_cost", lambda x: _q(x, 0.95)),
                mean_latency=("latency_seconds", lambda x: float(pd.to_numeric(x, errors="coerce").dropna().mean()) if len(pd.to_numeric(x, errors="coerce").dropna()) else 0.0),
                p95_latency=("latency_seconds", lambda x: _q(x, 0.95)),
            )
            .reset_index()
            .sort_values("date")
        )

        st.markdown("### Daily trends", help="Track how key metrics change over time. Look for anomalies, regressions, or improvements day-over-day.")

        vol_chart = daily_volume_chart(daily_metrics)
        out_chart = daily_outcome_chart(daily_metrics, outcome_order=[
            "Error",
            "Error (Empty)",
            "Defer",
            "Soft error",
            "Success",
        ])
        cost_chart = daily_cost_chart(daily_metrics)
        lat_chart = daily_latency_chart(daily_metrics)

        row1_c1, row1_c2 = st.columns(2)
        with row1_c1:
            st.markdown("#### Daily volume", help="Daily traces, unique users, and unique threads.")
            st.altair_chart(vol_chart, width="stretch")
        with row1_c2:
            st.markdown(
                "#### Daily outcomes",
                help=(
                    "Daily mix of outcomes as rates. Outcome rules: ANSWER = non-empty answer + tool usage; "
                    "DEFER = non-empty/non-error answer but no tool usage; SOFT_ERROR = answer text looks like an error via heuristics; "
                    "ERROR = trace has an AI message but the final extracted answer is empty; EMPTY = no AI message found."
                ),
            )
            st.altair_chart(out_chart, width="stretch")

        row2_c1, row2_c2 = st.columns(2)
        with row2_c1:
            st.markdown("#### Daily cost", help="Mean and p95 LLM cost per day.")
            st.altair_chart(cost_chart, width="stretch")
        with row2_c2:
            st.markdown("#### Daily latency", help="Mean and p95 latency per day.")
            st.altair_chart(lat_chart, width="stretch")

        # Additional insight: User activity over time
        if "user_id" in base_daily.columns:
            if has_user_first_seen:
                fs_lookup = segments.first_seen_by_user
                engaged_set = segments.engaged_users
            else:
                fs_lookup, _ = build_first_seen_lookup(base_daily, None, start_date, end_date)
                engaged_set = compute_engaged_users(base_daily)

            daily_segments = build_daily_user_segments(base_daily, fs_lookup, engaged_set)

            if len(daily_segments) > 1:

                # ── Shared helper: build a stacked-bar + pie pair ──
                def _user_segment_charts(
                    daily_df: pd.DataFrame,
                    col_a: str,
                    col_b: str,
                    label_a: str,
                    label_b: str,
                    color_a: str,
                    color_b: str,
                    pie_a: int,
                    pie_b: int,
                ) -> tuple:
                    day_total = daily_df[col_a] + daily_df[col_b]
                    long = daily_df.assign(day_total=day_total).melt(
                        id_vars=["date", "day_total"],
                        value_vars=[col_a, col_b],
                        var_name="user_type",
                        value_name="count",
                    )
                    long["pct_of_day"] = long["count"] / long["day_total"].clip(lower=1)
                    long["user_type"] = long["user_type"].replace({col_a: label_a, col_b: label_b})

                    domain = [label_a, label_b]
                    colors = [color_a, color_b]
                    scale = alt.Scale(domain=domain, range=colors)

                    bar = (
                        alt.Chart(long)
                        .mark_bar(size=17)
                        .encode(
                            x=alt.X("date:T", title="Date"),
                            y=alt.Y("count:Q", title="Users", stack=True),
                            color=alt.Color("user_type:N", title="User type", sort=domain, scale=scale),
                            tooltip=[
                                alt.Tooltip("date:T", title="Date"),
                                alt.Tooltip("user_type:N", title="Type"),
                                alt.Tooltip("count:Q", title="Users", format=","),
                                alt.Tooltip("pct_of_day:Q", title="% of day", format=".1%"),
                            ],
                        )
                        .properties(height=220)
                    )

                    pie_df = pd.DataFrame([
                        {"user_type": label_a, "count": pie_a},
                        {"user_type": label_b, "count": pie_b},
                    ])
                    pie_df = pie_df[pie_df["count"] > 0]
                    pie_df["percent"] = pie_df["count"] / max(1, int(pie_df["count"].sum())) * 100

                    pie = (
                        alt.Chart(pie_df)
                        .mark_arc(innerRadius=55)
                        .encode(
                            theta=alt.Theta("count:Q", title="Users"),
                            color=alt.Color("user_type:N", title="", sort=domain, scale=scale),
                            tooltip=[
                                alt.Tooltip("user_type:N", title="Type"),
                                alt.Tooltip("count:Q", title="Users", format=","),
                                alt.Tooltip("percent:Q", title="%", format=".1f"),
                            ],
                        )
                        .properties(height=220)
                    )
                    return bar, pie

                # ── Shared expander explaining both insights ──
                st.markdown("#### User segmentation")
                with st.expander("ℹ️ How are these categories defined?", expanded=False):
                    st.markdown(
                        f"""
Users are classified along **two independent dimensions**. Each is a simple binary split.

**Dimension 1 — Acquisition (New vs Returning):**

| Category | Rule |
|----------|------|
| **New** | User whose very first trace ever was **on or after {start_date}** |
| **Returning** | User whose first trace was **before {start_date}** |

On the **daily chart**, a user is "New" only on the day matching their `first_seen_date`.
On every subsequent active day they become "Returning".

**Dimension 2 — Engagement (Engaged vs Not Engaged):**

| Category | Rule |
|----------|------|
| **Engaged** | User with **≥ 2 sessions**, each containing **≥ 2 prompts** |
| **Not Engaged** | Everyone else |

Engagement applies to **all** users — a New user can be Engaged if they hit the threshold within the date range.

**Pie chart vs daily chart:**

The **pie / summary table** counts each user **once** across the entire range.
The **daily chart** counts each user **once per active day**, so a user active on 5 days adds 5 to the daily total but only 1 to the pie.
"""
                    )

                # ── Insight 1: New vs Returning ──
                pie_new = len(segments.new_users) if has_user_first_seen else int(
                    daily_segments["new_users"].sum()
                )
                pie_returning = len(segments.returning_users) if has_user_first_seen else int(
                    daily_segments["returning_users"].sum()
                )
                nr_bar, nr_pie = _user_segment_charts(
                    daily_segments,
                    "new_users", "returning_users",
                    "New", "Returning",
                    "#98a2b3", "#2e90fa",
                    pie_new, pie_returning,
                )
                nr_c1, nr_c2 = st.columns(2)
                with nr_c1:
                    st.markdown(
                        "##### Daily new vs returning",
                        help="New = first trace on that day. Returning = first trace before that day.",
                    )
                    st.altair_chart(nr_bar, width="stretch")
                with nr_c2:
                    st.markdown(
                        "##### Total new vs returning",
                        help="Unique users in the date range. New + Returning = total known users.",
                    )
                    st.altair_chart(nr_pie, width="stretch")

                # ── Insight 2: Engaged vs Not Engaged ──
                pie_engaged = len(segments.engaged_users) if has_user_first_seen else int(
                    daily_segments["engaged_users"].sum()
                )
                pie_not_engaged = len(segments.not_engaged_users) if has_user_first_seen else int(
                    daily_segments["not_engaged_users"].sum()
                )
                eng_bar, eng_pie = _user_segment_charts(
                    daily_segments,
                    "not_engaged_users", "engaged_users", 
                    "Not Engaged", "Engaged",
                    "#d0d5dd", "#12b76a", 
                    pie_not_engaged, pie_engaged,
                )
                eng_c1, eng_c2 = st.columns(2)
                with eng_c1:
                    st.markdown(
                        "##### Daily engaged vs not engaged",
                        help="Engaged = ≥2 sessions with ≥2 prompts each. Applies to all users.",
                    )
                    st.altair_chart(eng_bar, width="stretch")
                with eng_c2:
                    st.markdown(
                        "##### Total engaged vs not engaged",
                        help="Unique users in the date range. Engaged + Not Engaged = total known users.",
                    )
                    st.altair_chart(eng_pie, width="stretch")

    def _norm_prompt(s: Any) -> str:
        if not isinstance(s, str):
            return ""
        out = " ".join(s.strip().split()).lower()
        while out.endswith("."):
            out = out[:-1].rstrip()
        return out

    starter_path = Path(__file__).resolve().parents[1] / "starter-prompts.json"
    starter_prompts: list[str] = []
    try:
        payload = json.loads(starter_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and isinstance(payload.get("prompts"), list):
            starter_prompts = [str(p) for p in payload.get("prompts") if str(p).strip()]
    except Exception:
        starter_prompts = []

    starter_label_map_raw: dict[str, str] = {
        "Show changes in grassland extent in Montana.": "Grassland change",
        "Analyse forest loss trends in the Rusizi Basin, Rwanda, over the last 10 years.": "Forest loss trends",
        "How much of the Democratic Republic of Congo is natural land?": "DRC natural land",
        "Show forest loss from wildfires in California over the last five years.": "Wildfire forest loss",
        "How much natural land was converted to cropland in Spain last summer?": "Cropland conversion",
        "Where in Indonesia have restoration efforts occurred over the past decade?": "Restoration areas",
        "What are the trends in natural grassland area in Bolivia since 2015?": "Grassland trends",
        "Since 2015 in the US, how much natural land has been converted to other uses?": "Land conversion",
        "Compare disturbances in France last month: natural events vs human activity.": "Disturbances (FR)",
        "What were the top three causes of tree loss in Brazil last year?": "Tree loss causes",
        "Show the biggest changes in land cover in Kenya between 2015 and 2024.": "Land cover change",
    }

    starter_label_map = {
        _norm_prompt(k): v for k, v in starter_label_map_raw.items()
    }

    if starter_prompts and "prompt" in df.columns and df["prompt"].notna().any():
        starter_set = {_norm_prompt(p) for p in starter_prompts}
        prompt_norm = df["prompt"].map(_norm_prompt)
        starter_label = prompt_norm.map(lambda p: starter_label_map.get(p) if p in starter_set else "Other")

        starter_count = int((starter_label != "Other").sum())
        other_count = int((starter_label == "Other").sum())

        starter_only = starter_label[starter_label != "Other"]

        st.markdown("### Starter prompt mix", help="Starter prompts are pre-defined suggestions shown to users. Track which ones drive engagement vs. custom queries.")

        left, right = st.columns(2)

        with left:
            st.markdown(
                "#### Starter vs other prompts",
                help="Share of prompts that match your pre-defined starter prompt library.",
            )
            starter_vs_other = pd.DataFrame(
                [
                    {"label": "Starter", "count": starter_count},
                    {"label": "Other", "count": other_count},
                ]
            )
            starter_vs_other["percent"] = (
                starter_vs_other["count"] / max(1, starter_vs_other["count"].sum()) * 100
            ).round(1)

            starter_vs_other_chart = (
                alt.Chart(starter_vs_other)
                .mark_arc(innerRadius=55)
                .encode(
                    theta=alt.Theta("count:Q", title="Count"),
                    color=alt.Color("label:N", title=""),
                    tooltip=[
                        alt.Tooltip("label:N", title="Type"),
                        alt.Tooltip("count:Q", title="Count", format=","),
                        alt.Tooltip("percent:Q", title="%", format=".1f"),
                    ],
                )
                .properties(height=260)
            )
            st.altair_chart(starter_vs_other_chart, width="stretch")

        with right:
            st.markdown(
                "#### Starter prompt breakdown",
                help="Which starter prompts are being used most.",
            )
            starter_breakdown = starter_only.value_counts().reset_index()
            starter_breakdown.columns = ["label", "count"]
            starter_breakdown["percent"] = (
                starter_breakdown["count"] / max(1, starter_breakdown["count"].sum()) * 100
            ).round(1)

            starter_breakdown_chart = (
                alt.Chart(starter_breakdown)
                .mark_arc(innerRadius=55)
                .encode(
                    theta=alt.Theta("count:Q", title="Count"),
                    color=alt.Color("label:N", title=""),
                    tooltip=[
                        alt.Tooltip("label:N", title="Starter prompt"),
                        alt.Tooltip("count:Q", title="Count", format=","),
                        alt.Tooltip("percent:Q", title="%", format=".1f"),
                    ],
                )
                .properties(height=260)
            )
            st.altair_chart(starter_breakdown_chart, width="stretch")

    st.markdown("### Distributions & Breakdowns", help="Understand the shape of your data. Histograms reveal outliers, skew, and typical values better than averages alone.")

    if ("prompt_len_chars" in df.columns or "prompt_len_words" in df.columns) and len(df):
        prompt_len_chars_s = df["prompt_len_chars"].dropna() if "prompt_len_chars" in df.columns else pd.Series(dtype="float")
        prompt_len_words_s = df["prompt_len_words"].dropna() if "prompt_len_words" in df.columns else pd.Series(dtype="float")
        prompt_len_chars_s = prompt_len_chars_s[prompt_len_chars_s > 0]
        prompt_len_words_s = prompt_len_words_s[prompt_len_words_s > 0]

        st.markdown("#### Prompt length")
        plc, plw = st.columns(2)

        with plc:
            st.markdown(
                "##### Characters",
                help="Prompt length in characters. Helps spot unusually long prompts that may increase cost/latency.",
            )
            if len(prompt_len_chars_s):
                c1, c2, c3, c4, c5, c6 = st.columns(6)
                with c1:
                    st.metric("N", str(int(prompt_len_chars_s.count())))
                with c2:
                    st.metric("Mean", f"{float(prompt_len_chars_s.mean()):.0f}")
                with c3:
                    st.metric("Median", f"{float(prompt_len_chars_s.median()):.0f}")
                with c4:
                    st.metric("P90", f"{float(prompt_len_chars_s.quantile(0.9)):.0f}")
                with c5:
                    st.metric("P95", f"{float(prompt_len_chars_s.quantile(0.95)):.0f}")
                with c6:
                    st.metric("Max", f"{float(prompt_len_chars_s.max()):.0f}")

                prompt_len_chars_hist = (
                    alt.Chart(pd.DataFrame({"prompt_len_chars": prompt_len_chars_s}))
                    .transform_bin(
                        as_=["bin_start", "bin_end"],
                        field="prompt_len_chars",
                        bin=alt.Bin(maxbins=60),
                    )
                    .transform_calculate(bin_width="datum.bin_end - datum.bin_start")
                    .mark_bar()
                    .encode(
                        x=alt.X("bin_start:Q", title="Prompt length (characters)", bin=alt.Bin(binned=True)),
                        x2=alt.X2("bin_end:Q"),
                        y=alt.Y("count()", title="Count"),
                        tooltip=[
                            alt.Tooltip("bin_start:Q", title="Bin start", format=","),
                            alt.Tooltip("bin_end:Q", title="Bin end", format=","),
                            alt.Tooltip("bin_width:Q", title="Bin width", format=","),
                            alt.Tooltip("count()", title="Count", format=","),
                        ],
                    )
                    .properties(height=180)
                )
                st.altair_chart(prompt_len_chars_hist, width="stretch")
            else:
                st.info("No non-empty prompts available to chart.")

        with plw:
            st.markdown(
                "##### Words",
                help="Prompt length in words. Useful for understanding typical query complexity.",
            )
            if len(prompt_len_words_s):
                c1, c2, c3, c4, c5, c6 = st.columns(6)
                with c1:
                    st.metric("N", str(int(prompt_len_words_s.count())))
                with c2:
                    st.metric("Mean", f"{float(prompt_len_words_s.mean()):.0f}")
                with c3:
                    st.metric("Median", f"{float(prompt_len_words_s.median()):.0f}")
                with c4:
                    st.metric("P90", f"{float(prompt_len_words_s.quantile(0.9)):.0f}")
                with c5:
                    st.metric("P95", f"{float(prompt_len_words_s.quantile(0.95)):.0f}")
                with c6:
                    st.metric("Max", f"{float(prompt_len_words_s.max()):.0f}")

                prompt_len_words_hist = (
                    alt.Chart(pd.DataFrame({"prompt_len_words": prompt_len_words_s}))
                    .transform_bin(
                        as_=["bin_start", "bin_end"],
                        field="prompt_len_words",
                        bin=alt.Bin(maxbins=60),
                    )
                    .transform_calculate(bin_width="datum.bin_end - datum.bin_start")
                    .mark_bar()
                    .encode(
                        x=alt.X("bin_start:Q", title="Prompt length (words)", bin=alt.Bin(binned=True)),
                        x2=alt.X2("bin_end:Q"),
                        y=alt.Y("count()", title="Count"),
                        tooltip=[
                            alt.Tooltip("bin_start:Q", title="Bin start", format=","),
                            alt.Tooltip("bin_end:Q", title="Bin end", format=","),
                            alt.Tooltip("bin_width:Q", title="Bin width", format=","),
                            alt.Tooltip("count()", title="Count", format=","),
                        ],
                    )
                    .properties(height=180)
                )
                st.altair_chart(prompt_len_words_hist, width="stretch")
            else:
                st.info("No non-empty prompts available to chart.")

    outcome_chart = outcome_pie_chart(df)
    lang_chart = language_bar_chart(df)
    lat_chart_dist = latency_histogram(lat_s)
    cost_chart_dist = cost_histogram(cost_s)

    dist_c1, dist_c2 = st.columns(2)
    with dist_c1:
        st.markdown(
            "#### Outcome breakdown",
            help=(
                "Overall outcome mix across the selected period. Outcome rules: ANSWER = non-empty answer + tool usage; "
                "DEFER = non-empty/non-error answer but no tool usage; SOFT_ERROR = answer text looks like an error via heuristics; "
                "ERROR = empty answer."
            ),
        )
        st.altair_chart(outcome_chart, width="stretch")
    with dist_c2:
        if lang_chart:
            st.markdown("#### Top prompt languages", help="Detected languages for prompts (best-effort via langid).")
            st.altair_chart(lang_chart, width="stretch")

    st.markdown("#### Cost & Latency distributions", help="See how cost and latency are distributed across traces. p95 is a good measure of worst-case experience.")

    perf_c1, perf_c2 = st.columns(2)
    with perf_c1:
        st.markdown("##### Latency", help="Latency distribution and summary stats for the selected period.")
        if len(lat_s):
            lc1, lc2, lc3, lc4, lc5 = st.columns(5)
            with lc1:
                st.metric("Total traces", f"{int(lat_s.count()):,}")
            with lc2:
                st.metric("Mean", f"{float(lat_s.mean()):.2f}s")
            with lc3:
                st.metric("Median", f"{float(lat_s.median()):.2f}s")
            with lc4:
                st.metric("P95", f"{float(lat_s.quantile(0.95)):.2f}s")
            with lc5:
                st.metric("Max", f"{float(lat_s.max()):.2f}s")
            if lat_chart_dist:
                st.altair_chart(lat_chart_dist, width="stretch")
        else:
            st.info("No latency data available.")

    with perf_c2:
        st.markdown("##### Cost", help="Cost distribution and summary stats for the selected period.")
        if len(cost_s):
            total_cost = float(cost_s.sum())
            cc1, cc2, cc3, cc4, cc5 = st.columns(5)
            with cc1:
                st.metric("Total", f"${total_cost:.2f}")
            with cc2:
                st.metric("Mean", f"${float(cost_s.mean()):.4f}")
            with cc3:
                st.metric("Median", f"${float(cost_s.median()):.4f}")
            with cc4:
                st.metric("P95", f"${float(cost_s.quantile(0.95)):.4f}")
            with cc5:
                st.metric("Max", f"${float(cost_s.max()):.4f}")
            if cost_chart_dist:
                st.altair_chart(cost_chart_dist, width="stretch")
        else:
            st.info("No cost data available.")

    st.markdown("### GNW analysis usage", help="Product-specific metrics: which datasets, AOIs, and analysis types users are requesting.")

    pie_c1, pie_c2 = st.columns(2)
    with pie_c1:
        if "datasets_analysed" in df.columns:
            st.markdown("#### Datasets analysed", help="Datasets referenced in successful analyses (from trace context).")
            chart = category_pie_chart(df["datasets_analysed"], "dataset", "Datasets analysed", explode_csv=True)
            if chart:
                st.altair_chart(chart, width="stretch")
    with pie_c2:
        aoi_type_domain: list[str] = []
        if "aoi_type" in df.columns:
            aoi_type_counts = (
                df["aoi_type"]
                .fillna("")
                .astype(str)
                .str.strip()
                .replace({"": None})
                .dropna()
                .value_counts()
                .head(10)
            )
            if len(aoi_type_counts):
                aoi_type_domain = [str(x) for x in aoi_type_counts.index.tolist()]
                aoi_type_df = aoi_type_counts.rename_axis("aoi_type").reset_index(name="count")
                aoi_type_df["percent"] = (
                    aoi_type_df["count"] / max(1, int(aoi_type_df["count"].sum())) * 100
                ).round(1)
                aoi_type_scale = alt.Scale(domain=aoi_type_domain, scheme="tableau10")
                st.markdown("#### AOI type", help="What kinds of areas users are analysing (e.g., country, admin region, drawn polygon).")
                chart = (
                    alt.Chart(aoi_type_df)
                    .mark_arc(innerRadius=50)
                    .encode(
                        theta=alt.Theta("count:Q"),
                        color=alt.Color("aoi_type:N", title="AOI type", scale=aoi_type_scale),
                        tooltip=[
                            alt.Tooltip("aoi_type:N", title="AOI type"),
                            alt.Tooltip("count:Q", title="Count"),
                            alt.Tooltip("percent:Q", title="%", format=".1f"),
                        ],
                    )
                    .properties(title="AOI type", height=250)
                )
                st.altair_chart(chart, width="stretch")

    if "aoi_name" in df.columns:
        aoi_rows = df[["aoi_name", "aoi_type"]].copy() if "aoi_type" in df.columns else df[["aoi_name"]].copy()
        aoi_rows["aoi_name"] = (
            aoi_rows["aoi_name"].fillna("").astype(str).str.strip().replace({"": None})
        )
        if "aoi_type" in aoi_rows.columns:
            aoi_rows["aoi_type"] = (
                aoi_rows["aoi_type"].fillna("").astype(str).str.strip().replace({"": None})
            )

        aoi_rows = aoi_rows.dropna(subset=["aoi_name"])
        if len(aoi_rows):
            aoi_counts = aoi_rows["aoi_name"].value_counts().head(30)
            aoi_name_df = aoi_counts.rename_axis("aoi_name").reset_index(name="count")
            if "aoi_type" in aoi_rows.columns:
                # Choose most common type per AOI name so the bar can be colored.
                aoi_types = (
                    aoi_rows.dropna(subset=["aoi_type"])
                    .groupby("aoi_name")["aoi_type"]
                    .agg(lambda s: s.value_counts().index[0] if len(s.value_counts()) else None)
                    .reset_index()
                )
                aoi_name_df = aoi_name_df.merge(aoi_types, on="aoi_name", how="left")

            enc_color = None
            if "aoi_type" in aoi_name_df.columns and aoi_type_domain:
                enc_color = alt.Color(
                    "aoi_type:N",
                    title="AOI type",
                    scale=alt.Scale(domain=aoi_type_domain, scheme="tableau10"),
                )
            elif "aoi_type" in aoi_name_df.columns:
                enc_color = alt.Color("aoi_type:N", title="AOI type")

            chart = (
                alt.Chart(aoi_name_df)
                .mark_bar()
                .encode(
                    x=alt.X("count:Q", title="Count"),
                    y=alt.Y("aoi_name:N", sort="-x", title="AOI"),
                    color=enc_color,
                    tooltip=[
                        alt.Tooltip("aoi_name:N", title="AOI"),
                        alt.Tooltip("aoi_type:N", title="AOI type"),
                        alt.Tooltip("count:Q", title="Count", format=","),
                    ],
                )
                .properties(title="AOI selection counts", height=420)
            )
            st.markdown("#### AOI selection counts", help="Most commonly analysed places (AOIs) in this dataset.")
            st.altair_chart(chart, width="stretch")

    # -------------------------------------------------------------------------
    # Agentic Flow Analysis Section
    # -------------------------------------------------------------------------
    st.markdown(
        "### Agentic Flow Analysis",
        help="Deep dive into tool usage patterns, agent loops, token consumption, and internal vs user-visible errors.",
    )

    # Aggregate tool call stats across all traces
    all_tool_calls: list[dict[str, Any]] = []
    traces_with_ambiguity = 0
    for n in normed:
        calls = extract_tool_calls_and_results(n)
        all_tool_calls.extend(calls)
        if any(c["has_ambiguity"] for c in calls):
            traces_with_ambiguity += 1

    # Tool flow visualization
    st.markdown(
        "#### Tool call flow",
        help="Visualize how tool calls flow from START through tools to END. Line thickness = count, color = outcome status.",
    )
    flow_df = tool_flow_sankey_data(normed, extract_tool_flow)
    if len(flow_df):
        total_flows = int(flow_df["count"].sum())
        status_counts = (
            flow_df.groupby("status")["count"]
            .sum()
            .reset_index()
            .rename(columns={"count": "transitions"})
        )
        status_counts["percent"] = status_counts["transitions"] / max(1, total_flows) * 100

        total_traces_with_tools = sum(1 for n in normed if extract_tool_calls_and_results(n))

        flow_pie = (
            alt.Chart(status_counts)
            .mark_arc(innerRadius=55)
            .encode(
                theta=alt.Theta("transitions:Q", title="Transitions"),
                color=alt.Color("status:N", title="Status"),
                tooltip=[
                    alt.Tooltip("status:N", title="Status"),
                    alt.Tooltip("transitions:Q", title="Transitions", format=","),
                    alt.Tooltip("percent:Q", title="%", format=".1f"),
                ],
            )
            .properties(height=220)
        )

        clarity_df = pd.DataFrame(
            [
                {"label": "Clarification needed", "count": int(traces_with_ambiguity)},
                {"label": "No clarification", "count": int(total_traces_with_tools - traces_with_ambiguity)},
            ]
        )
        clarity_df["percent"] = clarity_df["count"] / max(1, int(clarity_df["count"].sum())) * 100
        clarity_pie = (
            alt.Chart(clarity_df)
            .mark_arc(innerRadius=55)
            .encode(
                theta=alt.Theta("count:Q", title="Traces"),
                color=alt.Color("label:N", title=""),
                tooltip=[
                    alt.Tooltip("label:N", title=""),
                    alt.Tooltip("count:Q", title="Traces", format=","),
                    alt.Tooltip("percent:Q", title="%", format=".1f"),
                ],
            )
            .properties(height=220)
        )

        pie_left, pie_right = st.columns(2)
        with pie_left:
            st.markdown(
                "##### Tool call flow (status)",
                help="Breakdown of transition outcomes (success, ambiguity, semantic error, error) across all tool transitions.",
            )
            st.altair_chart(flow_pie, width="stretch")
        with pie_right:
            st.markdown(
                "##### Clarification loop rate",
                help="Share of traces that triggered a clarification request (e.g. ambiguous AOI).",
            )
            if total_traces_with_tools > 0:
                st.altair_chart(clarity_pie, width="stretch")
            else:
                st.info("No tool calls found to calculate clarification rate.")

        # Show flow as table for exact values
        flow_summary = (
            flow_df.groupby(["source", "target", "status"])["count"]
            .sum()
            .reset_index()
            .sort_values("count", ascending=False)
            .head(20)
        )
        with st.expander("Top 20 tool transitions", expanded=False):
            st.dataframe(flow_summary, hide_index=True, width="stretch")
    else:
        st.info("No tool calls found in traces.")

    # Tool success rate by tool name
    if all_tool_calls:
        st.markdown(
            "#### Tool success rate by tool",
            help="Stacked bar showing outcomes (success, ambiguity, semantic error, error) for each tool.",
        )
        tool_calls_df = pd.DataFrame(all_tool_calls)
        tool_stats = (
            tool_calls_df.groupby("tool_name")
            .agg(
                total=("tool_name", "count"),
                success=("status", lambda x: int((x == "success").sum()) - int(tool_calls_df.loc[x.index, "has_ambiguity"].sum()) - int(tool_calls_df.loc[x.index, "is_semantic_error"].sum())),
                ambiguity=("has_ambiguity", "sum"),
                semantic_error=("is_semantic_error", "sum"),
                error=("status", lambda x: int((x == "error").sum())),
            )
            .reset_index()
        )
        # Fix negative success counts
        tool_stats["success"] = tool_stats["success"].clip(lower=0)
        tool_stats = tool_stats.sort_values("total", ascending=False)

        chart = tool_success_rate_chart(tool_stats)
        if chart:
            st.altair_chart(chart, width="stretch")

    # Reasoning tokens histogram
    if "reasoning_ratio" in df.columns:
        reasoning_ratios = df["reasoning_ratio"].dropna()
        reasoning_ratios = reasoning_ratios[reasoning_ratios > 0]
        if len(reasoning_ratios):
            st.markdown(
                "#### Reasoning tokens distribution",
                help="How much of output tokens are spent on 'reasoning' (chain-of-thought). High ratios may indicate overthinking or complex queries.",
            )
            rc1, rc2, rc3, rc4 = st.columns(4)
            with rc1:
                st.metric("Traces with reasoning", f"{len(reasoning_ratios):,}")
            with rc2:
                st.metric("Mean ratio", f"{reasoning_ratios.mean():.1%}")
            with rc3:
                st.metric("Median ratio", f"{reasoning_ratios.median():.1%}")
            with rc4:
                st.metric("P90 ratio", f"{reasoning_ratios.quantile(0.9):.1%}")

            chart = reasoning_tokens_histogram(reasoning_ratios)
            if chart:
                st.altair_chart(chart, width="stretch")

    # Tool calls vs latency scatter
    if "tool_call_count" in df.columns and "latency_seconds" in df.columns:
        plot_df = df[["tool_call_count", "latency_seconds", "outcome"]].dropna()
        if len(plot_df) and plot_df["tool_call_count"].max() > 0:
            st.markdown(
                "#### Tool calls vs latency",
                help="Scatter plot showing relationship between number of tool calls and trace latency. Trend line shows correlation.",
            )
            tc1, tc2, tc3 = st.columns(3)
            with tc1:
                st.metric("Max tool calls", f"{int(plot_df['tool_call_count'].max()):,}")
            with tc2:
                st.metric("Avg tool calls", f"{plot_df['tool_call_count'].mean():.1f}")
            with tc3:
                corr = plot_df["tool_call_count"].corr(plot_df["latency_seconds"])
                st.metric("Correlation", f"{corr:.2f}" if pd.notna(corr) else "N/A")

            chart = tool_calls_vs_latency_chart(plot_df)
            if chart:
                st.altair_chart(chart, width="stretch")

    # Internal vs user-visible error rate (supplements outcome chart)
    if "has_internal_error" in df.columns and "outcome" in df.columns:
        st.markdown(
            "#### Internal vs user-visible errors",
            help="Compare internal tool/API failures with errors visible in the final answer. Internal errors may be masked by agent recovery.",
        )
        internal_error_count = int(df["has_internal_error"].sum())
        user_visible_error_count = int((df["outcome"].isin(["ERROR", "EMPTY", "SOFT_ERROR"])).sum())
        total_traces = len(df)

        # Traces with internal error that still succeeded (agent recovered)
        recovered = int(((df["has_internal_error"]) & (df["outcome"] == "ANSWER")).sum())

        hidden_errors = max(0, internal_error_count - user_visible_error_count)

        err_pie_df = pd.DataFrame(
            [
                {"label": "No internal error", "count": int(total_traces - internal_error_count)},
                {"label": "Internal error", "count": int(internal_error_count)},
            ]
        )
        err_pie_df["percent"] = err_pie_df["count"] / max(1, int(err_pie_df["count"].sum())) * 100
        err_pie = (
            alt.Chart(err_pie_df)
            .mark_arc(innerRadius=55)
            .encode(
                theta=alt.Theta("count:Q", title="Traces"),
                color=alt.Color("label:N", title=""),
                tooltip=[
                    alt.Tooltip("label:N", title=""),
                    alt.Tooltip("count:Q", title="Traces", format=","),
                    alt.Tooltip("percent:Q", title="%", format=".1f"),
                ],
            )
            .properties(height=220)
        )

        both_internal_and_user_visible = int(
            ((df["has_internal_error"]) & (df["outcome"].isin(["ERROR", "EMPTY", "SOFT_ERROR"]))).sum()
        )
        internal_only = int(internal_error_count - both_internal_and_user_visible)
        user_visible_only = int(user_visible_error_count - both_internal_and_user_visible)
        no_errors = int(
            (
                (~df["has_internal_error"]) & (~df["outcome"].isin(["ERROR", "EMPTY", "SOFT_ERROR"]))
            ).sum()
        )

        overlap_df = pd.DataFrame(
            [
                {"label": "No errors", "count": no_errors},
                {"label": "Internal only", "count": internal_only},
                {"label": "User-visible only", "count": user_visible_only},
                {"label": "Both", "count": both_internal_and_user_visible},
            ]
        )
        overlap_df = overlap_df[overlap_df["count"] > 0]
        overlap_df["percent"] = overlap_df["count"] / max(1, int(overlap_df["count"].sum())) * 100

        overlap_pie = (
            alt.Chart(overlap_df)
            .mark_arc(innerRadius=55)
            .encode(
                theta=alt.Theta("count:Q", title="Traces"),
                color=alt.Color("label:N", title=""),
                tooltip=[
                    alt.Tooltip("label:N", title=""),
                    alt.Tooltip("count:Q", title="Traces", format=","),
                    alt.Tooltip("percent:Q", title="%", format=".1f"),
                ],
            )
            .properties(height=220)
        )

        e_left, e_right = st.columns(2)
        with e_left:
            st.markdown("##### Internal errors", help="Share of traces with any internal tool/API error.")
            st.altair_chart(err_pie, width="stretch")
        with e_right:
            st.markdown(
                "##### Internal vs user-visible overlap",
                help="How internal tool/API errors overlap with user-visible failures (ERROR/SOFT_ERROR).",
            )
            st.altair_chart(overlap_pie, width="stretch")

        with st.expander("Error metrics", expanded=False):
            ec1, ec2, ec3, ec4 = st.columns(4)
            with ec1:
                st.metric("Internal errors", f"{internal_error_count:,}", delta=f"{internal_error_count/max(1,total_traces)*100:.1f}%")
            with ec2:
                st.metric("User-visible errors", f"{user_visible_error_count:,}", delta=f"{user_visible_error_count/max(1,total_traces)*100:.1f}%")
            with ec3:
                st.metric("Agent recovered", f"{recovered:,}")
            with ec4:
                st.metric("Hidden errors", f"{hidden_errors:,}")
