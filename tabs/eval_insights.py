"""Eval Insights dashboard tab."""

from typing import Any

import pandas as pd
import streamlit as st
import altair as alt

from utils import (
    get_langfuse_headers,
    list_annotation_queues,
    list_annotation_queue_items,
    get_annotation_queue,
    fetch_scores_by_queue,
    delete_score,
    init_session_state,
)


def render(
    public_key: str,
    secret_key: str,
    base_url: str,
) -> None:
    """Render the Eval Insights dashboard."""
    st.subheader("📈 Eval Insights")
    st.caption(
        "Select an evaluation queue to view scoring insights, completion rates, and performance metrics."
    )

    init_session_state({
        "eval_insights_queue_id": "",
        "eval_insights_scores": [],
        "eval_insights_meta": {},
    })

    has_langfuse = bool(public_key and secret_key and base_url)
    if not has_langfuse:
        st.info("Add Langfuse credentials in the sidebar to view eval insights.")
        return

    headers = get_langfuse_headers(public_key, secret_key)

    # Fetch available queues
    try:
        queues = list_annotation_queues(base_url=base_url, headers=headers)
    except Exception as e:
        st.error(f"Failed to fetch annotation queues: {e}")
        return

    if not queues:
        st.info("No annotation queues found. Create one in the Human Eval page first.")
        return

    # Queue selection dropdown
    queue_options: dict[str, str] = {"": "Select an eval queue..."}
    for q in queues:
        if not isinstance(q, dict):
            continue
        qid = str(q.get("id") or "").strip()
        qname = str(q.get("name") or "").strip()
        if qid and qname:
            queue_options[qid] = qname

    queue_ids = list(queue_options.keys())
    selected_queue_id = st.selectbox(
        "Eval Queue",
        options=queue_ids,
        format_func=lambda qid: queue_options.get(qid, qid),
        index=queue_ids.index(st.session_state.eval_insights_queue_id)
        if st.session_state.eval_insights_queue_id in queue_ids
        else 0,
        key="eval_insights_queue_select",
    )

    if not selected_queue_id:
        st.info("Select a queue above to view insights.")
        return

    st.session_state.eval_insights_queue_id = selected_queue_id

    # Fetch queue details for rubric
    try:
        queue_details = get_annotation_queue(
            base_url=base_url,
            headers=headers,
            queue_id=selected_queue_id,
        )
    except Exception as e:
        st.warning(f"Could not fetch queue details: {e}")
        queue_details = {}

    queue_name = str(queue_details.get("name") or queue_options.get(selected_queue_id, ""))
    queue_description = str(queue_details.get("description") or "").strip()

    # Display rubric in expander
    with st.expander("📋 Eval Queue Rubric", expanded=False):
        if queue_description:
            st.markdown(queue_description)
        else:
            st.caption("No rubric description available for this queue.")

    # Fetch scores for this queue
    with st.spinner("Fetching scores..."):
        try:
            scores, meta = fetch_scores_by_queue(
                base_url=base_url,
                headers=headers,
                queue_id=selected_queue_id,
            )
            st.session_state.eval_insights_scores = scores
            st.session_state.eval_insights_meta = meta
        except Exception as e:
            st.error(f"Failed to fetch scores: {e}")
            return

    if not scores:
        st.info("No scores found for this queue yet. Complete some evaluations first.")
        return

    # Build DataFrame from scores
    df = _build_scores_dataframe(scores)

    # Display scores table in closed expander
    with st.expander(f"📊 Raw Scores Data ({len(df)} scores)", expanded=False):
        display_cols = ["trace_id", "score_config", "value", "comment", "source", "evaluator", "timestamp"]
        available_cols = [c for c in display_cols if c in df.columns]
        st.dataframe(df[available_cols] if available_cols else df, hide_index=True, width="stretch")

    st.divider()
    st.subheader("📊 Insights Dashboard")

    # Fetch queue items to calculate completion
    try:
        queue_items = _fetch_all_queue_items(
            base_url=base_url,
            headers=headers,
            queue_id=selected_queue_id,
        )
    except Exception:
        queue_items = []

    # Calculate metrics
    metrics = _calculate_metrics(df, queue_items)

    # Top row
    col1, col2 = st.columns(2)

    with col1:
        _render_completion_section(metrics, queue_items)

    with col2:
        _render_score_distribution_chart(df)

    st.divider()

    # Full-width Outcome Rates section
    _render_outcome_rates_section(df, metrics)

    st.divider()

    # Additional insights
    _render_temporal_insights(df)

    st.divider()

    _render_flagged_for_removal_section(
        base_url=base_url,
        headers=headers,
        df=df,
    )


def _fetch_all_queue_items(
    *,
    base_url: str,
    headers: dict[str, str],
    queue_id: str,
    page_size: int = 100,
    max_pages: int = 50,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    page = 1
    while page <= int(max_pages):
        batch = list_annotation_queue_items(
            base_url=base_url,
            headers=headers,
            queue_id=queue_id,
            status=None,
            page=page,
            limit=int(page_size),
        )
        if not batch:
            break
        rows.extend(batch)
        if len(batch) < int(page_size):
            break
        page += 1
    return rows


def _build_scores_dataframe(scores: list[dict[str, Any]]) -> pd.DataFrame:
    """Build a pandas DataFrame from raw scores data."""
    rows = []
    for s in scores:
        if not isinstance(s, dict):
            continue
        rows.append({
            "score_id": s.get("id"),
            "trace_id": s.get("traceId"),
            "queue_id": s.get("metadata", {}).get("queue_id", None),
            "evaluator": s.get("metadata", {}).get("evaluator", ""),
            "source": s.get("metadata", {}).get("source", None),
            "flagged_for_removal": bool(s.get("metadata", {}).get("flagged_for_removal", False)),
            "score_config": s.get("name"),
            "value": s.get("stringValue"),
            "comment": s.get("comment"),
            "timestamp": s.get("timestamp"),
            "data_type": s.get("dataType"),
        })
    return pd.DataFrame(rows)


def _render_flagged_for_removal_section(
    *,
    base_url: str,
    headers: dict[str, str],
    df: pd.DataFrame,
) -> None:
    flagged_df = df[df.get("flagged_for_removal", False) == True] if "flagged_for_removal" in df.columns else df.iloc[0:0]

    st.markdown("### 🗑️ Flagged for removal")
    st.caption(
        "These are evaluations where the reviewer indicated the trace was not relevant to the eval question / rubric."
    )

    if flagged_df.empty:
        st.caption("No items have been flagged for removal.")
        return

    display_cols = ["score_id", "trace_id", "evaluator", "value", "comment", "timestamp"]
    available_cols = [c for c in display_cols if c in flagged_df.columns]
    if available_cols:
        with st.expander(f"View flagged items ({len(flagged_df)})", expanded=False):
            st.dataframe(flagged_df[available_cols], hide_index=True, width="stretch")

def _calculate_metrics(df: pd.DataFrame, queue_items: list[dict[str, Any]]) -> dict[str, Any]:
    """Calculate all metrics from scores DataFrame."""
    metrics: dict[str, Any] = {}

    # Queue completion
    total_items = len(queue_items)
    completed_items = len([it for it in queue_items if str(it.get("status") or "").upper() == "COMPLETED"])
    pending_items = total_items - completed_items
    metrics["total_items"] = total_items
    metrics["completed_items"] = completed_items
    metrics["pending_items"] = pending_items
    metrics["completion_rate"] = (completed_items / total_items * 100) if total_items > 0 else 0

    # Outcome rates (pass/fail/unsure)
    if "string_value" in df.columns:
        value_counts = df["string_value"].str.lower().value_counts()
    elif "value" in df.columns:
        value_counts = df["value"].astype(str).str.lower().value_counts()
    else:
        value_counts = pd.Series(dtype=int)

    total_scores = len(df)
    
    # Map common values to pass/fail/unsure
    pass_count = sum(value_counts.get(v, 0) for v in ["pass", "passed", "1", "1.0", "true", "yes"])
    fail_count = sum(value_counts.get(v, 0) for v in ["fail", "failed", "0", "0.0", "false", "no"])
    unsure_count = sum(value_counts.get(v, 0) for v in ["unsure", "uncertain", "maybe", "0.5"])
    
    # If no matches found, try numeric interpretation
    if pass_count + fail_count + unsure_count == 0 and "value" in df.columns:
        numeric_vals = pd.to_numeric(df["value"], errors="coerce")
        pass_count = int((numeric_vals == 1).sum())
        fail_count = int((numeric_vals == 0).sum())
        unsure_count = int((numeric_vals == 0.5).sum())

    metrics["pass_count"] = pass_count
    metrics["fail_count"] = fail_count
    metrics["unsure_count"] = unsure_count
    metrics["pass_rate"] = (pass_count / total_scores * 100) if total_scores > 0 else 0
    metrics["fail_rate"] = (fail_count / total_scores * 100) if total_scores > 0 else 0
    metrics["unsure_rate"] = (unsure_count / total_scores * 100) if total_scores > 0 else 0

    # Accuracy (exclude "unsure" from calculations)
    binary_total = pass_count + fail_count
    metrics["accuracy"] = (pass_count / binary_total * 100) if binary_total > 0 else 0

    return metrics


def _render_completion_section(metrics: dict[str, Any], queue_items: list[dict[str, Any]]) -> None:
    """Render queue completion metrics and chart."""
    st.markdown("### 🎯 Queue Completion")
    
    with st.expander("ℹ️ What is Queue Completion?", expanded=False):
        st.markdown("""
        **Queue Completion** shows how much of the evaluation queue has been processed.
        
        - **Completed**: Items that have been evaluated and scored
        - **Pending**: Items still awaiting evaluation
        - **Completion Rate**: Percentage of total items completed
        
        A higher completion rate means more of your evaluation sample has been reviewed.
        """)

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric("Completed", metrics["completed_items"])
    with col_b:
        st.metric("Pending", metrics["pending_items"])
    with col_c:
        st.metric("Completion Rate", f"{metrics['completion_rate']:.1f}%")

    # Progress bar
    st.progress(metrics["completion_rate"] / 100)

    # Pie chart for completion
    if metrics["total_items"] > 0:
        completion_data = pd.DataFrame({
            "Status": ["Completed", "Pending"],
            "Count": [metrics["completed_items"], metrics["pending_items"]]
        })
        chart = alt.Chart(completion_data).mark_arc(innerRadius=40).encode(
            theta=alt.Theta("Count:Q"),
            color=alt.Color("Status:N", scale=alt.Scale(
                domain=["Completed", "Pending"],
                range=["#22c55e", "#f59e0b"]
            )),
            tooltip=["Status", "Count"]
        ).properties(height=200)
        st.altair_chart(chart, width="stretch")


def _render_outcome_rates_section(df: pd.DataFrame, metrics: dict[str, Any]) -> None:
    """Render pass/fail/unsure outcome rates."""
    st.markdown("### 📊 Outcome Rates")

    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        st.metric("Accuracy", f"{metrics['accuracy']:.1f}%", delta_arrow="off")
    with col_b:
        st.metric(
            "Pass Rate",
            f"{metrics['pass_rate']:.1f}%",
            delta=f"{metrics['pass_count']} scores",
            delta_arrow="off",
        )
    with col_c:
        st.metric(
            "Fail Rate",
            f"{metrics['fail_rate']:.1f}%",
            delta=f"{metrics['fail_count']} scores",
            delta_arrow="off",
        )
    with col_d:
        st.metric(
            "Unsure Rate",
            f"{metrics['unsure_rate']:.1f}%",
            delta=f"{metrics['unsure_count']} scores",
            delta_arrow="off",
        )

    left, right = st.columns([1, 2])
    with left:
        st.markdown(
            """
            **Outcome Rates** show the distribution of evaluation results:

            - **Pass Rate**: % of responses that met the rubric
            - **Fail Rate**: % of responses that did not meet the rubric
            - **Unsure Rate**: % where the reviewer couldn't decide
            - **Accuracy**: % of evaluated responses marked **Pass**, excluding **Unsure** (`(Pass) / (Pass + Fail)`)
            """
        )

    with right:
        total = metrics["pass_count"] + metrics["fail_count"] + metrics["unsure_count"]
        if total <= 0:
            st.caption("No outcome data to chart.")
            return

        binary_total = metrics["pass_count"] + metrics["fail_count"]
        outcome_data = pd.DataFrame({
            "Outcome": ["Accuracy", "Pass", "Fail", "Unsure"],
            "Count": [binary_total, metrics["pass_count"], metrics["fail_count"], metrics["unsure_count"]],
            "Percentage": [metrics["accuracy"], metrics["pass_rate"], metrics["fail_rate"], metrics["unsure_rate"]],
        })

        base = alt.Chart(outcome_data).encode(
            y=alt.Y("Outcome:N", sort=["Accuracy", "Pass", "Fail", "Unsure"], title=None),
            x=alt.X(
                "Percentage:Q",
                scale=alt.Scale(domain=[0, 100]),
                axis=alt.Axis(title="%", format=".0f"),
            ),
            color=alt.Color(
                "Outcome:N",
                scale=alt.Scale(
                    domain=["Accuracy", "Pass", "Fail", "Unsure"],
                    range=["#3b82f6", "#22c55e", "#ef4444", "#f59e0b"],
                ),
                legend=None,
            ),
            tooltip=[
                "Outcome",
                alt.Tooltip("Percentage:Q", title="%", format=".1f"),
                alt.Tooltip("Count:Q", title="Count"),
            ],
        )

        bars = base.mark_bar()

        target_df = pd.DataFrame({"x": [80.0], "label": ["Target pass rate"]})
        target_line = alt.Chart(target_df).mark_rule(
            color="#22c55e",
            strokeDash=[6, 4],
            strokeWidth=2,
        ).encode(x="x:Q")

        target_label = alt.Chart(target_df).mark_text(
            align="left",
            baseline="top",
            dx=6,
            dy=6,
            color="#ffffff",
        ).encode(
            x="x:Q",
            y=alt.value(0),
            text="label:N",
        )

        chart = alt.layer(bars, target_line, target_label).properties(height=180)
        st.altair_chart(chart, width="stretch")


def _render_score_distribution_chart(df: pd.DataFrame) -> None:
    """Render score value distribution."""
    st.markdown("### 📈 Score Distribution")
    
    with st.expander("ℹ️ About Score Distribution", expanded=False):
        st.markdown("""
        Shows the distribution of all score values in the queue.
        
        This helps identify:
        - Whether evaluators are using the full range of scores
        - Any clustering around certain values
        - Potential evaluator biases
        """)

    if "string_value" in df.columns and df["string_value"].notna().any():
        value_col = "string_value"
    elif "value" in df.columns:
        value_col = "value"
    else:
        st.caption("No score values to display.")
        return

    dist_data = df[value_col].value_counts().reset_index()
    dist_data.columns = ["Value", "Count"]

    present_vals = [str(v) for v in dist_data["Value"].tolist()]
    known_order = ["Pass", "Fail", "Unsure"]
    known_set = {v.lower() for v in known_order}
    present_set = {v.lower() for v in present_vals}
    uses_known = bool(present_set) and present_set.issubset(known_set)
    if uses_known:
        domain = [v for v in known_order if v.lower() in present_set]
        color_scale = alt.Scale(domain=domain, range=["#22c55e", "#ef4444", "#f59e0b"][: len(domain)])
    else:
        color_scale = alt.Scale(range=["#22c55e", "#ef4444", "#f59e0b"])

    chart = (
        alt.Chart(dist_data)
        .mark_arc(innerRadius=45)
        .encode(
            theta=alt.Theta("Count:Q"),
            color=alt.Color("Value:N", scale=color_scale),
            tooltip=["Value", "Count"],
        )
        .properties(height=220)
    )
    st.altair_chart(chart, width="stretch")


def _render_temporal_insights(df: pd.DataFrame) -> None:
    """Render time-based insights."""
    st.markdown("### 📅 Evaluation Timeline")
    
    with st.expander("ℹ️ About the Timeline", expanded=False):
        st.markdown("""
        Shows when evaluations were performed over time.
        
        Useful for:
        - Tracking evaluation progress
        - Identifying periods of high/low activity
        - Planning evaluation sprints
        """)

    if "timestamp" not in df.columns and "created_at" not in df.columns:
        st.caption("No timestamp data available.")
        return

    time_col = "timestamp" if "timestamp" in df.columns else "created_at"
    df_time = df.copy()
    df_time["datetime"] = pd.to_datetime(df_time[time_col], errors="coerce")
    df_time = df_time.dropna(subset=["datetime"])

    if df_time.empty:
        st.caption("No valid timestamps found.")
        return

    df_time["date"] = df_time["datetime"].dt.date
    daily_counts = df_time.groupby("date").size().reset_index(name="Count")
    daily_counts["date"] = pd.to_datetime(daily_counts["date"])

    chart = alt.Chart(daily_counts).mark_line(point=True).encode(
        x=alt.X("date:T", title="Date"),
        y=alt.Y("Count:Q", title="Evaluations"),
        tooltip=[alt.Tooltip("date:T", title="Date"), "Count"]
    ).properties(height=200)
    st.altair_chart(chart, width="stretch")

    # Summary stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("First Evaluation", str(df_time["datetime"].min().date()))
    with col2:
        st.metric("Last Evaluation", str(df_time["datetime"].max().date()))
    with col3:
        days_active = (df_time["datetime"].max() - df_time["datetime"].min()).days + 1
        avg_per_day = len(df_time) / max(1, days_active)
        st.metric("Avg per Day", f"{avg_per_day:.1f}")
