"""Chart utilities for the Streamlit app."""

import altair as alt
import pandas as pd


def daily_volume_chart(daily_metrics: pd.DataFrame) -> alt.Chart:
    """Create daily volume chart (traces, users, threads)."""
    vol_long = daily_metrics.melt(
        id_vars=["date"],
        value_vars=["traces", "unique_users", "unique_threads"],
        var_name="metric",
        value_name="value",
    )
    vol_long["metric"] = vol_long["metric"].replace({
        "traces": "Traces",
        "unique_users": "Unique users",
        "unique_threads": "Unique threads",
    })
    return (
        alt.Chart(vol_long)
        .mark_line(point=True)
        .encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("value:Q", title="Count"),
            color=alt.Color("metric:N", title="Metric"),
            tooltip=[
                alt.Tooltip("date:T", title="Date"),
                alt.Tooltip("metric:N", title="Metric"),
                alt.Tooltip("value:Q", title="Count", format=","),
            ],
        )
        .properties(title="Daily volume (traces, users, threads)")
    )


def daily_outcome_chart(daily_metrics: pd.DataFrame) -> alt.Chart:
    """Create daily outcome rates stacked area chart."""
    outcome_long = daily_metrics.melt(
        id_vars=["date"],
        value_vars=["success_rate", "defer_rate", "soft_error_rate", "error_rate"],
        var_name="metric",
        value_name="value",
    )
    outcome_long["metric"] = outcome_long["metric"].replace({
        "success_rate": "Success",
        "defer_rate": "Defer",
        "soft_error_rate": "Soft error",
        "error_rate": "Error",
    })
    return (
        alt.Chart(outcome_long)
        .mark_area()
        .encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("value:Q", title="Rate", stack="normalize", axis=alt.Axis(format="%")),
            color=alt.Color("metric:N", title="Outcome", sort=["Success", "Defer", "Soft error", "Error"]),
            order=alt.Order("metric:N", sort="descending"),
            tooltip=[
                alt.Tooltip("date:T", title="Date"),
                alt.Tooltip("metric:N", title="Outcome"),
                alt.Tooltip("value:Q", title="Rate", format=".1%"),
            ],
        )
        .properties(title="Daily outcome rates")
    )


def daily_cost_chart(daily_metrics: pd.DataFrame) -> alt.Chart:
    """Create daily cost chart."""
    cost_long = daily_metrics.melt(
        id_vars=["date"],
        value_vars=["mean_cost", "p95_cost"],
        var_name="metric",
        value_name="value",
    )
    cost_long["metric"] = cost_long["metric"].replace({"mean_cost": "Mean cost", "p95_cost": "p95 cost"})
    return (
        alt.Chart(cost_long)
        .mark_line(point=True)
        .encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("value:Q", title="USD"),
            color=alt.Color("metric:N", title="Cost"),
            tooltip=[
                alt.Tooltip("date:T", title="Date"),
                alt.Tooltip("metric:N", title="Cost type"),
                alt.Tooltip("value:Q", title="USD", format="$.4f"),
            ],
        )
        .properties(title="Daily cost")
    )


def daily_latency_chart(daily_metrics: pd.DataFrame) -> alt.Chart:
    """Create daily latency chart."""
    lat_long = daily_metrics.melt(
        id_vars=["date"],
        value_vars=["mean_latency", "p95_latency"],
        var_name="metric",
        value_name="value",
    )
    lat_long["metric"] = lat_long["metric"].replace({"mean_latency": "Mean latency", "p95_latency": "p95 latency"})
    return (
        alt.Chart(lat_long)
        .mark_line(point=True)
        .encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("value:Q", title="Seconds"),
            color=alt.Color("metric:N", title="Latency"),
            tooltip=[
                alt.Tooltip("date:T", title="Date"),
                alt.Tooltip("metric:N", title="Latency type"),
                alt.Tooltip("value:Q", title="Seconds", format=".2f"),
            ],
        )
        .properties(title="Daily latency")
    )


def outcome_pie_chart(df: pd.DataFrame) -> alt.Chart:
    """Create outcome breakdown pie chart."""
    outcome_counts = df["outcome"].value_counts().reset_index()
    outcome_counts.columns = ["outcome", "count"]
    outcome_counts["percent"] = (outcome_counts["count"] / outcome_counts["count"].sum() * 100).round(1)
    return (
        alt.Chart(outcome_counts)
        .mark_arc(innerRadius=50)
        .encode(
            theta=alt.Theta("count:Q", title="Count"),
            color=alt.Color("outcome:N", title="Outcome"),
            tooltip=[
                alt.Tooltip("outcome:N", title="Outcome"),
                alt.Tooltip("count:Q", title="Count", format=","),
                alt.Tooltip("percent:Q", title="%", format=".1f"),
            ],
        )
        .properties(title="Outcome breakdown", height=250)
    )


def language_bar_chart(df: pd.DataFrame, top_n: int = 15) -> alt.Chart | None:
    """Create top languages bar chart."""
    if "lang_query" not in df.columns:
        return None
    lang_counts = (
        df["lang_query"]
        .fillna("")
        .astype(str)
        .str.strip()
        .replace({"": None})
        .dropna()
        .value_counts()
        .head(top_n)
    )
    if not len(lang_counts):
        return None
    lang_df = lang_counts.rename_axis("language").reset_index(name="count")
    lang_df["percent"] = (lang_df["count"] / lang_df["count"].sum() * 100).round(1)
    return (
        alt.Chart(lang_df)
        .mark_bar()
        .encode(
            x=alt.X("count:Q", title="Count"),
            y=alt.Y("language:N", sort="-x", title="Language"),
            tooltip=[
                alt.Tooltip("language:N", title="Language"),
                alt.Tooltip("count:Q", title="Count"),
                alt.Tooltip("percent:Q", title="%", format=".1f"),
            ],
        )
        .properties(title="Top prompt languages (langid)")
    )


def latency_histogram(latency_series: pd.Series) -> alt.Chart | None:
    """Create latency distribution histogram."""
    if not len(latency_series):
        return None
    lat_df = pd.DataFrame({"latency_seconds": latency_series})
    return (
        alt.Chart(lat_df)
        .mark_bar()
        .encode(
            x=alt.X("latency_seconds:Q", bin=alt.Bin(maxbins=30), title="Latency (s)"),
            y=alt.Y("count():Q", title="Traces"),
            tooltip=[
                alt.Tooltip("latency_seconds:Q", title="Latency (s)", format=".2f", bin=True),
                alt.Tooltip("count():Q", title="Traces"),
            ],
        )
        .properties(title="Latency distribution")
    )


def cost_histogram(cost_series: pd.Series) -> alt.Chart | None:
    """Create cost distribution histogram."""
    if not len(cost_series):
        return None
    cost_df = pd.DataFrame({"total_cost": cost_series})
    return (
        alt.Chart(cost_df)
        .mark_bar()
        .encode(
            x=alt.X("total_cost:Q", bin=alt.Bin(maxbins=30), title="Total cost (USD)"),
            y=alt.Y("count():Q", title="Traces"),
            tooltip=[
                alt.Tooltip("total_cost:Q", title="Cost (USD)", format="$.4f", bin=True),
                alt.Tooltip("count():Q", title="Traces"),
            ],
        )
        .properties(title="Cost distribution")
    )


def category_pie_chart(
    series: pd.Series,
    label: str,
    title: str,
    top_n: int = 10,
    explode_csv: bool = False,
) -> alt.Chart | None:
    """Create a pie chart for a categorical series."""
    if explode_csv:
        counts = (
            series.fillna("")
            .astype(str)
            .str.split(",")
            .explode()
            .astype(str)
            .str.strip()
            .replace({"": None})
            .dropna()
            .value_counts()
            .head(top_n)
        )
    else:
        counts = (
            series.fillna("")
            .astype(str)
            .str.strip()
            .replace({"": None})
            .dropna()
            .value_counts()
            .head(top_n)
        )
    if not len(counts):
        return None
    df = counts.rename_axis(label).reset_index(name="count")
    df["percent"] = (df["count"] / df["count"].sum() * 100).round(1)
    return (
        alt.Chart(df)
        .mark_arc(innerRadius=50)
        .encode(
            theta=alt.Theta("count:Q"),
            color=alt.Color(f"{label}:N", title=title),
            tooltip=[
                alt.Tooltip(f"{label}:N", title=title),
                alt.Tooltip("count:Q", title="Count"),
                alt.Tooltip("percent:Q", title="%", format=".1f"),
            ],
        )
        .properties(title=title, height=250)
    )


def success_rate_bar_chart(
    metric_df: pd.DataFrame,
    group_col: str,
    title: str,
) -> alt.Chart | None:
    """Create success rate bar chart from a metric table."""
    if not len(metric_df):
        return None
    return (
        alt.Chart(metric_df)
        .mark_bar()
        .encode(
            x=alt.X("success_rate:Q", title="Success rate", axis=alt.Axis(format="%")),
            y=alt.Y(f"{group_col}:N", sort="-x", title=title),
            tooltip=[
                alt.Tooltip(f"{group_col}:N", title=title),
                alt.Tooltip("traces:Q", title="Traces", format=","),
                alt.Tooltip("success_rate:Q", title="Success rate", format=".1%"),
                alt.Tooltip("defer_rate:Q", title="Defer rate", format=".1%"),
                alt.Tooltip("error_rate:Q", title="Error rate", format=".1%"),
            ],
        )
        .properties(title=f"Success rate by {title.lower()}")
    )
