"""Chart utilities for the Streamlit app."""

import altair as alt
import pandas as pd


_LANG_NAME_MAP: dict[str, str] = {
    "af": "Afrikaans",
    "ar": "Arabic",
    "bg": "Bulgarian",
    "bn": "Bengali",
    "ca": "Catalan",
    "cs": "Czech",
    "da": "Danish",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "es": "Spanish",
    "et": "Estonian",
    "fa": "Persian",
    "fi": "Finnish",
    "fr": "French",
    "gu": "Gujarati",
    "he": "Hebrew",
    "hi": "Hindi",
    "hr": "Croatian",
    "hu": "Hungarian",
    "id": "Indonesian",
    "it": "Italian",
    "ja": "Japanese",
    "kn": "Kannada",
    "ko": "Korean",
    "lt": "Lithuanian",
    "lv": "Latvian",
    "mk": "Macedonian",
    "ml": "Malayalam",
    "mr": "Marathi",
    "ne": "Nepali",
    "nl": "Dutch",
    "no": "Norwegian",
    "pa": "Punjabi",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "sk": "Slovak",
    "sl": "Slovenian",
    "so": "Somali",
    "sq": "Albanian",
    "sv": "Swedish",
    "sw": "Swahili",
    "ta": "Tamil",
    "te": "Telugu",
    "th": "Thai",
    "tl": "Tagalog",
    "tr": "Turkish",
    "uk": "Ukrainian",
    "ur": "Urdu",
    "vi": "Vietnamese",
    "zh": "Chinese",
    "unknown": "Unknown",
}


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
    lang_df["language_name"] = (
        lang_df["language"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map(lambda c: _LANG_NAME_MAP.get(c, c or "Unknown"))
    )
    lang_df["percent"] = (lang_df["count"] / lang_df["count"].sum() * 100).round(1)
    return (
        alt.Chart(lang_df)
        .mark_bar()
        .encode(
            x=alt.X("count:Q", title="Count"),
            y=alt.Y("language:N", sort="-x", title="Language"),
            tooltip=[
                alt.Tooltip("language_name:N", title="Language"),
                alt.Tooltip("language:N", title="Code"),
                alt.Tooltip("count:Q", title="Count"),
                alt.Tooltip("percent:Q", title="%", format=".1f"),
            ],
        )
        .properties(title="Top prompt languages (langid)")
    )


def latency_histogram(latency_series: pd.Series) -> alt.Chart | None:
    """Create latency distribution histogram with proper bin tooltips."""
    if not len(latency_series):
        return None
    lat_df = pd.DataFrame({"latency_seconds": latency_series})
    return (
        alt.Chart(lat_df)
        .transform_bin(
            as_=["bin_start", "bin_end"],
            field="latency_seconds",
            bin=alt.Bin(maxbins=30),
        )
        .transform_aggregate(
            count="count()",
            groupby=["bin_start", "bin_end"],
        )
        .transform_calculate(
            bin_range="format(datum.bin_start, '.1f') + 's – ' + format(datum.bin_end, '.1f') + 's'"
        )
        .mark_bar()
        .encode(
            x=alt.X("bin_start:Q", title="Latency (s)", bin="binned"),
            x2=alt.X2("bin_end:Q"),
            y=alt.Y("count:Q", title="Traces"),
            tooltip=[
                alt.Tooltip("bin_range:N", title="Latency range"),
                alt.Tooltip("count:Q", title="Traces", format=","),
            ],
        )
        .properties(title="Latency distribution", height=220)
    )


def cost_histogram(cost_series: pd.Series) -> alt.Chart | None:
    """Create cost distribution histogram with proper bin tooltips."""
    if not len(cost_series):
        return None
    cost_df = pd.DataFrame({"total_cost": cost_series})
    return (
        alt.Chart(cost_df)
        .transform_bin(
            as_=["bin_start", "bin_end"],
            field="total_cost",
            bin=alt.Bin(maxbins=30),
        )
        .transform_aggregate(
            count="count()",
            groupby=["bin_start", "bin_end"],
        )
        .transform_calculate(
            bin_range="'$' + format(datum.bin_start, '.4f') + ' – $' + format(datum.bin_end, '.4f')"
        )
        .mark_bar()
        .encode(
            x=alt.X("bin_start:Q", title="Total cost (USD)", bin="binned"),
            x2=alt.X2("bin_end:Q"),
            y=alt.Y("count:Q", title="Traces"),
            tooltip=[
                alt.Tooltip("bin_range:N", title="Cost range"),
                alt.Tooltip("count:Q", title="Traces", format=","),
            ],
        )
        .properties(title="Cost distribution", height=220)
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


# ---------------------------------------------------------------------------
# Agentic / Tool Analysis Charts
# ---------------------------------------------------------------------------

def tool_success_rate_chart(tool_stats_df: pd.DataFrame) -> alt.Chart | None:
    """Create stacked bar chart of tool call outcomes by tool name.

    Expects df with columns: tool_name, success, ambiguity, semantic_error, error
    """
    if not len(tool_stats_df):
        return None

    # Melt to long form for stacking
    long_df = tool_stats_df.melt(
        id_vars=["tool_name", "total"],
        value_vars=["success", "ambiguity", "semantic_error", "error"],
        var_name="outcome",
        value_name="count",
    )
    long_df["outcome"] = long_df["outcome"].replace({
        "success": "Success",
        "ambiguity": "Ambiguity",
        "semantic_error": "Semantic Error",
        "error": "Error",
    })

    # Calculate percentage for tooltip
    long_df["pct"] = long_df["count"] / long_df["total"].clip(lower=1)

    color_scale = alt.Scale(
        domain=["Success", "Ambiguity", "Semantic Error", "Error"],
        range=["#4CAF50", "#FFC107", "#FF9800", "#F44336"],
    )

    return (
        alt.Chart(long_df)
        .mark_bar()
        .encode(
            x=alt.X("count:Q", title="Tool calls", stack="zero"),
            y=alt.Y("tool_name:N", sort="-x", title="Tool"),
            color=alt.Color("outcome:N", title="Outcome", scale=color_scale),
            tooltip=[
                alt.Tooltip("tool_name:N", title="Tool"),
                alt.Tooltip("outcome:N", title="Outcome"),
                alt.Tooltip("count:Q", title="Count", format=","),
                alt.Tooltip("pct:Q", title="% of tool calls", format=".1%"),
                alt.Tooltip("total:Q", title="Total calls", format=","),
            ],
        )
        .properties(title="Tool call outcomes by tool", height=max(260, len(tool_stats_df) * 38))
    )


def tool_calls_vs_latency_chart(df: pd.DataFrame) -> alt.Chart | None:
    """Create scatter plot of tool call count vs latency per trace.

    Expects df with columns: tool_call_count, latency_seconds, outcome (optional)
    """
    if not len(df) or "tool_call_count" not in df.columns or "latency_seconds" not in df.columns:
        return None

    plot_df = df[["tool_call_count", "latency_seconds"]].dropna().copy()
    if "outcome" in df.columns:
        plot_df["outcome"] = df.loc[plot_df.index, "outcome"]
    else:
        plot_df["outcome"] = "Unknown"

    if not len(plot_df):
        return None

    base = alt.Chart(plot_df)

    points = base.mark_circle(size=60, opacity=0.6).encode(
        x=alt.X("tool_call_count:Q", title="Tool calls per trace"),
        y=alt.Y("latency_seconds:Q", title="Latency (s)"),
        color=alt.Color("outcome:N", title="Outcome"),
        tooltip=[
            alt.Tooltip("tool_call_count:Q", title="Tool calls"),
            alt.Tooltip("latency_seconds:Q", title="Latency (s)", format=".2f"),
            alt.Tooltip("outcome:N", title="Outcome"),
        ],
    )

    # Add trend line
    trend = base.transform_regression(
        "tool_call_count", "latency_seconds"
    ).mark_line(color="gray", strokeDash=[4, 4], opacity=0.7)

    return (points + trend).properties(
        title="Tool calls vs latency",
        height=280,
    )


def reasoning_tokens_histogram(reasoning_ratios: pd.Series) -> alt.Chart | None:
    """Create histogram of reasoning token ratio (reasoning / output tokens).

    Expects a series of ratios between 0 and 1.
    """
    ratios = reasoning_ratios.dropna()
    ratios = ratios[ratios >= 0]
    if not len(ratios):
        return None

    ratio_df = pd.DataFrame({"reasoning_ratio": ratios})

    return (
        alt.Chart(ratio_df)
        .transform_bin(
            as_=["bin_start", "bin_end"],
            field="reasoning_ratio",
            bin=alt.Bin(maxbins=20, extent=[0, 1]),
        )
        .transform_aggregate(
            count="count()",
            groupby=["bin_start", "bin_end"],
        )
        .transform_calculate(
            bin_range="format(datum.bin_start * 100, '.0f') + '% – ' + format(datum.bin_end * 100, '.0f') + '%'"
        )
        .mark_bar()
        .encode(
            x=alt.X("bin_start:Q", title="Reasoning tokens (% of output)", bin="binned", axis=alt.Axis(format="%")),
            x2=alt.X2("bin_end:Q"),
            y=alt.Y("count:Q", title="Traces"),
            tooltip=[
                alt.Tooltip("bin_range:N", title="Reasoning ratio"),
                alt.Tooltip("count:Q", title="Traces", format=","),
            ],
        )
        .properties(title="Reasoning tokens share distribution", height=220)
    )


def tool_flow_sankey_data(traces: list, extract_flow_fn) -> pd.DataFrame:
    """Build edge list for tool flow visualization.

    Returns df with columns: source, target, count, status
    Where source/target are "START", tool names, or "END".
    """
    edges: dict[tuple[str, str, str], int] = {}

    for t in traces:
        flow = extract_flow_fn(t)
        if not flow:
            continue

        # START -> first tool
        first_tool, first_status = flow[0]
        key = ("START", first_tool, first_status)
        edges[key] = edges.get(key, 0) + 1

        # tool -> tool transitions
        for i in range(len(flow) - 1):
            curr_tool, _ = flow[i]
            next_tool, next_status = flow[i + 1]
            key = (curr_tool, next_tool, next_status)
            edges[key] = edges.get(key, 0) + 1

        # last tool -> END
        last_tool, last_status = flow[-1]
        key = (last_tool, "END", last_status)
        edges[key] = edges.get(key, 0) + 1

    if not edges:
        return pd.DataFrame()

    rows = [
        {"source": s, "target": t, "status": st, "count": c}
        for (s, t, st), c in edges.items()
    ]
    return pd.DataFrame(rows)


def tool_flow_arc_chart(flow_df: pd.DataFrame) -> alt.Chart | None:
    """Create arc diagram showing tool call flows with status coloring.

    This is an alternative to Sankey that works in Altair.
    """
    if not len(flow_df):
        return None

    # Get unique nodes and assign positions
    nodes = sorted(set(flow_df["source"].tolist() + flow_df["target"].tolist()))
    node_order = {"START": 0, "END": 999}
    for i, n in enumerate(nodes):
        if n not in node_order:
            node_order[n] = i + 1
    nodes = sorted(nodes, key=lambda x: node_order.get(x, 100))

    node_df = pd.DataFrame({
        "node": nodes,
        "x": list(range(len(nodes))),
    })

    # Add source/target x positions to flow_df
    flow_df = flow_df.copy()
    flow_df["source_x"] = flow_df["source"].map(lambda n: node_order.get(n, 100))
    flow_df["target_x"] = flow_df["target"].map(lambda n: node_order.get(n, 100))

    color_scale = alt.Scale(
        domain=["success", "ambiguity", "semantic_error", "error"],
        range=["#4CAF50", "#FFC107", "#FF9800", "#F44336"],
    )

    status_order = ["success", "ambiguity", "semantic_error", "error"]

    # Node labels
    node_chart = (
        alt.Chart(node_df)
        .mark_text(fontSize=11, fontWeight="bold")
        .encode(
            x=alt.X("x:Q", axis=None),
            text=alt.Text("node:N"),
        )
    )

    # Arcs as rule marks between nodes
    arc_chart = (
        alt.Chart(flow_df)
        .mark_rule(opacity=0.6)
        .encode(
            x=alt.X("source_x:Q", axis=None),
            x2=alt.X2("target_x:Q"),
            y=alt.Y("status:N", title=None, sort=status_order),
            y2=alt.Y2("status:N"),
            color=alt.Color("status:N", title="Status", scale=color_scale),
            strokeWidth=alt.StrokeWidth("count:Q", title="Count", scale=alt.Scale(range=[1, 8])),
            tooltip=[
                alt.Tooltip("source:N", title="From"),
                alt.Tooltip("target:N", title="To"),
                alt.Tooltip("status:N", title="Status"),
                alt.Tooltip("count:Q", title="Count", format=","),
            ],
        )
    )

    return (
        alt.layer(arc_chart, node_chart)
        .properties(title="Tool call flow", height=420, width="container")
        .configure_view(strokeWidth=0)
    )
