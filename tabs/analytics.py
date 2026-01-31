"""Trace Analytics Reports tab."""

import hashlib
import json
import time as time_mod
from pathlib import Path
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st

from utils.prompt_fixtures import (
    DEFAULT_ENRICH_BATCH_PROMPT_TEMPLATE,
    DEFAULT_ENRICH_PROMPT,
)
from utils import (
    normalize_trace_format,
    parse_trace_dt,
    first_human_prompt,
    final_ai_message,
    classify_outcome,
    extract_trace_context,
    as_float,
    safe_json_loads,
    strip_code_fences,
    get_gemini_model_options,
    csv_bytes_any,
    save_bytes_to_local_path,
    daily_volume_chart,
    daily_outcome_chart,
    daily_cost_chart,
    daily_latency_chart,
    outcome_pie_chart,
    language_bar_chart,
    latency_histogram,
    cost_histogram,
    category_pie_chart,
    success_rate_bar_chart,
)


def render(
    public_key: str,
    secret_key: str,
    base_url: str,
    base_thread_url: str,
    gemini_api_key: str,
    use_date_filter: bool,
    start_date,
    end_date,
    envs: list[str],
    stats_page_limit: int,
    stats_max_traces: int,
) -> None:
    """Render the Trace Analytics Reports tab."""
    st.subheader("Trace stats for selected date range")

    if "stats_traces" not in st.session_state:
        st.session_state.stats_traces = []

    if "stats_enrich_cache" not in st.session_state:
        st.session_state.stats_enrich_cache = {}

    traces: list[dict[str, Any]] = st.session_state.stats_traces
    if not traces:
        st.info(
            "This tab gives you a high-level view of the currently loaded trace dataset: volumes, outcomes, latency, cost, "
            "and optional Gemini enrichment.\n\n"
            "1. Use the sidebar **🚀 Fetch traces** button to load a dataset once.\n"
            "2. Switch between tabs to explore different views of the **same** traces without re-fetching."
        )
        return

    normed = [normalize_trace_format(t) for t in traces]

    rows: list[dict[str, Any]] = []
    for n in normed:
        prompt = first_human_prompt(n)
        answer = final_ai_message(n)
        dt = parse_trace_dt(n)
        outcome = classify_outcome(n, answer or "")

        ctx = extract_trace_context(n)

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

    enrich_model = "gemini-2.5-flash-lite"
    enrich_prompt = DEFAULT_ENRICH_PROMPT
    is_default_enrich_prompt = True

    def _enrich_cache_key(prompt_text: str) -> str:
        sig = f"{enrich_model}|{enrich_prompt}|{prompt_text}".encode("utf-8")
        return hashlib.sha256(sig).hexdigest()

    st.markdown("---")
    st.markdown("### 🔍 Gemini Enrichment")
    enrich_max_labels = 100
    batch_size = 100
    if not gemini_api_key:
        st.warning("Set GEMINI_API_KEY in the sidebar to run enrichment.")
    else:
        c_clear, c_run = st.columns([1, 3])
        with c_clear:
            if st.button("Clear enrichment cache", key="stats_enrich_clear"):
                st.session_state.stats_enrich_cache = {}
                st.success("Cleared enrichment cache")

        with st.expander("Enrichment settings", expanded=False):
            enrich_model_options = get_gemini_model_options(gemini_api_key)
            default_model = "gemini-2.5-flash-lite"
            if default_model not in enrich_model_options and len(enrich_model_options):
                default_model = enrich_model_options[0]
            default_idx = (
                enrich_model_options.index(default_model)
                if default_model in enrich_model_options
                else 0
            )
            enrich_model = st.selectbox(
                "Gemini model",
                options=enrich_model_options,
                index=default_idx,
                key="analytics_enrich_model",
            )
            enrich_prompt = st.text_area("Enrichment prompt", value=DEFAULT_ENRICH_PROMPT, height=120)
            is_default_enrich_prompt = enrich_prompt.strip() == DEFAULT_ENRICH_PROMPT.strip()
            enrich_max_labels = st.number_input("Max prompts to label", min_value=1, max_value=5000, value=100)
            batch_size = st.number_input("Batch size (prompts per API call)", min_value=1, max_value=200, value=100)
        with c_run:
            run_clicked = st.button("🚀 Run enrichment", key="stats_enrich", type="primary")

        if run_clicked:
            import google.generativeai as genai

            genai.configure(api_key=gemini_api_key)
            model = genai.GenerativeModel(enrich_model)

            prompts = (
                df[["trace_id", "prompt"]]
                .dropna(subset=["prompt"])
                .astype({"prompt": "string"})
                .to_dict("records")
            )
            prompts = [p for p in prompts if str(p.get("prompt") or "").strip()]

            missing: list[dict[str, Any]] = []
            for p in prompts:
                key = _enrich_cache_key(str(p.get("prompt") or ""))
                cached = st.session_state.stats_enrich_cache.get(key)
                has_error = isinstance(cached, dict) and bool(cached.get("error"))
                if key not in st.session_state.stats_enrich_cache or has_error:
                    missing.append({"key": key, **p})

            missing = missing[: int(enrich_max_labels)]
            if not missing:
                st.info("No new prompts to label (everything already cached).")
            else:
                prog = st.progress(0)
                status = st.empty()

                batch_prompt_template = DEFAULT_ENRICH_BATCH_PROMPT_TEMPLATE

                def process_batch(batch_items: list[dict]) -> None:
                    batch_prompts = [str(item.get("prompt") or "") for item in batch_items]
                    prompts_json = json.dumps([{"id": i, "prompt": p} for i, p in enumerate(batch_prompts)], ensure_ascii=False)
                    batch_prompt_text = batch_prompt_template.format(prompts_json=prompts_json)

                    try:
                        resp = model.generate_content(batch_prompt_text)
                        txt = str(getattr(resp, "text", "") or "")
                        cleaned = strip_code_fences(txt)
                        payload: Any
                        try:
                            payload = json.loads(cleaned)
                        except Exception:
                            # Fallback: try to parse the first JSON array/object substring found.
                            payload = None
                            try:
                                if "[" in cleaned and "]" in cleaned:
                                    start = cleaned.find("[")
                                    end = cleaned.rfind("]")
                                    payload = json.loads(cleaned[start : end + 1])
                                elif "{" in cleaned and "}" in cleaned:
                                    start = cleaned.find("{")
                                    end = cleaned.rfind("}")
                                    payload = json.loads(cleaned[start : end + 1])
                            except Exception:
                                payload = None

                        if payload is None:
                            # Last resort: keep prior behavior, which wraps non-dict JSON under {"raw": ...}
                            payload = safe_json_loads(cleaned)

                        def _extract_result_list(payload: Any) -> list[Any] | None:
                            if isinstance(payload, list):
                                return payload
                            if isinstance(payload, dict):
                                # Handle safe_json_loads wrapping and common API wrappers.
                                if "raw" in payload:
                                    raw_val = payload.get("raw")
                                    # raw may itself be a list/dict/str
                                    extracted = _extract_result_list(raw_val)
                                    if extracted is not None:
                                        return extracted
                                    if isinstance(raw_val, str):
                                        try:
                                            return _extract_result_list(json.loads(strip_code_fences(raw_val)))
                                        except Exception:
                                            return None
                                for k in ["results", "items", "data", "outputs", "responses"]:
                                    v = payload.get(k)
                                    if isinstance(v, list):
                                        return v
                                if all(isinstance(k, str) and k.isdigit() for k in payload.keys()):
                                    as_pairs = [(int(k), v) for k, v in payload.items() if isinstance(k, str) and k.isdigit()]
                                    as_pairs = [(i, v) for i, v in as_pairs if 0 <= i < len(batch_items)]
                                    if as_pairs:
                                        out = [None] * len(batch_items)
                                        for i, v in as_pairs:
                                            out[i] = v
                                        return out
                            return None

                        def _map_results_to_items(result_list: list[Any]) -> list[Any]:
                            mapped: list[Any | None] = [None] * len(batch_items)
                            for idx, r in enumerate(result_list):
                                if isinstance(r, dict) and isinstance(r.get("id"), int):
                                    rid = int(r.get("id"))
                                    if 0 <= rid < len(mapped):
                                        mapped[rid] = r
                                        continue
                                if idx < len(mapped):
                                    mapped[idx] = r

                            return [m if m is not None else {"error": "Missing result"} for m in mapped]

                        result_list = _extract_result_list(payload)
                        if result_list is None:
                            for item in batch_items:
                                st.session_state.stats_enrich_cache[item["key"]] = {
                                    "error": "Could not extract per-prompt results from response",
                                    "raw": {
                                        "response_preview": cleaned[:800],
                                    },
                                }
                            return

                        mapped_results = _map_results_to_items(result_list)
                        for item, parsed in zip(batch_items, mapped_results):
                            if isinstance(parsed, dict) and parsed.get("error"):
                                st.session_state.stats_enrich_cache[item["key"]] = {
                                    "error": str(parsed.get("error")),
                                    "raw": parsed,
                                }
                                continue

                            st.session_state.stats_enrich_cache[item["key"]] = {
                                "datasets": parsed.get("datasets") if isinstance(parsed, dict) else None,
                                "topics": parsed.get("topics") if isinstance(parsed, dict) else None,
                                "query_flavour": parsed.get("query_flavour") if isinstance(parsed, dict) else None,
                                "raw": parsed if isinstance(parsed, dict) else {},
                            }
                    except Exception as e:
                        for item in batch_items:
                            st.session_state.stats_enrich_cache[item["key"]] = {"error": str(e)}

                batches = [missing[i:i + int(batch_size)] for i in range(0, len(missing), int(batch_size))]
                for batch_idx, batch in enumerate(batches):
                    status.text(f"Processing batch {batch_idx + 1}/{len(batches)} ({len(batch)} prompts)")
                    process_batch(batch)
                    prog.progress(min(1.0, (batch_idx + 1) / max(1, len(batches))))
                    time_mod.sleep(0.1)

                status.empty()
                st.success(f"Enriched {len(missing)} prompts in {len(batches)} batches")

    has_enrichment_cache = len(st.session_state.stats_enrich_cache) > 0
    if has_enrichment_cache:
        def _apply_enrichment(row: pd.Series) -> dict[str, Any]:
            p = str(row.get("prompt") or "")
            if not p.strip():
                return {}
            key = _enrich_cache_key(p)
            val = st.session_state.stats_enrich_cache.get(key)
            if not isinstance(val, dict):
                return {}

            if val.get("error"):
                return {
                    "enrich_error": str(val.get("error")),
                }

            def as_list(x: Any) -> list[str]:
                if x is None:
                    return []
                if isinstance(x, list):
                    return [str(i).strip() for i in x if str(i).strip()]
                if isinstance(x, str):
                    s = x.strip()
                    if not s:
                        return []
                    return [s]
                return [str(x)]

            datasets = as_list(val.get("datasets"))
            topics = as_list(val.get("topics"))
            flavour = val.get("query_flavour")

            return {
                "enrich_datasets": ", ".join(datasets),
                "enrich_topics": ", ".join(topics),
                "enrich_query_flavour": str(flavour).strip() if isinstance(flavour, str) and str(flavour).strip() else "",
                "enrich_raw": json.dumps(val.get("raw") or {}, ensure_ascii=False),
            }

        enrich_cols = df.apply(_apply_enrichment, axis=1, result_type="expand")
        if isinstance(enrich_cols, pd.DataFrame) and len(enrich_cols.columns):
            df = pd.concat([df, enrich_cols], axis=1)

    total_traces = int(len(df))
    unique_users = int(df["user_id"].dropna().nunique()) if "user_id" in df.columns else 0
    unique_threads = int(df["session_id"].dropna().nunique()) if "session_id" in df.columns else 0

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
    else:
        success_rate = defer_rate = soft_error_rate = error_rate = 0.0

    cost_s = df["total_cost"].dropna() if "total_cost" in df.columns else pd.Series(dtype=float)
    lat_s = df["latency_seconds"].dropna() if "latency_seconds" in df.columns else pd.Series(dtype=float)

    mean_cost = float(cost_s.mean()) if len(cost_s) else 0.0
    median_cost = float(cost_s.median()) if len(cost_s) else 0.0
    p95_cost = float(cost_s.quantile(0.95)) if len(cost_s) else 0.0
    avg_latency = float(lat_s.mean()) if len(lat_s) else 0.0
    p95_latency = float(lat_s.quantile(0.95)) if len(lat_s) else 0.0

    st.markdown(f"### Summary Statistics ({(end_date - start_date).days + 1} days: {start_date} to {end_date})")
    summary_text = f"""📊 *GNW Trace Analytics Report*
📅 {start_date} → {end_date} ({(end_date - start_date).days + 1} days)

*Volume*
• Total traces: {total_traces:,}
• Unique users: {unique_users:,}
• Unique threads: {unique_threads:,}

*Outcomes*
• Success rate: {success_rate:.1%}
• Defer rate: {defer_rate:.1%}
• Soft error rate: {soft_error_rate:.1%}
• Error rate: {error_rate:.1%}

*Performance*
• Mean cost: ${mean_cost:.4f}
• Median cost: ${median_cost:.4f}
• p95 cost: ${p95_cost:.4f}
• Avg latency: {avg_latency:.2f}s
• p95 latency: {p95_latency:.2f}s

*Prompt utilisation*
• User-days: {util_user_days:,}
• Mean prompts/user/day: {util_mean_prompts:.2f}
• Median prompts/user/day: {util_median_prompts:.0f}
• p95 prompts/user/day: {util_p95_prompts:.0f}"""

    with st.expander("📋 Copy summary for Slack", expanded=False):
        st.code(summary_text, language=None)

    summary_tbl = pd.DataFrame(
        [
            {"Section": "Volume", "Metric": "Total traces", "Value": f"{total_traces:,}"},
            {"Section": "Volume", "Metric": "Unique users", "Value": f"{unique_users:,}"},
            {"Section": "Volume", "Metric": "Unique threads", "Value": f"{unique_threads:,}"},
            {"Section": "Outcomes", "Metric": "Success rate", "Value": f"{success_rate:.1%}"},
            {"Section": "Outcomes", "Metric": "Defer rate", "Value": f"{defer_rate:.1%}"},
            {"Section": "Outcomes", "Metric": "Soft error rate", "Value": f"{soft_error_rate:.1%}"},
            {"Section": "Outcomes", "Metric": "Error rate", "Value": f"{error_rate:.1%}"},
            {"Section": "Performance", "Metric": "Mean cost", "Value": f"${mean_cost:.4f}"},
            {"Section": "Performance", "Metric": "Median cost", "Value": f"${median_cost:.4f}"},
            {"Section": "Performance", "Metric": "p95 cost", "Value": f"${p95_cost:.4f}"},
            {"Section": "Performance", "Metric": "Average latency", "Value": f"{avg_latency:.2f}s"},
            {"Section": "Performance", "Metric": "P95 latency", "Value": f"{p95_latency:.2f}s"},
            {"Section": "Prompt utilisation", "Metric": "User-days", "Value": f"{util_user_days:,}"},
            {"Section": "Prompt utilisation", "Metric": "Mean prompts/user/day", "Value": f"{util_mean_prompts:.2f}"},
            {"Section": "Prompt utilisation", "Metric": "Median prompts/user/day", "Value": f"{util_median_prompts:.0f}"},
            {"Section": "Prompt utilisation", "Metric": "p95 prompts/user/day", "Value": f"{util_p95_prompts:.0f}"},
        ]
    )
    st.dataframe(summary_tbl, width="stretch", hide_index=True)

    report_rows = [
        {
            **{k: ("" if pd.isna(v) else v) for k, v in r.items()},
            "url": f"{base_thread_url.rstrip('/')}/{r.get('session_id')}" if r.get("session_id") else "",
        }
        for r in df.to_dict("records")
    ]
    report_csv_bytes = csv_bytes_any(report_rows)

    if st.button("💾 Save report CSV to disk", key="analytics_report_save_disk"):
        try:
            out_path = save_bytes_to_local_path(
                report_csv_bytes,
                str(st.session_state.get("csv_export_path") or ""),
                "stats_report_rows.csv",
            )
            st.toast(f"Saved: {out_path}")
        except Exception as e:
            st.error(f"Could not save: {e}")

    st.download_button(
        label="Download report data `.csv`",
        data=report_csv_bytes,
        file_name="stats_report_rows.csv",
        mime="text/csv",
        key="analytics_report_csv",
    )

    with st.expander("Raw rows"):
        st.dataframe(df, width="stretch")

    st.markdown("### Prompt utilisation")
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
                                alt.Tooltip("count():Q", title="User-days"),
                            ],
                        )
                        .properties(height=260)
                    )
                    st.altair_chart(hist, width="stretch")

                with right:
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
        base_daily["date"] = pd.to_datetime(base_daily["date"])

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
                mean_cost=("total_cost", lambda x: float(pd.to_numeric(x, errors="coerce").dropna().mean()) if len(pd.to_numeric(x, errors="coerce").dropna()) else 0.0),
                p95_cost=("total_cost", lambda x: _q(x, 0.95)),
                mean_latency=("latency_seconds", lambda x: float(pd.to_numeric(x, errors="coerce").dropna().mean()) if len(pd.to_numeric(x, errors="coerce").dropna()) else 0.0),
                p95_latency=("latency_seconds", lambda x: _q(x, 0.95)),
            )
            .reset_index()
            .sort_values("date")
        )

        st.markdown("### Daily trends")

        vol_chart = daily_volume_chart(daily_metrics)
        out_chart = daily_outcome_chart(daily_metrics)
        cost_chart = daily_cost_chart(daily_metrics)
        lat_chart = daily_latency_chart(daily_metrics)

        row1_c1, row1_c2 = st.columns(2)
        with row1_c1:
            st.altair_chart(vol_chart, width="stretch")
        with row1_c2:
            st.altair_chart(out_chart, width="stretch")

        row2_c1, row2_c2 = st.columns(2)
        with row2_c1:
            st.altair_chart(cost_chart, width="stretch")
        with row2_c2:
            st.altair_chart(lat_chart, width="stretch")

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

        st.markdown("### Starter prompt mix")

        left, right = st.columns(2)

        with left:
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

    st.markdown("### Distributions & Breakdowns")

    if ("prompt_len_chars" in df.columns or "prompt_len_words" in df.columns) and len(df):
        prompt_len_chars_s = df["prompt_len_chars"].dropna() if "prompt_len_chars" in df.columns else pd.Series(dtype="float")
        prompt_len_words_s = df["prompt_len_words"].dropna() if "prompt_len_words" in df.columns else pd.Series(dtype="float")
        prompt_len_chars_s = prompt_len_chars_s[prompt_len_chars_s > 0]
        prompt_len_words_s = prompt_len_words_s[prompt_len_words_s > 0]

        st.markdown("#### Prompt length")
        plc, plw = st.columns(2)

        with plc:
            st.markdown("##### Characters")
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
            st.markdown("##### Words")
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
        st.altair_chart(outcome_chart, width="stretch")
        if lat_chart_dist:
            st.altair_chart(lat_chart_dist, width="stretch")
    with dist_c2:
        if lang_chart:
            st.altair_chart(lang_chart, width="stretch")
        if cost_chart_dist:
            st.altair_chart(cost_chart_dist, width="stretch")

    st.markdown("### GNW analysis usage")

    pie_c1, pie_c2 = st.columns(2)
    with pie_c1:
        if "datasets_analysed" in df.columns:
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
            st.altair_chart(chart, width="stretch")

    if has_enrichment_cache and is_default_enrich_prompt:
        has_enrich_data = any(
            col in df.columns for col in ["enrich_query_flavour", "enrich_topics", "enrich_datasets"]
        )
        if has_enrich_data:
            st.markdown("### Enrichment Insights")

            pie_c1, pie_c2, pie_c3 = st.columns(3)

            with pie_c1:
                if "enrich_query_flavour" in df.columns:
                    chart = category_pie_chart(df["enrich_query_flavour"], "query_flavour", "Query flavour")
                    if chart:
                        st.altair_chart(chart, width="stretch")

            with pie_c2:
                if "enrich_topics" in df.columns:
                    chart = category_pie_chart(df["enrich_topics"], "topic", "Top topics", explode_csv=True)
                    if chart:
                        st.altair_chart(chart, width="stretch")

            with pie_c3:
                if "enrich_datasets" in df.columns:
                    chart = category_pie_chart(df["enrich_datasets"], "dataset", "Top datasets", explode_csv=True)
                    if chart:
                        st.altair_chart(chart, width="stretch")

    if has_enrichment_cache:
        st.markdown("### Prompt enrichment")

        breakdown_top_n = st.number_input("Breakdowns: top N categories", min_value=3, max_value=50, value=15)

        def _metric_table(group_key: str, keys: pd.Series) -> pd.DataFrame:
            if not isinstance(keys, pd.Series):
                return pd.DataFrame()

            # If `keys` comes from an explode(), it will typically have duplicate index labels.
            # Assigning it directly triggers pandas reindexing (and can crash with
            # "cannot reindex on an axis with duplicate labels"). Instead, expand the
            # base rows to match `keys` and assign by position.
            try:
                base = df.loc[keys.index].copy()
            except Exception:
                base = df.copy()
                keys = keys.reset_index(drop=True)
                base = base.reset_index(drop=True)

            base["_group"] = list(keys.values)
            base = base[base["_group"].notna()]
            if not len(base):
                return pd.DataFrame()

            def _q(s: pd.Series, q: float) -> float:
                try:
                    return float(s.dropna().quantile(q))
                except Exception:
                    return 0.0

            out = (
                base.groupby("_group", dropna=True)
                .agg(
                    traces=("trace_id", "count"),
                    success_rate=("outcome", lambda x: float((x == "ANSWER").mean())),
                    defer_rate=("outcome", lambda x: float((x == "DEFER").mean())),
                    soft_error_rate=("outcome", lambda x: float((x == "SOFT_ERROR").mean())),
                    error_rate=("outcome", lambda x: float((x == "ERROR").mean())),
                    mean_cost=("total_cost", lambda x: float(pd.to_numeric(x, errors="coerce").dropna().mean()) if len(pd.to_numeric(x, errors="coerce").dropna()) else 0.0),
                    p95_cost=("total_cost", lambda x: _q(pd.to_numeric(x, errors="coerce"), 0.95)),
                    mean_latency=("latency_seconds", lambda x: float(pd.to_numeric(x, errors="coerce").dropna().mean()) if len(pd.to_numeric(x, errors="coerce").dropna()) else 0.0),
                    p95_latency=("latency_seconds", lambda x: _q(pd.to_numeric(x, errors="coerce"), 0.95)),
                )
                .reset_index()
                .rename(columns={"_group": group_key})
            )
            out = out.sort_values("traces", ascending=False).head(int(breakdown_top_n))
            return out

        def _explode_csv_series(s: pd.Series) -> pd.Series:
            return (
                s.fillna("")
                .astype(str)
                .str.split(",")
                .explode()
                .astype(str)
                .str.strip()
                .replace({"": None})
            )

        st.markdown("### Enrichment breakdowns")

        lang_tbl = pd.DataFrame()
        if "lang_query" in df.columns:
            lang_tbl = _metric_table("language", df["lang_query"].replace({"": None}))
        if len(lang_tbl):
            st.markdown("#### By language")
            st.dataframe(lang_tbl, width="stretch", hide_index=True)
            chart = success_rate_bar_chart(lang_tbl, "language", "Language")
            if chart:
                st.altair_chart(chart, width="stretch")

        flav_tbl = pd.DataFrame()
        if "enrich_query_flavour" in df.columns:
            flav_tbl = _metric_table("query_flavour", df["enrich_query_flavour"].replace({"": None}))
        if len(flav_tbl):
            st.markdown("#### By query flavour")
            st.dataframe(flav_tbl, width="stretch", hide_index=True)
            chart = success_rate_bar_chart(flav_tbl, "query_flavour", "Query flavour")
            if chart:
                st.altair_chart(chart, width="stretch")

        if "enrich_datasets" in df.columns:
            keys = _explode_csv_series(df["enrich_datasets"])
            dt = _metric_table("dataset", keys)
            if len(dt):
                st.markdown("#### By dataset")
                st.dataframe(dt, width="stretch", hide_index=True)

        if "enrich_topics" in df.columns:
            keys = _explode_csv_series(df["enrich_topics"])
