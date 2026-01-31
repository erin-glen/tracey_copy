import os
from datetime import date, datetime, time, timedelta, timezone
import inspect
from pathlib import Path
from typing import Any

import streamlit as st

from utils import (
    as_float,
    classify_outcome,
    csv_bytes_any,
    fetch_traces_window,
    first_human_prompt,
    get_langfuse_headers,
    iso_utc,
    maybe_load_dotenv,
    normalize_trace_format,
    parse_trace_dt,
    final_ai_message,
)
from tabs import (
    render_session_urls,
    render_human_eval,
    render_product_dev,
    render_analytics,
    render_trace_explorer,
)


def main() -> None:
    st.set_page_config(page_title="GNW Langfuse Session Pull", layout="wide")

    maybe_load_dotenv()

    with st.sidebar:
        ### title with small text to the right
        st.title("💬🧠📎 Tracey. `v0.1`")
        st.caption("Think: _Clippy_... but for GNW traces.")
        st.markdown(
            "**ℹ️ What this tool does**\n\n"
            "Tracey allows you quickly pull and explore traces from Langfuse.\n"
            "_Ta, Trace!_\n\n"
            "- **📥 Fetch** a single set of traces once\n"
            "- **📊 Explore** the same dataset across tabs\n"
            "- **📋 Generate** reports & understand user behaviour"
            "- **🧪 Sample** for human eval & product mining\n"
        )

        st.markdown("---")

        st.markdown("**🌍 Environment**")
        environment = st.selectbox(
            "",
            options=["production", "production,default", "staging", "all"],
            index=0,
            label_visibility="collapsed",
        )

        envs: list[str] | None
        if environment == "all":
            envs = None
        else:
            envs = [e.strip() for e in environment.split(",") if e.strip()]

        st.markdown("**📅 Date range**")
        date_preset = st.selectbox(
            "",
            options=["All", "Last day", "Last week", "Last month", "Custom"],
            index=1,
            label_visibility="collapsed",
        )

        default_end = date.today()
        default_start = default_end - timedelta(days=7)

        use_date_filter = date_preset != "All"
        if date_preset == "Last day":
            start_date = default_end - timedelta(days=1)
            end_date = default_end
            st.caption(f"Using {start_date} to {end_date}")
        elif date_preset == "Last week":
            start_date = default_end - timedelta(days=7)
            end_date = default_end
            st.caption(f"Using {start_date} to {end_date}")
        elif date_preset == "Last month":
            start_date = default_end - timedelta(days=30)
            end_date = default_end
            st.caption(f"Using {start_date} to {end_date}")
        elif date_preset == "Custom":
            start_date = st.date_input(
                "Start date",
                value=default_start,
            )
            end_date = st.date_input(
                "End date",
                value=default_end,
            )
        else:
            start_date = default_start
            end_date = default_end

        if "stats_traces" not in st.session_state:
            st.session_state.stats_traces = []

        if "csv_export_path" not in st.session_state:
            st.session_state.csv_export_path = str(Path.home() / "Downloads")

        st.markdown(
            """
<style>
/* Sidebar button styling */
section[data-testid="stSidebar"] div[data-testid="stButton"] button[kind="primary"] {
  background: #e5484d !important;
  color: white !important;
  border: 1px solid rgba(255,255,255,0.25) !important;
}
section[data-testid="stSidebar"] div[data-testid="stButton"] button[kind="primary"]:hover {
  background: #d83f45 !important;
}

/* Make the raw CSV download button visually prominent when data is available */
section[data-testid="stSidebar"] div[data-testid="stDownloadButton"] button {
  background: #1fb6a6 !important;
  color: white !important;
  border: 1px solid rgba(255,255,255,0.25) !important;
}
section[data-testid="stSidebar"] div[data-testid="stDownloadButton"] button:hover {
  background: #17a596 !important;
}
</style>
            """,
            unsafe_allow_html=True,
        )

        c_fetch, c_dl = st.columns(2)
        with c_fetch:
            fetch_clicked = st.button("🚀 Fetch traces", type="primary", use_container_width=True)
        with c_dl:
            traces_for_dl = st.session_state.get("stats_traces", [])
            if traces_for_dl:
                normed_for_dl = [normalize_trace_format(t) for t in traces_for_dl]
                out_rows = []
                for n in normed_for_dl:
                    prompt = first_human_prompt(n)
                    answer = final_ai_message(n)
                    dt = parse_trace_dt(n)
                    out_rows.append(
                        {
                            "trace_id": n.get("id"),
                            "timestamp": dt,
                            "date": dt.date() if dt else None,
                            "environment": n.get("environment"),
                            "session_id": n.get("sessionId"),
                            "user_id": n.get("userId")
                            or (n.get("metadata") or {}).get("user_id")
                            or (n.get("metadata") or {}).get("userId"),
                            "latency_seconds": as_float(n.get("latency")),
                            "total_cost": as_float(n.get("totalCost")),
                            "outcome": classify_outcome(n, answer or ""),
                            "prompt": prompt,
                            "answer": answer,
                        }
                    )

                raw_csv_bytes = csv_bytes_any(out_rows)

                st.download_button(
                    label="⬇️ Download raw csv",
                    data=raw_csv_bytes,
                    file_name="gnw_traces_raw.csv",
                    mime="text/csv",
                    key="raw_csv_download",
                    use_container_width=True,
                )
            else:
                st.button("⬇️ Download raw csv", disabled=True, use_container_width=True)

        fetch_status = st.empty()

        with st.expander("🔎 Fetch debug (Langfuse API)", expanded=False):
            dbg = st.session_state.get("fetch_debug")
            if isinstance(dbg, dict) and dbg:
                st.json(dbg)
            else:
                st.caption("Fetch traces to populate request/response metadata.")

        st.markdown("---")

        with st.expander("🔐 Credentials", expanded=False):
            public_key = st.text_input(
                "LANGFUSE_PUBLIC_KEY",
                value=os.getenv("LANGFUSE_PUBLIC_KEY", ""),
            )
            secret_key = st.text_input(
                "LANGFUSE_SECRET_KEY",
                value=os.getenv("LANGFUSE_SECRET_KEY", ""),
                type="password",
            )
            base_url = st.text_input(
                "LANGFUSE_BASE_URL",
                value=os.getenv("LANGFUSE_BASE_URL", ""),
            )
            gemini_api_key = st.text_input(
                "GEMINI_API_KEY",
                value=os.getenv("GEMINI_API_KEY", os.getenv("GOOGLE_API_KEY", "")),
                type="password",
            )

        with st.expander("⚠️ Limits", expanded=False):
            stats_page_limit = st.number_input(
                "Max pages",
                min_value=1,
                max_value=500,
                value=50,
                key="stats_page_limit",
            )
            stats_page_size = st.number_input(
                "Traces per page",
                min_value=1,
                max_value=100,
                value=100,
                key="stats_page_size",
                help="This is the API page size (per request). It is not the overall max; that's controlled by 'Max traces'.",
            )
            stats_max_traces = st.number_input(
                "Max traces",
                min_value=1,
                max_value=200_000,
                value=25_000,
                key="stats_max_traces",
            )

            base_thread_url = f"https://www.{'staging.' if environment == 'staging' else ''}globalnaturewatch.org/app/threads"

        with st.expander("⬇️ Exports", expanded=False):
            st.text_input(
                "Local export path",
                value=str(st.session_state.get("csv_export_path") or ""),
                key="csv_export_path",
                placeholder="e.g. ~/Downloads or /tmp/gnw_traces.csv",
                help=(
                    "Browser downloads can't pick a destination folder. "
                    "If you provide a directory or full .csv path here, the app can also save exports directly to disk."
                ),
            )

        if fetch_clicked:
            if not public_key or not secret_key or not base_url:
                st.error("Missing LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_BASE_URL")
            elif not use_date_filter:
                st.error("Pick a date range other than 'All' to fetch traces.")
            else:
                if start_date > end_date:
                    st.error("Start date must be on or before end date")
                else:
                    headers = get_langfuse_headers(public_key, secret_key)
                    from_iso = iso_utc(datetime.combine(start_date, time.min).replace(tzinfo=timezone.utc))
                    to_iso = iso_utc(datetime.combine(end_date, time.max).replace(tzinfo=timezone.utc))
                    with fetch_status.status("Fetching traces...", expanded=False):
                        fetch_debug: dict[str, Any] = {}
                        sig = None
                        try:
                            sig = inspect.signature(fetch_traces_window)
                        except Exception:
                            sig = None

                        supports_debug_out = bool(sig and "debug_out" in sig.parameters)
                        supports_page_size = bool(sig and "page_size" in sig.parameters)

                        call_kwargs: dict[str, Any] = {
                            "base_url": base_url,
                            "headers": headers,
                            "from_iso": from_iso,
                            "to_iso": to_iso,
                            "envs": envs,
                            "page_limit": int(stats_page_limit),
                            "max_traces": int(stats_max_traces),
                            "retry": 2,
                            "backoff": 0.5,
                        }
                        if supports_page_size:
                            call_kwargs["page_size"] = int(stats_page_size)

                        if supports_debug_out:
                            call_kwargs["debug_out"] = fetch_debug

                        if supports_debug_out or supports_page_size:
                            traces = fetch_traces_window(
                                **call_kwargs,
                            )
                        else:
                            # Fallback for older function objects (e.g. Streamlit hot-reload holding
                            # an older signature). You can restart Streamlit to pick up debug support.
                            fetch_debug.update(
                                {
                                    "url": f"{base_url.rstrip('/')}/api/public/traces",
                                    "from_iso": from_iso,
                                    "to_iso": to_iso,
                                    "envs": envs,
                                    "page_limit": int(stats_page_limit),
                                    "page_size": int(stats_page_size),
                                    "max_traces": int(stats_max_traces),
                                    "note": "fetch_traces_window() does not support debug_out in this runtime; restart Streamlit to enable per-page debug.",
                                }
                            )
                            traces = fetch_traces_window(
                                base_url=base_url,
                                headers=headers,
                                from_iso=from_iso,
                                to_iso=to_iso,
                                envs=envs,
                                page_limit=int(stats_page_limit),
                                max_traces=int(stats_max_traces),
                                retry=2,
                                backoff=0.5,
                            )
                        fetch_status.status(f"Fetched {len(traces)} traces", state="complete", expanded=False)
                    st.session_state.fetch_debug = fetch_debug
                    st.session_state.stats_traces = traces
                    st.rerun()

    tabs = st.tabs([
        "📊 Analytics Report",
        "🔗 Conversation URLs",
        "✅ Human eval tool",
        "🧠 Product intelligence",
        "🔎 Trace Explorer",
    ])

    with tabs[0]:
        render_analytics(
            public_key=public_key,
            secret_key=secret_key,
            base_url=base_url,
            base_thread_url=base_thread_url,
            gemini_api_key=gemini_api_key,
            use_date_filter=use_date_filter,
            start_date=start_date,
            end_date=end_date,
            envs=envs,
            stats_page_limit=stats_page_limit,
            stats_max_traces=stats_max_traces,
        )

    with tabs[1]:
        render_session_urls(
            base_thread_url=base_thread_url,
        )

    with tabs[2]:
        render_human_eval(
            base_thread_url=base_thread_url,
        )

    with tabs[3]:
        render_product_dev(
            base_thread_url=base_thread_url,
            gemini_api_key=gemini_api_key,
        )

    with tabs[4]:
        render_trace_explorer(
            base_thread_url=base_thread_url,
        )


if __name__ == "__main__":
    main()
