"""Shared UI components for multipage Streamlit app."""

import os
import hmac
import inspect
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, time, timedelta, timezone
from typing import Any

import streamlit as st

from utils.data_helpers import maybe_load_dotenv, iso_utc, csv_bytes_any, init_session_state
from utils.langfuse_api import fetch_traces_window, get_langfuse_headers
from utils.trace_parsing import (
    normalize_trace_format,
    first_human_prompt,
    final_ai_message,
    parse_trace_dt,
    classify_outcome,
)
from utils.data_helpers import as_float


def configure_page(title: str = "Tracey", layout: str = "wide") -> None:
    """Configure Streamlit page settings."""
    st.set_page_config(page_title=title, layout=layout)


def check_authentication() -> bool:
    """Check if user is authenticated. Returns True if authenticated or no password required."""
    maybe_load_dotenv()
    
    app_password = os.getenv("APP_PASSWORD", "")
    if "app_authenticated" not in st.session_state:
        st.session_state.app_authenticated = False

    if app_password and not st.session_state.app_authenticated:
        st.title("🔒 Tracey")
        st.caption("Enter the app password to continue.")

        pw = st.text_input("Password", type="password", key="_app_password_input")
        col_login, _ = st.columns([1, 3])
        with col_login:
            if st.button("Log in", type="primary"):
                if hmac.compare_digest(str(pw), str(app_password)):
                    st.session_state.app_authenticated = True
                    st.session_state.pop("_app_password_input", None)
                    st.rerun()
                else:
                    st.error("Incorrect password")
        return False
    
    return True


def get_app_config() -> dict[str, Any]:
    """Get current app configuration from session state."""
    return {
        "public_key": st.session_state.get("langfuse_public_key", ""),
        "secret_key": st.session_state.get("langfuse_secret_key", ""),
        "base_url": st.session_state.get("langfuse_base_url", ""),
        "gemini_api_key": st.session_state.get("gemini_api_key", ""),
        "base_thread_url": st.session_state.get("base_thread_url", ""),
        "environment": st.session_state.get("environment", "production"),
        "envs": st.session_state.get("envs"),
        "start_date": st.session_state.get("start_date"),
        "end_date": st.session_state.get("end_date"),
        "use_date_filter": st.session_state.get("use_date_filter", True),
        "stats_page_limit": st.session_state.get("stats_page_limit", 500),
        "stats_max_traces": st.session_state.get("stats_max_traces", 25000),
    }


def render_sidebar() -> dict[str, Any]:
    """Render the shared sidebar and return configuration dict."""
    maybe_load_dotenv()
    app_password = os.getenv("APP_PASSWORD", "")
    
    with st.sidebar:
        if app_password and st.session_state.get("app_authenticated"):
            if st.button("Log out", width="stretch"):
                st.session_state.app_authenticated = False
                st.rerun()

        with st.expander("**⁉️ Getting started...**", expanded=False):
            st.markdown(
                "- Configure credentials below (expand 🔐 Credentials)\n"
                "- Select 🌍 environment and 📅 date range below\n"
                "- Click '🚀 Fetch traces' to load data\n"
                "- Navigate to a page using the sidebar menu\n\n"
            )

        st.markdown("**🌍 Environment**")
        environment = st.selectbox(
            "Environment",
            options=["production"],
            index=0,
            label_visibility="collapsed",
            key="environment_select",
        )

        envs: list[str] | None
        if environment == "all":
            envs = None
        else:
            envs = [e.strip() for e in environment.split(",") if e.strip()]

        st.markdown("**📅 Date range**")
        date_preset = st.selectbox(
            "Date preset",
            options=["Custom", "Last day", "Last 3 days", "Last week", "Last month", "All"],
            index=1,
            label_visibility="collapsed",
            key="date_preset_select",
        )

        default_end = date.today()
        default_start = default_end - timedelta(days=7)

        use_date_filter = date_preset != "All"
        if date_preset == "Last day":
            start_date = default_end - timedelta(days=1)
            end_date = default_end
            st.caption(f"Using {start_date} to {end_date}")
        elif date_preset == "Last 3 days":
            start_date = default_end - timedelta(days=3)
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
            start_date = st.date_input("Start date", value=default_start)
            end_date = st.date_input("End date", value=default_end)
        else:
            start_date = date(2025, 9, 17)  # Launch date
            end_date = default_end
            st.caption(f"Using {start_date} to {end_date}")

        # Initialize traces in session state
        if "stats_traces" not in st.session_state:
            st.session_state.stats_traces = []

        # Sidebar button styling
        st.markdown(
            """
<style>
section[data-testid="stSidebar"] div[data-testid="stButton"] button[kind="primary"] {
  background: #e5484d !important;
  color: white !important;
  border: 1px solid rgba(255,255,255,0.25) !important;
}
section[data-testid="stSidebar"] div[data-testid="stButton"] button[kind="primary"]:hover {
  background: #d83f45 !important;
}
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
            fetch_clicked = st.button("🚀 Fetch traces", type="primary", width="stretch")
        with c_dl:
            traces_for_dl = st.session_state.get("stats_traces", [])
            if traces_for_dl:
                normed_for_dl = [normalize_trace_format(t) for t in traces_for_dl]
                out_rows = []
                for n in normed_for_dl:
                    prompt = first_human_prompt(n)
                    answer = final_ai_message(n)
                    dt = parse_trace_dt(n)
                    out_rows.append({
                        "trace_id": n.get("id"),
                        "timestamp": dt,
                        "date": dt.date() if dt else None,
                        "environment": n.get("environment"),
                        "session_id": n.get("sessionId"),
                        "user_id": n.get("userId") or (n.get("metadata") or {}).get("user_id") or (n.get("metadata") or {}).get("userId"),
                        "latency_seconds": as_float(n.get("latency")),
                        "total_cost": as_float(n.get("totalCost")),
                        "outcome": classify_outcome(n, answer or ""),
                        "prompt": prompt,
                        "answer": answer,
                    })
                raw_csv_bytes = csv_bytes_any(out_rows)
                st.download_button(
                    label="⬇️ Download csv",
                    data=raw_csv_bytes,
                    file_name="gnw_traces_raw.csv",
                    mime="text/csv",
                    key="raw_csv_download",
                    width="stretch",
                )
            else:
                st.button("⬇️ Download csv", disabled=True, width="stretch")

        fetch_status = st.empty()

        with st.expander("🔎 Debug Langfuse call", expanded=False):
            dbg = st.session_state.get("fetch_debug")
            if isinstance(dbg, dict) and dbg:
                st.json(dbg)
            else:
                st.caption("Fetch traces to populate request/response metadata.")

        with st.expander("🔐 Credentials", expanded=False):
            public_key = st.text_input(
                "LANGFUSE_PUBLIC_KEY",
                value=os.getenv("LANGFUSE_PUBLIC_KEY", ""),
                key="langfuse_public_key_input",
            )
            secret_key = str(st.session_state.get("langfuse_secret_key") or os.getenv("LANGFUSE_SECRET_KEY", ""))
            base_url = st.text_input(
                "LANGFUSE_BASE_URL",
                value=os.getenv("LANGFUSE_BASE_URL", ""),
                key="langfuse_base_url_input",
            )
            gemini_override = st.text_input(
                "BYO GEMINI_API_KEY (_optional_)",
                value="",
                type="password",
                key="gemini_api_key_override_input",
                help="If left blank, Tracey will use GEMINI_API_KEY (or GOOGLE_API_KEY) from the environment.",
            )
            gemini_api_key = str(gemini_override or st.session_state.get("gemini_api_key") or os.getenv("GEMINI_API_KEY", os.getenv("GOOGLE_API_KEY", "")))

        with st.expander("⚠️ Limits", expanded=False):
            stats_page_limit = st.number_input(
                "Max pages",
                min_value=1,
                max_value=1000,
                value=500,
                key="stats_page_limit",
            )
            stats_http_timeout_s = st.number_input(
                "HTTP timeout (seconds)",
                min_value=5,
                max_value=600,
                value=60,
                key="stats_http_timeout_s",
                help="If multi-month fetches time out, increase this (e.g. 120-300s).",
            )
            stats_page_size = st.number_input(
                "Traces per page",
                min_value=1,
                max_value=100,
                value=100,
                key="stats_page_size",
                help="API page size (per request).",
            )
            stats_max_traces = st.number_input(
                "Max traces",
                min_value=1,
                max_value=200000,
                value=25000,
                key="stats_max_traces",
            )
            stats_parallel_workers = st.number_input(
                "Parallel workers",
                min_value=1,
                max_value=5,
                value=3,
                key="stats_parallel_workers",
                help="Number of weekly chunks to fetch in parallel.",
            )

        base_thread_url = f"https://www.{'staging.' if environment == 'staging' else ''}globalnaturewatch.org/app/threads"

        # Store config in session state for access by pages
        st.session_state.langfuse_public_key = public_key
        st.session_state.langfuse_secret_key = secret_key
        st.session_state.langfuse_base_url = base_url
        st.session_state.gemini_api_key = gemini_api_key
        st.session_state.base_thread_url = base_thread_url
        st.session_state.environment = environment
        st.session_state.envs = envs
        st.session_state.start_date = start_date
        st.session_state.end_date = end_date
        st.session_state.use_date_filter = use_date_filter

        # Handle fetch
        if fetch_clicked:
            _handle_fetch(
                public_key=public_key,
                secret_key=secret_key,
                base_url=base_url,
                start_date=start_date,
                end_date=end_date,
                envs=envs,
                stats_page_limit=int(stats_page_limit),
                stats_page_size=int(stats_page_size),
                stats_max_traces=int(stats_max_traces),
                stats_http_timeout_s=float(stats_http_timeout_s),
                stats_parallel_workers=int(stats_parallel_workers),
                fetch_status=fetch_status,
            )

    return {
        "public_key": public_key,
        "secret_key": secret_key,
        "base_url": base_url,
        "gemini_api_key": gemini_api_key,
        "base_thread_url": base_thread_url,
        "environment": environment,
        "envs": envs,
        "start_date": start_date,
        "end_date": end_date,
        "use_date_filter": use_date_filter,
        "stats_page_limit": stats_page_limit,
        "stats_max_traces": stats_max_traces,
    }


def _handle_fetch(
    public_key: str,
    secret_key: str,
    base_url: str,
    start_date: date,
    end_date: date,
    envs: list[str] | None,
    stats_page_limit: int,
    stats_page_size: int,
    stats_max_traces: int,
    stats_http_timeout_s: float,
    stats_parallel_workers: int,
    fetch_status: Any,
) -> None:
    """Handle the fetch traces button click."""
    if not public_key or not secret_key or not base_url:
        st.error("Missing LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_BASE_URL")
        return

    if start_date > end_date:
        st.error("Start date must be on or before end date")
        return

    headers = get_langfuse_headers(public_key, secret_key)
    date_range_days = (end_date - start_date).days
    chunk_by_block = date_range_days > 5

    if chunk_by_block:
        traces = _fetch_chunked(
            headers=headers,
            base_url=base_url,
            start_date=start_date,
            end_date=end_date,
            envs=envs,
            stats_page_limit=stats_page_limit,
            stats_page_size=stats_page_size,
            stats_max_traces=stats_max_traces,
            stats_http_timeout_s=stats_http_timeout_s,
            stats_parallel_workers=stats_parallel_workers,
            fetch_status=fetch_status,
        )
    else:
        traces = _fetch_single(
            headers=headers,
            base_url=base_url,
            start_date=start_date,
            end_date=end_date,
            envs=envs,
            stats_page_limit=stats_page_limit,
            stats_page_size=stats_page_size,
            stats_max_traces=stats_max_traces,
            stats_http_timeout_s=stats_http_timeout_s,
            fetch_status=fetch_status,
        )

    st.session_state.stats_traces = traces
    st.rerun()


def _fetch_chunked(
    headers: dict[str, str],
    base_url: str,
    start_date: date,
    end_date: date,
    envs: list[str] | None,
    stats_page_limit: int,
    stats_page_size: int,
    stats_max_traces: int,
    stats_http_timeout_s: float,
    stats_parallel_workers: int,
    fetch_status: Any,
) -> list[dict[str, Any]]:
    """Fetch traces in chunks for large date ranges."""
    chunks: list[tuple[int, date, date]] = []
    current_start = start_date
    idx = 1
    while current_start <= end_date:
        current_end = min(current_start + timedelta(days=4), end_date)
        chunks.append((idx, current_start, current_end))
        current_start = current_end + timedelta(days=1)
        idx += 1

    sig = None
    try:
        sig = inspect.signature(fetch_traces_window)
    except Exception:
        pass
    supports_debug_out = bool(sig and "debug_out" in sig.parameters)
    supports_page_size = bool(sig and "page_size" in sig.parameters)
    supports_http_timeout = bool(sig and "http_timeout_s" in sig.parameters)

    def fetch_chunk(chunk_info: tuple[int, date, date]) -> tuple[int, date, date, list[dict[str, Any]], dict[str, Any]]:
        chunk_idx, chunk_start, chunk_end = chunk_info
        from_iso = iso_utc(datetime.combine(chunk_start, time.min).replace(tzinfo=timezone.utc))
        to_iso = iso_utc(datetime.combine(chunk_end, time.max).replace(tzinfo=timezone.utc))

        chunk_debug: dict[str, Any] = {}
        call_kwargs: dict[str, Any] = {
            "base_url": base_url,
            "headers": headers,
            "from_iso": from_iso,
            "to_iso": to_iso,
            "envs": envs,
            "page_limit": stats_page_limit,
            "max_traces": stats_max_traces,
            "retry": 2,
            "backoff": 0.5,
        }
        if supports_page_size:
            call_kwargs["page_size"] = stats_page_size
        if supports_http_timeout:
            call_kwargs["http_timeout_s"] = stats_http_timeout_s
        if supports_debug_out:
            call_kwargs["debug_out"] = chunk_debug

        chunk_traces = fetch_traces_window(**call_kwargs)
        return (chunk_idx, chunk_start, chunk_end, chunk_traces, chunk_debug)

    all_traces: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    fetch_debug: dict[str, Any] = {"chunks": [], "total_traces": 0, "parallel_workers": stats_parallel_workers}
    completed_count = 0

    with fetch_status.status(f"Fetching traces in {len(chunks)} chunks of 5 days ({stats_parallel_workers} parallel)...", expanded=True) as status:
        with ThreadPoolExecutor(max_workers=stats_parallel_workers) as executor:
            futures = {executor.submit(fetch_chunk, chunk): chunk for chunk in chunks}

            for future in as_completed(futures):
                completed_count += 1
                try:
                    chunk_idx, chunk_start, chunk_end, chunk_traces, chunk_debug = future.result()

                    new_traces = 0
                    for trace in chunk_traces:
                        trace_id = trace.get("id")
                        if isinstance(trace_id, str) and trace_id not in seen_ids:
                            seen_ids.add(trace_id)
                            all_traces.append(trace)
                            new_traces += 1

                    fetch_debug["chunks"].append({
                        "chunk": chunk_idx,
                        "start": chunk_start.isoformat(),
                        "end": chunk_end.isoformat(),
                        "fetched": len(chunk_traces),
                        "new": new_traces,
                        "debug": chunk_debug,
                    })

                    status.update(label=f"Completed {completed_count}/{len(chunks)} chunks ({len(all_traces)} traces so far)")
                except Exception as e:
                    chunk_info = futures[future]
                    fetch_debug["chunks"].append({
                        "chunk": chunk_info[0],
                        "start": chunk_info[1].isoformat(),
                        "end": chunk_info[2].isoformat(),
                        "error": str(e),
                    })
                    status.update(label=f"Chunk {chunk_info[0]} failed: {e}")

        fetch_debug["total_traces"] = len(all_traces)
        status.update(label=f"Fetched {len(all_traces)} total traces from {len(chunks)} chunks", state="complete")

    st.session_state.fetch_debug = fetch_debug
    return all_traces


def _fetch_single(
    headers: dict[str, str],
    base_url: str,
    start_date: date,
    end_date: date,
    envs: list[str] | None,
    stats_page_limit: int,
    stats_page_size: int,
    stats_max_traces: int,
    stats_http_timeout_s: float,
    fetch_status: Any,
) -> list[dict[str, Any]]:
    """Fetch traces in a single request for small date ranges."""
    from_iso = iso_utc(datetime.combine(start_date, time.min).replace(tzinfo=timezone.utc))
    to_iso = iso_utc(datetime.combine(end_date, time.max).replace(tzinfo=timezone.utc))

    with fetch_status.status("Fetching traces...", expanded=False):
        fetch_debug: dict[str, Any] = {}
        sig = None
        try:
            sig = inspect.signature(fetch_traces_window)
        except Exception:
            pass

        supports_debug_out = bool(sig and "debug_out" in sig.parameters)
        supports_page_size = bool(sig and "page_size" in sig.parameters)
        supports_http_timeout = bool(sig and "http_timeout_s" in sig.parameters)

        call_kwargs: dict[str, Any] = {
            "base_url": base_url,
            "headers": headers,
            "from_iso": from_iso,
            "to_iso": to_iso,
            "envs": envs,
            "page_limit": stats_page_limit,
            "max_traces": stats_max_traces,
            "retry": 2,
            "backoff": 0.5,
        }
        if supports_page_size:
            call_kwargs["page_size"] = stats_page_size
        if supports_http_timeout:
            call_kwargs["http_timeout_s"] = stats_http_timeout_s
        if supports_debug_out:
            call_kwargs["debug_out"] = fetch_debug

        traces = fetch_traces_window(**call_kwargs)
        fetch_status.status(f"Fetched {len(traces)} traces", state="complete", expanded=False)

    st.session_state.fetch_debug = fetch_debug
    return traces
