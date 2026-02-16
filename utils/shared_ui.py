"""Shared UI components for multipage Streamlit app."""

import os
import hmac
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any

import streamlit as st

import time as time_mod

from utils.data_helpers import maybe_load_dotenv, iso_utc, csv_bytes_any, init_session_state
from utils.fetch_throttle import TokenBucket, SharedBudget
from utils.langfuse_api import fetch_traces_window, fetch_user_first_seen, invalidate_user_first_seen_cache, get_langfuse_headers
from utils.trace_parsing import (
    normalize_trace_format,
    first_human_prompt,
    final_ai_message,
    parse_trace_dt,
    classify_outcome,
    active_turn_prompt,
    active_turn_answer,
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


@st.cache_data(show_spinner=False)
def _load_internal_user_ids() -> set[str]:
    try:
        path = Path(__file__).resolve().parent / "fixtures" / "internal_users.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        users = payload.get("internal_users") if isinstance(payload, dict) else None
        if not isinstance(users, list):
            return set()
        return {str(u).strip() for u in users if str(u).strip()}
    except Exception:
        return set()


def _trace_user_id(t: dict[str, Any]) -> str:
    try:
        return str(
            t.get("userId")
            or (t.get("metadata") or {}).get("user_id")
            or (t.get("metadata") or {}).get("userId")
            or ""
        ).strip()
    except Exception:
        return ""


def _is_machine_user_id(user_id: str) -> bool:
    try:
        return "machine" in str(user_id).lower()
    except Exception:
        return False


def _apply_trace_filters(
    *,
    traces_raw: list[dict[str, Any]],
    exclude_internal_users: bool,
    internal_user_ids: set[str],
) -> list[dict[str, Any]]:
    traces = [t for t in (traces_raw or []) if not _is_machine_user_id(_trace_user_id(t))]
    if exclude_internal_users and internal_user_ids:
        traces = [t for t in traces if _trace_user_id(t) not in internal_user_ids]
    return traces


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

    default_end = date.today()
    default_start = default_end - timedelta(days=7)
    init_session_state(
        {
            "start_date": default_start,
            "end_date": default_end,
            "use_date_filter": True,
            # Shadow keys survive Streamlit widget-key cleanup between page navigations.
            "_shadow_date_preset": "Last week",
            "_shadow_exclude_internal": True,
        }
    )
    
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
            key="environment",
        )

        envs: list[str] | None
        if environment == "all":
            envs = None
        else:
            envs = [e.strip() for e in environment.split(",") if e.strip()]

        st.markdown("**📅 Date range**")
        preset_options = ["Custom", "Last day", "Last 3 days", "Last week", "Last 2 weeks", "Last month"]
        _saved_preset = st.session_state.get("_shadow_date_preset", "Last week")
        if _saved_preset not in preset_options:
            _saved_preset = "Last week"
        prev_date_preset = str(st.session_state.get("_prev_date_preset") or _saved_preset)
        date_preset = st.selectbox(
            "Date preset",
            options=preset_options,
            index=preset_options.index(_saved_preset),
            label_visibility="collapsed",
            key="date_preset",
        )
        st.session_state._shadow_date_preset = date_preset

        use_date_filter = True
        recompute_preset_dates = bool(date_preset != "Custom" and date_preset != prev_date_preset)
        if date_preset == "Custom":
            start_date = st.date_input(
                "Start date",
                value=st.session_state.get("start_date") or default_start,
                key="_start_date_widget",
            )
            end_date = st.date_input(
                "End date",
                value=st.session_state.get("end_date") or default_end,
                key="_end_date_widget",
            )
            st.session_state.start_date = start_date
            st.session_state.end_date = end_date
        else:
            if (not recompute_preset_dates) and isinstance(st.session_state.get("start_date"), date) and isinstance(
                st.session_state.get("end_date"), date
            ):
                start_date = st.session_state.start_date
                end_date = st.session_state.end_date
            else:
                if date_preset == "Last day":
                    start_date = default_end - timedelta(days=1)
                    end_date = default_end
                elif date_preset == "Last 3 days":
                    start_date = default_end - timedelta(days=3)
                    end_date = default_end
                elif date_preset == "Last week":
                    start_date = default_end - timedelta(days=7)
                    end_date = default_end
                elif date_preset == "Last 2 weeks":
                    start_date = default_end - timedelta(days=14)
                    end_date = default_end
                else:
                    start_date = default_end - timedelta(days=30)
                    end_date = default_end
                st.session_state.start_date = start_date
                st.session_state.end_date = end_date

            st.caption(f"Using {start_date} to {end_date}")

        st.session_state._prev_date_preset = date_preset
        st.session_state.use_date_filter = use_date_filter

        # Initialize traces in session state
        if "stats_traces" not in st.session_state:
            st.session_state.stats_traces = []
        if "stats_traces_raw" not in st.session_state:
            st.session_state.stats_traces_raw = []

        internal_user_ids = _load_internal_user_ids()
        exclude_internal = bool(st.session_state.get("_shadow_exclude_internal", True))

        # Re-derive the filtered view only when inputs change.
        _filter_key = (len(st.session_state.get("stats_traces_raw", [])), exclude_internal)
        if st.session_state.get("_trace_filter_key") != _filter_key:
            st.session_state.stats_traces = _apply_trace_filters(
                traces_raw=st.session_state.get("stats_traces_raw", []),
                exclude_internal_users=exclude_internal,
                internal_user_ids=internal_user_ids,
            )
            st.session_state._trace_filter_key = _filter_key

        # --- Trace status (based on raw traces so conditional sections stay stable) ---
        traces_loaded = st.session_state.get("stats_traces_raw", [])
        has_raw_traces = bool(traces_loaded)
        traces_loaded_str = f"_({len(traces_loaded):,} loaded)_" if traces_loaded else ""

        st.markdown(f"**📊 Traces** {traces_loaded_str}")

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
            if has_raw_traces:
                # Cache CSV bytes in session state so we don't re-normalise on every page nav.
                _raw_count = len(st.session_state.get("stats_traces_raw", []))
                if st.session_state.get("_raw_csv_cache_count") != _raw_count:
                    normed_for_dl = [normalize_trace_format(t) for t in st.session_state.stats_traces_raw]
                    out_rows = []
                    for n in normed_for_dl:
                        prompt = active_turn_prompt(n) or first_human_prompt(n)
                        answer = active_turn_answer(n) or final_ai_message(n)
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
                    st.session_state._raw_csv_bytes = csv_bytes_any(out_rows)
                    st.session_state._raw_csv_cache_count = _raw_count
                st.download_button(
                    label="⬇️ Raw traces csv",
                    data=st.session_state._raw_csv_bytes,
                    file_name=f"gnw_traces_raw_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    key="raw_csv_download",
                    width="stretch",
                )
            else:
                st.button("⬇️ Download csv", disabled=True, width="stretch")

        fetch_status = st.empty()

        fetch_warn = st.session_state.get("fetch_warning")
        if isinstance(fetch_warn, dict) and str(fetch_warn.get("message") or "").strip():
            st.warning(str(fetch_warn.get("message") or ""))

        # --- CSV upload as alternative to LF pull ---
        with st.expander("📤 Upload trace CSV", expanded=False):
            st.caption("Upload a previously downloaded trace CSV instead of pulling from Langfuse.")
            uploaded_files = st.file_uploader(
                "Trace CSV",
                type=["csv"],
                key="trace_csv_uploader",
                label_visibility="collapsed",
                accept_multiple_files=True,
            )
            upload_clicked = st.button(
                "Load CSV",
                disabled=not bool(uploaded_files),
                width="stretch",
            )

        # --- User data enrichment (shown after traces are loaded) ---
        user_fetch_clicked = False
        user_invalidate_clicked = False
        if has_raw_traces:
            if "analytics_user_first_seen" not in st.session_state:
                st.session_state.analytics_user_first_seen = None
            if "analytics_user_first_seen_debug" not in st.session_state:
                st.session_state.analytics_user_first_seen_debug = {}

            import pandas as pd
            has_user_data = isinstance(st.session_state.get("analytics_user_first_seen"), pd.DataFrame) and len(st.session_state.analytics_user_first_seen) > 0
            users_loaded_str = f"_({len(st.session_state.analytics_user_first_seen):,} loaded)_" if has_user_data else ""
            st.markdown(f"**👥 User data** {users_loaded_str}")

            _saved_excl = st.session_state.get("_shadow_exclude_internal", True)
            st.checkbox(
                "Exclude internal users",
                value=_saved_excl,
                key="exclude_internal_users_checkbox",
                help="If enabled, remove internal/test accounts from the fetched user table.",
            )
            st.session_state._shadow_exclude_internal = st.session_state.get("exclude_internal_users_checkbox", True)

            u_c1, u_c2 = st.columns(2)
            with u_c1:
                btn_type = "secondary" if has_user_data else "primary"
                user_fetch_clicked = st.button("🚀 Fetch users", type=btn_type, width="stretch")
            with u_c2:
                if has_user_data:
                    cached_df = st.session_state.analytics_user_first_seen
                    st.download_button(
                        "⬇️ Users csv",
                        csv_bytes_any(cached_df.assign(first_seen=cached_df["first_seen"].astype(str)).to_dict("records")),
                        "user_first_seen.csv",
                        "text/csv",
                        key="sidebar_user_first_seen_csv",
                        width="stretch",
                    )
                else:
                    st.button("⬇️ Users csv", disabled=True, width="stretch")

            user_invalidate_clicked = st.button(
                "🧹 Invalidate user cache",
                disabled=not has_user_data,
                width="stretch",
            )

        st.markdown("---")
        st.markdown("**⚙️ Configuration**")
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
            stats_page_size = st.number_input(
                "Traces per page",
                min_value=1,
                max_value=100,
                value=50,
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
                max_value=10,
                value=5,
                key="stats_parallel_workers",
                help="Number of chunks to fetch in parallel.",
            )
            stats_http_timeout_s = st.number_input(
                "HTTP timeout (seconds)",
                min_value=5,
                max_value=600,
                value=180,
                key="stats_http_timeout_s",
                help="If multi-month fetches time out, increase this (e.g. 120-300s).",
            )
            stats_req_per_sec = st.number_input(
                "Max requests/sec",
                min_value=1,
                max_value=50,
                value=10,
                key="stats_req_per_sec",
                help="Global rate limit across all parallel workers.",
            )
            stats_req_burst = st.number_input(
                "Burst capacity",
                min_value=1,
                max_value=20,
                value=3,
                key="stats_req_burst",
                help="Max requests that can fire instantly before throttling.",
            )

        base_thread_url = f"https://www.{'staging.' if environment == 'staging' else ''}globalnaturewatch.org/app/threads"

        # Store config in session state for access by pages
        st.session_state.langfuse_public_key = public_key
        st.session_state.langfuse_secret_key = secret_key
        st.session_state.langfuse_base_url = base_url
        st.session_state.gemini_api_key = gemini_api_key
        st.session_state.base_thread_url = base_thread_url
        st.session_state.envs = envs
        # NOTE: Do not assign to keys that are bound to widgets (environment/date_preset)
        # after the widget is instantiated. Streamlit manages those keys.
        st.session_state.use_date_filter = use_date_filter

    # Handle CSV upload (outside the sidebar layout)
    if upload_clicked and uploaded_files:
        _handle_csv_upload(uploaded_files)

    # Handle user data fetch
    if user_fetch_clicked:
        _handle_user_fetch(
            public_key=public_key,
            secret_key=secret_key,
            base_url=base_url,
            envs=envs,
            stats_page_limit=int(stats_page_limit),
            stats_page_size=int(stats_page_size),
            exclude_internal_users=bool(st.session_state.get("_shadow_exclude_internal", True)),
            internal_user_ids=internal_user_ids,
        )

    # Handle user cache invalidation
    if user_invalidate_clicked:
        _handle_user_invalidate(
            base_url=base_url,
            envs=envs,
            stats_page_limit=int(stats_page_limit),
            stats_page_size=int(stats_page_size),
        )

    # Handle fetch (outside the sidebar layout, but still within this function)
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
            stats_req_per_sec=float(stats_req_per_sec),
            stats_req_burst=int(stats_req_burst),
            fetch_status=fetch_status,
        )

    return get_app_config()


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
    stats_req_per_sec: float,
    stats_req_burst: int,
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
            stats_req_per_sec=stats_req_per_sec,
            stats_req_burst=stats_req_burst,
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
            stats_req_per_sec=stats_req_per_sec,
            stats_req_burst=stats_req_burst,
            fetch_status=fetch_status,
        )

    traces_raw = [t for t in traces if not _is_machine_user_id(_trace_user_id(t))]
    st.session_state.stats_traces_raw = traces_raw

    internal_user_ids = _load_internal_user_ids()
    exclude_internal_users = bool(st.session_state.get("_shadow_exclude_internal", True))
    traces = _apply_trace_filters(
        traces_raw=traces_raw,
        exclude_internal_users=exclude_internal_users,
        internal_user_ids=internal_user_ids,
    )
    st.session_state.stats_traces = traces
    st.session_state.fetch_warning = _build_fetch_warning(
        fetch_debug=st.session_state.get("fetch_debug"),
        start_date=start_date,
        end_date=end_date,
        traces=traces,
        stats_max_traces=stats_max_traces,
        stats_page_limit=stats_page_limit,
    )
    st.rerun()


def _build_fetch_warning(
    *,
    fetch_debug: Any,
    start_date: date,
    end_date: date,
    traces: list[dict[str, Any]],
    stats_max_traces: int,
    stats_page_limit: int,
) -> dict[str, Any]:
    failed_chunks = 0
    early_stop_chunks = 0
    page_limit_chunks = 0
    stopped_reasons: list[str] = []
    missing_ranges: list[str] = []

    if isinstance(fetch_debug, dict) and isinstance(fetch_debug.get("chunks"), list):
        for c in fetch_debug.get("chunks") or []:
            if not isinstance(c, dict):
                continue
            if c.get("error"):
                failed_chunks += 1
                _start = c.get("start", "?")
                _end = c.get("end", "?")
                missing_ranges.append(f"{_start} → {_end}")
                continue
            dbg = c.get("debug")
            if isinstance(dbg, dict):
                reason = dbg.get("stopped_early_reason")
                if isinstance(reason, str) and reason.strip():
                    early_stop_chunks += 1
                    stopped_reasons.append(reason.strip())
                    _start = c.get("start", "?")
                    _end = c.get("end", "?")
                    missing_ranges.append(f"{_start} → {_end} (partial)")
                if dbg.get("stopped_due_to_page_limit"):
                    page_limit_chunks += 1

    single_reason = None
    if isinstance(fetch_debug, dict) and "chunks" not in fetch_debug:
        r = fetch_debug.get("stopped_early_reason")
        if isinstance(r, str) and r.strip():
            single_reason = r.strip()
        if fetch_debug.get("stopped_due_to_page_limit"):
            page_limit_chunks += 1

    truncated = bool(isinstance(stats_max_traces, int) and stats_max_traces > 0 and len(traces) >= stats_max_traces)
    incomplete = bool(failed_chunks or early_stop_chunks or page_limit_chunks or truncated or single_reason)

    if not incomplete:
        return {}

    lines: list[str] = []
    lines.append(
        "Some trace fetch requests did not fully complete. Your dataset may be incomplete, which can show up as missing days in charts."
    )
    lines.append(f"Date range: {start_date.isoformat()} → {end_date.isoformat()}")
    lines.append(f"Loaded traces: {len(traces):,}")
    if failed_chunks:
        lines.append(f"Failed chunks: {failed_chunks}")
    if early_stop_chunks:
        lines.append(f"Chunks stopped early: {early_stop_chunks}")
    if page_limit_chunks:
        lines.append(f"Chunks hit page limit: {page_limit_chunks} (max pages = {int(stats_page_limit)})")
    if truncated:
        lines.append(f"Hit max traces limit: {int(stats_max_traces):,}")
    if single_reason:
        lines.append(f"Stopped early: {single_reason}")
    if missing_ranges:
        lines.append(f"Missing/incomplete date ranges: {', '.join(missing_ranges[:5])}")
        if len(missing_ranges) > 5:
            lines.append(f"  … and {len(missing_ranges) - 5} more")
    if stopped_reasons:
        uniq = []
        seen = set()
        for r in stopped_reasons:
            if r not in seen:
                seen.add(r)
                uniq.append(r)
            if len(uniq) >= 3:
                break
        if uniq:
            lines.append("Examples: " + " | ".join(uniq))
    lines.append("Suggested fixes: reduce date range, increase 'Max pages', increase 'HTTP timeout', or raise 'Max traces'.")

    return {"message": "\n".join(lines)}


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
    stats_req_per_sec: float,
    stats_req_burst: int,
    fetch_status: Any,
) -> list[dict[str, Any]]:
    """Fetch traces in chunks for large date ranges."""
    chunks: list[tuple[int, date, date]] = []
    current_start = start_date
    idx = 1
    while current_start <= end_date:
        current_end = min(current_start + timedelta(days=2), end_date)
        chunks.append((idx, current_start, current_end))
        current_start = current_end + timedelta(days=1)
        idx += 1

    # Shared rate limiter + trace budget across all workers
    rate_limiter = TokenBucket(rate=stats_req_per_sec, capacity=stats_req_burst)
    budget = SharedBudget(total=stats_max_traces)

    def fetch_chunk(chunk_info: tuple[int, date, date]) -> tuple[int, date, date, list[dict[str, Any]], dict[str, Any]]:
        chunk_idx, chunk_start, chunk_end = chunk_info
        from_iso = iso_utc(datetime.combine(chunk_start, time.min).replace(tzinfo=timezone.utc))
        to_iso = iso_utc(datetime.combine(chunk_end, time.max).replace(tzinfo=timezone.utc))

        chunk_debug: dict[str, Any] = {}
        chunk_traces = fetch_traces_window(
            base_url=base_url,
            headers=headers,
            from_iso=from_iso,
            to_iso=to_iso,
            envs=envs,
            page_size=stats_page_size,
            page_limit=stats_page_limit,
            max_traces=stats_max_traces,
            retry=4,
            backoff=1.0,
            http_timeout_s=stats_http_timeout_s,
            debug_out=chunk_debug,
            rate_limiter=rate_limiter,
            budget=budget,
        )
        return (chunk_idx, chunk_start, chunk_end, chunk_traces, chunk_debug)

    all_traces: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    fetch_debug: dict[str, Any] = {
        "chunks": [],
        "total_traces": 0,
        "parallel_workers": stats_parallel_workers,
        "rate_limit_req_per_sec": stats_req_per_sec,
        "rate_limit_burst": stats_req_burst,
    }
    completed_count = 0
    failed_count = 0
    fetch_start = time_mod.monotonic()
    # Rolling window for recent throughput (last N chunk completions)
    _recent_completions: list[tuple[float, int]] = []  # (monotonic_time, cumulative_traces)

    def _format_eta(remaining_chunks: int, recent: list[tuple[float, int]]) -> str:
        """Estimate time remaining from rolling chunk throughput."""
        if len(recent) < 2 or remaining_chunks <= 0:
            return ""
        t0, n0 = recent[0]
        t1, n1 = recent[-1]
        dt = t1 - t0
        dn = n1 - n0
        if dt <= 0 or dn <= 0:
            return ""
        # Average traces per second over the rolling window
        traces_per_s = dn / dt
        # Average traces per chunk over the rolling window
        chunks_in_window = len(recent) - 1
        traces_per_chunk = dn / chunks_in_window if chunks_in_window > 0 else 0
        if traces_per_chunk <= 0:
            return ""
        remaining_traces = traces_per_chunk * remaining_chunks
        eta_s = remaining_traces / traces_per_s
        if eta_s < 60:
            return f"~{eta_s:.0f}s left"
        elif eta_s < 3600:
            return f"~{eta_s / 60:.1f}m left"
        return f"~{eta_s / 3600:.1f}h left"

    total_chunks = len(chunks)
    with fetch_status.status(f"Fetching traces in {total_chunks} chunks ({stats_parallel_workers} workers, {stats_req_per_sec:.0f} req/s)…", expanded=True) as status:
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

                    # Rolling throughput window (keep last 8 data points)
                    now = time_mod.monotonic()
                    _recent_completions.append((now, len(all_traces)))
                    if len(_recent_completions) > 8:
                        _recent_completions.pop(0)

                    elapsed = now - fetch_start
                    rate = len(all_traces) / elapsed if elapsed > 0 else 0
                    remaining = total_chunks - completed_count
                    eta = _format_eta(remaining, _recent_completions)
                    fail_note = f" ⚠️ {failed_count} failed" if failed_count else ""
                    eta_note = f" · {eta}" if eta else ""

                    status.update(
                        label=(
                            f"Chunk {completed_count}/{total_chunks}"
                            f" — {len(all_traces):,} traces ({rate:.0f}/s)"
                            f"{eta_note}{fail_note}"
                        )
                    )

                    # If global budget exhausted, cancel remaining futures
                    if budget.exhausted():
                        for f in futures:
                            f.cancel()
                        status.update(
                            label=f"Global trace limit reached ({stats_max_traces:,}). {len(all_traces):,} traces collected."
                        )
                        break

                except Exception as e:
                    failed_count += 1
                    chunk_info = futures[future]
                    fetch_debug["chunks"].append({
                        "chunk": chunk_info[0],
                        "start": chunk_info[1].isoformat(),
                        "end": chunk_info[2].isoformat(),
                        "error": str(e),
                    })
                    remaining = total_chunks - completed_count
                    status.update(
                        label=(
                            f"Chunk {completed_count}/{total_chunks}"
                            f" — ⚠️ chunk {chunk_info[0]} failed ({chunk_info[1]}→{chunk_info[2]})"
                        )
                    )

        fetch_debug["total_traces"] = len(all_traces)
        elapsed = time_mod.monotonic() - fetch_start
        fail_summary = f" ({failed_count} chunks failed)" if failed_count else ""
        status.update(
            label=f"Fetched {len(all_traces):,} traces from {total_chunks} chunks in {elapsed:.1f}s{fail_summary}",
            state="complete",
        )

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
    stats_req_per_sec: float,
    stats_req_burst: int,
    fetch_status: Any,
) -> list[dict[str, Any]]:
    """Fetch traces in a single request for small date ranges."""
    from_iso = iso_utc(datetime.combine(start_date, time.min).replace(tzinfo=timezone.utc))
    to_iso = iso_utc(datetime.combine(end_date, time.max).replace(tzinfo=timezone.utc))

    rate_limiter = TokenBucket(rate=stats_req_per_sec, capacity=stats_req_burst)

    with fetch_status.status("Fetching traces...", expanded=True) as status:
        fetch_debug: dict[str, Any] = {}
        fetch_start = time_mod.monotonic()
        # Track recent page completions for rolling throughput / ETA
        _page_times: list[tuple[float, int]] = []  # (monotonic, traces_so_far)

        def on_progress(pages_done: int, traces_so_far: int):
            now = time_mod.monotonic()
            elapsed = now - fetch_start
            _page_times.append((now, traces_so_far))
            if len(_page_times) > 10:
                _page_times.pop(0)

            # Rolling throughput from recent pages
            if len(_page_times) >= 2:
                dt = _page_times[-1][0] - _page_times[0][0]
                dn = _page_times[-1][1] - _page_times[0][1]
                recent_rate = dn / dt if dt > 0 else 0
            else:
                recent_rate = traces_so_far / elapsed if elapsed > 0 else 0

            # ETA: we don't know total pages, but we know page_limit is the upper bound.
            # Use traces_per_page to estimate remaining pages.
            traces_per_page = traces_so_far / pages_done if pages_done > 0 else 0
            # If the last page was full, there's probably more data.
            # Estimate remaining time from recent rate.
            eta_str = ""
            if recent_rate > 0 and traces_so_far < stats_max_traces:
                # Rough: assume we'll fetch up to max_traces or run out of pages
                remaining_traces = stats_max_traces - traces_so_far
                remaining_pages = stats_page_limit - pages_done
                if remaining_pages > 0 and traces_per_page > 0:
                    est_traces_left = min(remaining_traces, remaining_pages * traces_per_page)
                    eta_s = est_traces_left / recent_rate
                    if eta_s < 60:
                        eta_str = f" · ~{eta_s:.0f}s left"
                    elif eta_s < 3600:
                        eta_str = f" · ~{eta_s / 60:.1f}m left"

            status.update(
                label=f"Page {pages_done} — {traces_so_far:,} traces ({recent_rate:.0f}/s){eta_str}",
            )

        traces = fetch_traces_window(
            base_url=base_url,
            headers=headers,
            from_iso=from_iso,
            to_iso=to_iso,
            envs=envs,
            page_size=stats_page_size,
            page_limit=stats_page_limit,
            max_traces=stats_max_traces,
            retry=4,
            backoff=1.0,
            http_timeout_s=stats_http_timeout_s,
            debug_out=fetch_debug,
            rate_limiter=rate_limiter,
            on_progress=on_progress,
        )
        elapsed = time_mod.monotonic() - fetch_start
        status.update(
            label=f"Fetched {len(traces):,} traces in {elapsed:.1f}s",
            state="complete",
        )

    st.session_state.fetch_debug = fetch_debug
    return traces


def _parse_trace_csv_bytes_list(raw_csvs: list[bytes]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    import io
    import csv as csv_mod

    traces: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    stats = {
        "files": 0,
        "rows": 0,
        "loaded": 0,
        "dupe_trace_id": 0,
        "machine_user": 0,
    }

    for raw in (raw_csvs or []):
        if not raw:
            continue
        stats["files"] += 1
        text = raw.decode("utf-8-sig") if raw[:3] == b"\xef\xbb\xbf" else raw.decode("utf-8")
        reader = csv_mod.DictReader(io.StringIO(text))
        for row in reader:
            stats["rows"] += 1

            trace_id = str(row.get("trace_id") or "").strip()
            if trace_id and trace_id in seen_ids:
                stats["dupe_trace_id"] += 1
                continue

            user_id = str(row.get("user_id") or "").strip()
            if _is_machine_user_id(user_id):
                stats["machine_user"] += 1
                continue

            trace: dict[str, Any] = {
                "id": trace_id,
                "timestamp": row.get("timestamp") or "",
                "environment": row.get("environment") or "",
                "sessionId": row.get("session_id") or "",
                "userId": user_id,
                "latency": as_float(row.get("latency_seconds")),
                "totalCost": as_float(row.get("total_cost")),
                "input": {"messages": [{"type": "human", "content": row.get("prompt") or ""}]},
                "output": {"messages": [{"type": "ai", "content": row.get("answer") or ""}]},
                "metadata": {},
                "_from_csv": True,
            }
            traces.append(trace)
            stats["loaded"] += 1
            if trace_id:
                seen_ids.add(trace_id)

    return traces, stats


def _handle_csv_upload(uploaded_files: list[Any]) -> None:
    """Handle CSV upload as alternative to Langfuse fetch.

    Reconstructs minimal trace dicts from the CSV columns so that
    existing parsing/analytics pipelines work transparently.
    """
    try:
        raw_csvs: list[bytes] = []
        for f in (uploaded_files or []):
            try:
                raw_csvs.append(f.getvalue())
            except Exception:
                continue

        traces, stats = _parse_trace_csv_bytes_list(raw_csvs)

        st.session_state.stats_traces_raw = traces

        internal_user_ids = _load_internal_user_ids()
        exclude_internal_users = bool(st.session_state.get("_shadow_exclude_internal", True))
        st.session_state.stats_traces = _apply_trace_filters(
            traces_raw=traces,
            exclude_internal_users=exclude_internal_users,
            internal_user_ids=internal_user_ids,
        )

        st.session_state.fetch_debug = {"source": "csv_upload", **stats}
        st.session_state.fetch_warning = {}
        st.toast(f"Loaded {len(traces):,} traces from CSV")
        st.rerun()
    except Exception as e:
        st.error(f"Failed to parse CSV: {e}")


def _handle_user_fetch(
    public_key: str,
    secret_key: str,
    base_url: str,
    envs: list[str] | None,
    stats_page_limit: int,
    stats_page_size: int,
    exclude_internal_users: bool,
    internal_user_ids: set[str],
) -> None:
    """Handle the fetch users button click."""
    import pandas as pd

    if not public_key or not secret_key or not base_url:
        st.error("Missing LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_BASE_URL")
        return

    debug_out: dict[str, Any] = {}
    try:
        headers = get_langfuse_headers(public_key, secret_key)
        from_iso = datetime(2025, 9, 17, tzinfo=timezone.utc).isoformat()
        first_seen_map = fetch_user_first_seen(
            base_url=base_url,
            headers=headers,
            from_iso=from_iso,
            envs=envs,
            page_size=stats_page_size,
            page_limit=stats_page_limit,
            retry=3,
            backoff=0.75,
            debug_out=debug_out,
        )
        user_first_seen_df = pd.DataFrame(
            [{"user_id": k, "first_seen": v} for k, v in (first_seen_map or {}).items()]
        )
        if len(user_first_seen_df):
            user_first_seen_df["user_id"] = user_first_seen_df["user_id"].astype(str).map(lambda x: x.strip())
            user_first_seen_df = user_first_seen_df[user_first_seen_df["user_id"].ne("")]
            user_first_seen_df = user_first_seen_df[
                ~user_first_seen_df["user_id"].astype(str).str.contains("machine", case=False, na=False)
            ]
            if exclude_internal_users and internal_user_ids:
                user_first_seen_df = user_first_seen_df[~user_first_seen_df["user_id"].isin(internal_user_ids)]
            user_first_seen_df["first_seen"] = pd.to_datetime(
                user_first_seen_df["first_seen"], errors="coerce", utc=True
            )
            user_first_seen_df = user_first_seen_df.dropna(subset=["first_seen"])
            user_first_seen_df = user_first_seen_df.sort_values("first_seen")
        st.session_state.analytics_user_first_seen = user_first_seen_df
        st.session_state.analytics_user_first_seen_debug = debug_out
        st.toast(f"Fetched first-seen for {len(user_first_seen_df):,} users")
        st.rerun()
    except Exception as e:
        st.session_state.analytics_user_first_seen_debug = debug_out
        st.error(f"Could not fetch user first-seen: {e}")


def _handle_user_invalidate(
    base_url: str,
    envs: list[str] | None,
    stats_page_limit: int,
    stats_page_size: int,
) -> None:
    """Handle the invalidate user cache button click."""
    from_iso = datetime(2025, 9, 17, tzinfo=timezone.utc).isoformat()
    removed = invalidate_user_first_seen_cache(
        base_url=base_url,
        from_iso=from_iso,
        envs=envs,
        page_size=stats_page_size,
        page_limit=stats_page_limit,
    )
    st.session_state.analytics_user_first_seen = None
    if removed:
        st.toast("Cache cleared ✅")
    else:
        st.toast("No cache file found (cleared session cache) ⚠️")
    st.rerun()
