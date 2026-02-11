"""Shared UI components for multipage Streamlit app."""

import os
import hmac
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any

import streamlit as st

from utils.data_helpers import maybe_load_dotenv, iso_utc, csv_bytes_any, init_session_state
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
            "environment": "production",
            "date_preset": "Last week",
            "start_date": default_start,
            "end_date": default_end,
            "use_date_filter": True,
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
        preset_options = ["Custom", "Last day", "Last 3 days", "Last week", "Last month"]
        _preset_state = str(st.session_state.get("date_preset") or "Last week")
        if _preset_state not in preset_options:
            _preset_state = "Last week"
        prev_date_preset = str(st.session_state.get("_prev_date_preset") or _preset_state)
        date_preset = st.selectbox(
            "Date preset",
            options=preset_options,
            index=preset_options.index(_preset_state),
            label_visibility="collapsed",
            key="date_preset",
        )

        use_date_filter = True
        recompute_preset_dates = bool(date_preset != "Custom" and date_preset != prev_date_preset)
        if date_preset == "Custom":
            start_date = st.date_input(
                "Start date",
                value=st.session_state.get("start_date") or default_start,
                key="start_date",
            )
            end_date = st.date_input(
                "End date",
                value=st.session_state.get("end_date") or default_end,
                key="end_date",
            )
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

         # --- Trace status ---
        traces_loaded = st.session_state.get("stats_traces", [])
        traces_loaded_str = f"_({len(traces_loaded):,} traces loaded)_" if traces_loaded else ""

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
            traces_for_dl = st.session_state.get("stats_traces", [])
            if traces_for_dl:
                normed_for_dl = [normalize_trace_format(t) for t in traces_for_dl]
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
                raw_csv_bytes = csv_bytes_any(out_rows)
                st.download_button(
                    label="⬇️ Raw traces csv",
                    data=raw_csv_bytes,
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
            uploaded_file = st.file_uploader(
                "Trace CSV",
                type=["csv"],
                key="trace_csv_uploader",
                label_visibility="collapsed",
            )
            upload_clicked = st.button("Load CSV", disabled=uploaded_file is None, width="stretch")

        internal_user_ids = _load_internal_user_ids()

        # --- User data enrichment (shown after traces are loaded) ---
        user_fetch_clicked = False
        user_invalidate_clicked = False
        if traces_loaded:
            if "analytics_user_first_seen" not in st.session_state:
                st.session_state.analytics_user_first_seen = None
            if "analytics_user_first_seen_debug" not in st.session_state:
                st.session_state.analytics_user_first_seen_debug = {}

            import pandas as pd
            has_user_data = isinstance(st.session_state.get("analytics_user_first_seen"), pd.DataFrame) and len(st.session_state.analytics_user_first_seen) > 0
            users_loaded_str = f"_({len(st.session_state.analytics_user_first_seen):,} users loaded)_" if has_user_data else ""
            st.markdown(f"**👥 User data** {users_loaded_str}")

            exclude_internal_users = st.checkbox(
                "Exclude internal users",
                value=True,
                key="exclude_internal_users_checkbox",
                help="If enabled, remove internal/test accounts from the fetched user table.",
            )

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

        base_thread_url = f"https://www.{'staging.' if environment == 'staging' else ''}globalnaturewatch.org/app/threads"

        # Store config in session state for access by pages
        st.session_state.langfuse_public_key = public_key
        st.session_state.langfuse_secret_key = secret_key
        st.session_state.langfuse_base_url = base_url
        st.session_state.gemini_api_key = gemini_api_key
        st.session_state.base_thread_url = base_thread_url
        st.session_state.envs = envs
        # NOTE: Do not assign to keys that are bound to widgets (environment/date_preset/start_date/end_date)
        # after the widget is instantiated. Streamlit manages these keys.
        st.session_state.use_date_filter = use_date_filter

    # Handle CSV upload (outside the sidebar layout)
    if upload_clicked and uploaded_file is not None:
        _handle_csv_upload(uploaded_file)

    # Handle user data fetch
    if user_fetch_clicked:
        _handle_user_fetch(
            public_key=public_key,
            secret_key=secret_key,
            base_url=base_url,
            envs=envs,
            stats_page_limit=int(stats_page_limit),
            stats_page_size=int(stats_page_size),
            exclude_internal_users=bool(st.session_state.get("exclude_internal_users_checkbox", True)),
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

    if isinstance(fetch_debug, dict) and isinstance(fetch_debug.get("chunks"), list):
        for c in fetch_debug.get("chunks") or []:
            if not isinstance(c, dict):
                continue
            if c.get("error"):
                failed_chunks += 1
                continue
            dbg = c.get("debug")
            if isinstance(dbg, dict):
                reason = dbg.get("stopped_early_reason")
                if isinstance(reason, str) and reason.strip():
                    early_stop_chunks += 1
                    stopped_reasons.append(reason.strip())
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
            retry=2,
            backoff=0.5,
            http_timeout_s=stats_http_timeout_s,
            debug_out=chunk_debug,
        )
        return (chunk_idx, chunk_start, chunk_end, chunk_traces, chunk_debug)

    all_traces: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    fetch_debug: dict[str, Any] = {"chunks": [], "total_traces": 0, "parallel_workers": stats_parallel_workers}
    completed_count = 0

    with fetch_status.status(f"Fetching traces in {len(chunks)} chunks of 3 days ({stats_parallel_workers} parallel)...", expanded=True) as status:
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
        traces = fetch_traces_window(
            base_url=base_url,
            headers=headers,
            from_iso=from_iso,
            to_iso=to_iso,
            envs=envs,
            page_size=stats_page_size,
            page_limit=stats_page_limit,
            max_traces=stats_max_traces,
            retry=2,
            backoff=0.5,
            http_timeout_s=stats_http_timeout_s,
            debug_out=fetch_debug,
        )
        fetch_status.status(f"Fetched {len(traces)} traces", state="complete", expanded=False)

    st.session_state.fetch_debug = fetch_debug
    return traces


def _handle_csv_upload(uploaded_file: Any) -> None:
    """Handle CSV upload as alternative to Langfuse fetch.

    Reconstructs minimal trace dicts from the CSV columns so that
    existing parsing/analytics pipelines work transparently.
    """
    import io
    import csv as csv_mod

    try:
        raw = uploaded_file.getvalue()
        text = raw.decode("utf-8-sig") if raw[:3] == b"\xef\xbb\xbf" else raw.decode("utf-8")
        reader = csv_mod.DictReader(io.StringIO(text))
        traces: list[dict[str, Any]] = []
        for row in reader:
            trace: dict[str, Any] = {
                "id": row.get("trace_id") or "",
                "timestamp": row.get("timestamp") or "",
                "environment": row.get("environment") or "",
                "sessionId": row.get("session_id") or "",
                "userId": row.get("user_id") or "",
                "latency": as_float(row.get("latency_seconds")),
                "totalCost": as_float(row.get("total_cost")),
                "input": {"messages": [{"type": "human", "content": row.get("prompt") or ""}]},
                "output": {"messages": [{"type": "ai", "content": row.get("answer") or ""}]},
                "metadata": {},
                "_from_csv": True,
            }
            traces.append(trace)

        st.session_state.stats_traces = traces
        st.session_state.fetch_debug = {"source": "csv_upload", "rows": len(traces)}
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
