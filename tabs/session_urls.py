"""Session URLs tab."""

from typing import Any

import pandas as pd
import streamlit as st

from utils import csv_bytes_any, normalize_trace_format


def render(
    base_thread_url: str,
) -> None:
    """Render the Session URLs tab."""
    st.subheader("Session URLs")

    traces: list[dict[str, Any]] = st.session_state.get("stats_traces", [])

    if not traces:
        st.info(
            "This tab turns the currently loaded traces into a list of **unique session links** you can click to open the "
            "GNW Threads UI for each conversation.\n\n"
            "Use the sidebar **🚀 Fetch traces** button first, then come back here to export session URLs."
        )
        return

    normed = [normalize_trace_format(t) for t in traces]

    session_ids = set()
    rows: list[dict[str, Any]] = []
    for n in normed:
        sid = n.get("sessionId")
        if sid and sid not in session_ids:
            session_ids.add(sid)
            url = f"{base_thread_url.rstrip('/')}/{sid}"
            rows.append({
                "session_id": sid,
                "url": url,
            })

    if not rows:
        st.warning("No sessions found in the fetched traces.")
        return

    st.write(f"**{len(rows)}** unique conversation threads")

    df = pd.DataFrame(rows)
    df["link"] = df["url"].apply(lambda u: f'<a href="{u}" target="_blank">{u}</a>')

    st.markdown(
        df[["session_id", "link"]].to_html(escape=False, index=False),
        unsafe_allow_html=True,
    )

    csv_data = csv_bytes_any(rows)
    st.download_button(
        label="Download CSV",
        data=csv_data,
        file_name="gnw_session_urls.csv",
        mime="text/csv",
        key="session_urls_csv",
    )
