"""Analytics Report page."""

import streamlit as st

from utils.shared_ui import check_authentication, render_sidebar
from tabs.analytics import render as render_analytics


st.set_page_config(page_title="Analytics Report - Tracey", page_icon="📊", layout="wide")

if not check_authentication():
    st.stop()

config = render_sidebar()

render_analytics(
    base_url=config["base_url"],
    base_thread_url=config["base_thread_url"],
    gemini_api_key=config["gemini_api_key"],
    use_date_filter=config["use_date_filter"],
    start_date=config["start_date"],
    end_date=config["end_date"],
    envs=config["envs"],
    stats_max_traces=config["stats_max_traces"],
)
