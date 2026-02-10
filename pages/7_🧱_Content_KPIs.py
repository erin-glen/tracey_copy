"""Content KPIs page."""

import streamlit as st

from tabs.content_kpis import render as render_content_kpis
from utils.shared_ui import check_authentication, render_sidebar


st.set_page_config(page_title="Content KPIs - Tracey", page_icon="🧱", layout="wide")

if not check_authentication():
    st.stop()

config = render_sidebar()

render_content_kpis(
    public_key=config["public_key"],
    secret_key=config["secret_key"],
    base_url=config["base_url"],
    base_thread_url=config["base_thread_url"],
    gemini_api_key=config["gemini_api_key"],
    use_date_filter=config["use_date_filter"],
    start_date=config["start_date"],
    end_date=config["end_date"],
    envs=config["envs"],
    stats_page_limit=config["stats_page_limit"],
    stats_max_traces=config["stats_max_traces"],
)
