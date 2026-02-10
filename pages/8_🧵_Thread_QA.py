"""Thread QA page."""

import streamlit as st

from tabs.thread_qa import render as render_thread_qa
from utils.shared_ui import check_authentication, render_sidebar


st.set_page_config(page_title="Thread QA - Tracey", page_icon="🧵", layout="wide")

if not check_authentication():
    st.stop()

config = render_sidebar()

render_thread_qa(
    public_key=config["public_key"],
    secret_key=config["secret_key"],
    base_url=config["base_url"],
    base_thread_url=config["base_thread_url"],
)
