"""QA Samples page."""

import streamlit as st

from tabs.qa_samples import render as render_qa_samples
from utils.shared_ui import check_authentication, render_sidebar


st.set_page_config(page_title="QA Samples - Tracey", page_icon="📦", layout="wide")

if not check_authentication():
    st.stop()

config = render_sidebar()

render_qa_samples(
    public_key=config["public_key"],
    secret_key=config["secret_key"],
    base_url=config["base_url"],
    base_thread_url=config["base_thread_url"],
)
