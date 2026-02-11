"""CodeAct Explorer page."""

import streamlit as st

from tabs.codeact_explorer import render as render_codeact_explorer
from utils.shared_ui import check_authentication, render_sidebar


st.set_page_config(page_title="CodeAct Explorer - Tracey", page_icon="🔎", layout="wide")

if not check_authentication():
    st.stop()

config = render_sidebar()

render_codeact_explorer(base_thread_url=config["base_thread_url"])
