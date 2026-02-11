"""CodeAct QA page."""

import streamlit as st

from tabs.codeact_qa import render as render_codeact_qa
from utils.shared_ui import check_authentication, render_sidebar


st.set_page_config(page_title="CodeAct QA - Tracey", page_icon="🧩", layout="wide")

if not check_authentication():
    st.stop()

config = render_sidebar()

render_codeact_qa(base_thread_url=config["base_thread_url"])
