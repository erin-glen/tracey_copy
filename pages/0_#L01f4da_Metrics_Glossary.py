"""Metrics Glossary page."""

import streamlit as st

from utils.shared_ui import check_authentication, render_sidebar
from utils.docs_ui import render_metrics_glossary_page


st.set_page_config(page_title="Metrics Glossary - Tracey", page_icon="📚", layout="wide")

if not check_authentication():
    st.stop()

# Keep the sidebar consistent across the app (date/env/credentials + fetch button)
render_sidebar()

render_metrics_glossary_page()
