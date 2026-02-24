"""Tracey - GNW Trace Analysis Tool.

This is the home page for the multipage Streamlit app.
Navigate to other pages using the sidebar.
"""

import streamlit as st

from utils.shared_ui import check_authentication, render_sidebar


st.set_page_config(page_title="Tracey.", page_icon="💬", layout="wide")

if not check_authentication():
    st.stop()

config = render_sidebar()

# Home page content
st.title("💬🧠📎 Tracey. `v0.1`")
st.markdown(
    """
**Tracey** is a trace analysis and human evaluation tool for Global Nature Watch.
Quickly pull and explore traces from Langfuse. _Ta, Trace!_

### 🛠️ What's in the toolkit?

| Tool | Description |
|------|-------------|
| 📊 **Analytics** | Overview charts, metrics, and reports |
| ✅ **Human Eval** | Sample and rate traces for quality evaluation |
| 📈 **Eval Insights** | A simple dashboard of human evaluation results |
| 🧠 **Product Intelligence** | AI-powered insights and pattern discovery |
| 🔎 **Trace Explorer** | Browse and filter individual traces |
| 🔗 **Conversation Browser** | View full conversation threads |
| 🧱 **Content KPIs** | Monitor content quality outcomes and KPI trends |
| 🧵 **Thread QA** | Audit thread-level quality checks and outcomes |
| 📦 **QA Samples** | Curated QA sample sets for quick review |
| 🧩 **CodeAct QA** | QA workflows for CodeAct traces and outputs |
| 📚 **Metrics Glossary** | Definitions and interpretation guidance for metrics |

### Current Session

"""
)

# Show current session info
traces = st.session_state.get("stats_traces", [])
if traces:
    st.success(f"✅ **{len(traces):,} traces loaded** - Navigate to a page to explore them.")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Traces", f"{len(traces):,}")
    with col2:
        env = config.get("environment", "production")
        st.metric("Environment", env)
    with col3:
        start = config.get("start_date")
        end = config.get("end_date")
        if start and end:
            st.metric("Date Range", f"{start} → {end}")
else:
    st.info("👈 Use the sidebar to fetch traces, then navigate to a page to explore them.")
