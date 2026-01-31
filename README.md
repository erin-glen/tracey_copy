# GNW Trace Evals (Tracey)

## Run locally

### Prereqs

- Python `>=3.11`
- Recommended: [`uv`](https://github.com/astral-sh/uv)

### 1) Set environment variables

Create a `.env` file in the repo root:

```bash
LANGFUSE_PUBLIC_KEY="..."
LANGFUSE_SECRET_KEY="..."
LANGFUSE_BASE_URL="..."

# Optional (only needed for Gemini-powered features)
GEMINI_API_KEY="..."  # or GOOGLE_API_KEY
```

### 2) Install dependencies

#### Option A: uv (recommended)

```bash
uv sync
```

#### Option B: pip

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e .
```

### 3) Run the Streamlit app

```bash
uv run streamlit run streamlit_app.py
```

Then open the URL Streamlit prints (usually `http://localhost:8501`).

## Notes

- The app fetches traces **once** from the sidebar, then reuses the same dataset across tabs.
- Human eval exports are always available via the **Download CSV** button.

## Product Development Mining

The **🧠 Product intelligence** tab contains **Product Development Mining**, split into three sub-tabs:

- **Evidence Mining**: search for traces that support a hypothesis (LLM-scored relevance).
- **Tagging**: LLM-as-judge tagging for prompt topics/flavours and other criteria.
- **Gap Analysis**: generate a markdown report on user jobs, coverage, and gaps.

### LLM settings

In that tab, open **⚙️ LLM Settings** to configure:

- Gemini model
- Optional batching (**Batch traces per Gemini request**, batch size, max chars per trace)

### Editable prompts

Each sub-tab exposes an **📝 Edit system prompt** expander so you can inspect and tweak the prompts used for Gemini.

## (Optional) Run the Marimo notebook

```bash
uv run marimo edit gnw_trace_pull.py