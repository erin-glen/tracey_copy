# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "langfuse==3.12.0",
#     "marimo",
# ]
# ///

import marimo

__generated_with = "0.19.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    from langfuse import get_client
    return (get_client,)


@app.cell
def _(get_client):
    langfuse = get_client()
    return (langfuse,)


@app.cell
def _(langfuse):
    all_sessions = langfuse.api.sessions.list(limit=30)
    return (all_sessions,)


@app.cell
def _(langfuse):
    def get_session_datasets(session_id):
        session = langfuse.api.sessions.get(session_id=session_id)
        datasets = set()
        for trace in session.traces:
            try:
                output = trace.output
                if output and isinstance(output, dict):
                    dataset_info = output.get("dataset", {})
                    if dataset_info and isinstance(dataset_info, dict):
                        dataset_name = dataset_info.get("dataset_name")
                        if dataset_name:
                            datasets.add(dataset_name)
            except (AttributeError, TypeError, KeyError):
                continue
        return ", ".join(sorted(datasets))
    return (get_session_datasets,)


@app.cell
def _(all_sessions, get_session_datasets):
    BASE_URL = "https://www.globalnaturewatch.org/app/threads"

    csv_data = []
    for session in all_sessions.data[:30]:
        url = f"{BASE_URL}/{session.id}"
        datasets = get_session_datasets(session.id)
        csv_data.append({"url": url, "datasets": datasets})
    return (csv_data,)


@app.cell
def _(csv_data, mo):
    import csv

    # Write to CSV file
    with open("gnw_sessions.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["url", "datasets"])
        writer.writeheader()
        writer.writerows(csv_data)

    # Display in notebook
    mo.ui.table(csv_data)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
