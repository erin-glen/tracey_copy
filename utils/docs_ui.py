"""Reusable documentation UI components.

These helpers power:
  - per-page "How to read this" panels
  - inline KPI info popovers
  - the Metrics Glossary page

All *content* lives in :mod:`utils.metrics_registry`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from utils.metrics_registry import CLASSIFICATION_SPECS, METRICS, PAGES


def _resolve_glossary_page_path() -> str:
    """Best-effort resolution for the Metrics Glossary page path.

    The repository includes hashed page filenames (e.g. `0_#L01f4da_Metrics_Glossary.py`).
    Hard-coding is brittle, so we look for the file by pattern.
    """

    try:
        repo_root = Path(__file__).resolve().parents[1]
        pages_dir = repo_root / "pages"
        candidates = sorted(pages_dir.glob("*Metrics_Glossary*.py"))
        if candidates:
            return str(candidates[0].relative_to(repo_root)).replace("\\", "/")
    except Exception:
        pass

    # Fallback to the most common name in this repo.
    return "pages/0_#L01f4da_Metrics_Glossary.py"


# This is used by `st.page_link` / `st.switch_page` so users can jump from
# inline help to the full glossary (with deep-linking via query params).
GLOSSARY_PAGE_PATH = _resolve_glossary_page_path()


def _supports_popover() -> bool:
    """Return True if the running Streamlit version supports st.popover()."""
    return hasattr(st, "popover")


def _supports_page_link() -> bool:
    """Return True if the running Streamlit version supports st.page_link()."""
    return hasattr(st, "page_link")


def _supports_switch_page() -> bool:
    """Return True if the running Streamlit version supports st.switch_page()."""
    return hasattr(st, "switch_page")


def _get_query_param(key: str) -> str | None:
    """Read a query param in a Streamlit-version-tolerant way."""
    k = str(key or "").strip()
    if not k:
        return None

    # Modern Streamlit (>=1.30) supports st.query_params (dict-like property).
    if hasattr(st, "query_params"):
        try:
            v = st.query_params.get(k)
            if v is None:
                return None
            return str(v)
        except Exception:
            return None

    # Legacy fallback.
    if hasattr(st, "experimental_get_query_params"):
        try:
            qp = st.experimental_get_query_params()  # type: ignore[attr-defined]
            vv = qp.get(k)
            if not vv:
                return None
            if isinstance(vv, (list, tuple)):
                return str(vv[0]) if vv else None
            return str(vv)
        except Exception:
            return None

    return None


def glossary_link(
    metric_id: str | None = None,
    *,
    label: str = "📚 Open Metrics Glossary",
    key: str | None = None,
) -> None:
    """Render a link/button to the Metrics Glossary (optionally deep-linked to a metric).

    Uses `st.page_link` when available; falls back to `st.switch_page`.
    """

    qp = None
    mid = str(metric_id or "").strip()
    if mid:
        qp = {"metric": mid}

    if _supports_page_link():
        try:
            st.page_link(GLOSSARY_PAGE_PATH, label=label, query_params=qp, width="content")
            return
        except TypeError:
            st.page_link(GLOSSARY_PAGE_PATH, label=label, query_params=qp)
            return

    if _supports_switch_page():
        if st.button(label, key=key or f"glossary_link_{mid}"):
            st.switch_page(GLOSSARY_PAGE_PATH, query_params=qp)
        return

    st.caption("Navigate to the 📚 Metrics Glossary page for full definitions.")


def get_metric_doc(metric_id: str) -> dict[str, Any] | None:
    """Return the documentation dict for a metric_id, if present."""
    return METRICS.get(str(metric_id or "").strip())


def get_classification_spec(spec_id: str) -> dict[str, Any] | None:
    """Return a classification spec doc, if present."""
    return CLASSIFICATION_SPECS.get(str(spec_id or "").strip())


def render_metric_doc(metric_id: str, *, show_id: bool = True) -> None:
    """Render the full documentation for a metric inside the current container."""
    metric_id = str(metric_id or "").strip()
    doc = get_metric_doc(metric_id)
    if not doc:
        st.caption("No documentation available for this metric yet.")
        if metric_id:
            st.code(metric_id)
        return

    st.markdown(f"**{doc.get('name', metric_id)}**")
    if show_id:
        st.caption(f"ID: `{metric_id}`")

    definition = str(doc.get("definition") or "").strip()
    if definition:
        st.markdown(definition)

    # Data contract (denominator semantics)
    data_contract_rows: list[tuple[str, str]] = []
    for label, key in [
        ("Population", "population"),
        ("Numerator", "numerator"),
        ("Denominator", "denominator"),
        ("Unit", "unit"),
        ("Method", "method"),
    ]:
        v = str(doc.get(key) or "").strip()
        if v:
            data_contract_rows.append((label, v))

    if data_contract_rows:
        with st.expander("Data contract", expanded=False):
            for k, v in data_contract_rows:
                st.markdown(f"**{k}:** {v}")

    formula = str(doc.get("formula") or "").strip()
    if formula:
        st.markdown("**How it's computed**")
        st.markdown(formula)

    provenance = str(doc.get("provenance") or "").strip()
    if provenance:
        st.markdown("**Where it comes from**")
        st.markdown(provenance)

    # Code references
    code_refs = doc.get("code_refs") or []
    if isinstance(code_refs, list) and code_refs:
        with st.expander("Code references", expanded=False):
            for ref in code_refs:
                if str(ref).strip():
                    st.markdown(f"- `{ref}`")

    # Classification specs (derived labels)
    spec_refs = doc.get("spec_refs") or []
    if isinstance(spec_refs, list) and spec_refs:
        with st.expander("Classification spec", expanded=False):
            for sid in spec_refs:
                sid = str(sid or "").strip()
                if not sid:
                    continue
                sdoc = get_classification_spec(sid)
                if not sdoc:
                    st.markdown(f"- `{sid}`")
                    continue

                title = str(sdoc.get("name") or sid).strip()
                with st.expander(title, expanded=False):
                    md = str(sdoc.get("markdown") or "").strip()
                    if md:
                        st.markdown(md)

                    s_code_refs = sdoc.get("code_refs") or []
                    if isinstance(s_code_refs, list) and s_code_refs:
                        st.markdown("**Code refs**")
                        for ref in s_code_refs:
                            if str(ref).strip():
                                st.markdown(f"- `{ref}`")

                    test_refs = sdoc.get("test_refs") or []
                    if isinstance(test_refs, list) and test_refs:
                        st.markdown("**Executable spec tests**")
                        for tref in test_refs:
                            if str(tref).strip():
                                st.markdown(f"- `{tref}`")

    caveats = doc.get("caveats") or []
    if isinstance(caveats, list) and caveats:
        st.markdown("**Caveats**")
        for c in caveats:
            if str(c).strip():
                st.markdown(f"- {c}")

    used_in = doc.get("used_in") or []
    if isinstance(used_in, list) and used_in:
        st.caption("Used in: " + ", ".join(str(x) for x in used_in if str(x).strip()))


def _render_help_popover(
    *,
    label: str = "ℹ️",
    metric_id: str | None = None,
    help_md: str | None = None,
    key: str | None = None,
) -> None:
    """Render a help affordance (popover if supported; else expander)."""
    metric_id = str(metric_id or "").strip() or None
    help_md = str(help_md or "").strip() or None

    if not metric_id and not help_md:
        return

    if _supports_popover():
        # NOTE: Streamlit popover supports a key parameter in modern versions.
        # To stay compatible across versions, only pass key if provided.
        if key is None:
            pop = st.popover(label)
        else:
            try:
                pop = st.popover(label, key=key)
            except TypeError:
                pop = st.popover(label)
        with pop:
            if metric_id:
                render_metric_doc(metric_id)
                st.divider()
                glossary_link(metric_id, label="📚 Open in glossary")
            if help_md:
                st.markdown(help_md)
    else:
        # Fallback for older Streamlit.
        with st.expander(label, expanded=False):
            if metric_id:
                render_metric_doc(metric_id)
                st.divider()
                glossary_link(metric_id, label="📚 Open in glossary")
            if help_md:
                st.markdown(help_md)


def metric_with_help(
    label: str,
    value: Any,
    *,
    metric_id: str | None = None,
    delta: Any | None = None,
    delta_color: str = "normal",
    delta_arrow: str | None = None,
    help_md: str | None = None,
    key: str | None = None,
) -> None:
    """Render a metric with a small inline help popover.

    This is a drop-in alternative to st.metric() when you want a glossary-backed
    tooltip right next to the KPI.
    """
    # Layout: metric on the left, small help button on the right.
    left, right = st.columns([0.86, 0.14], gap="small")
    with left:
        try:
            kwargs: dict[str, Any] = {"delta": delta}
            # delta_color introduced in newer Streamlit; safe to try.
            kwargs["delta_color"] = delta_color
            if delta_arrow is not None:
                kwargs["delta_arrow"] = delta_arrow
            st.metric(label, value, **kwargs)
        except TypeError:
            # Older Streamlit versions don't support delta_color.
            st.metric(label, value, delta=delta)

    with right:
        _render_help_popover(
            label="ℹ️",
            metric_id=metric_id,
            help_md=help_md,
            key=key or (f"help_{metric_id}_{label}" if metric_id else None),
        )


def render_page_help(page_id: str, *, expanded: bool = False) -> None:
    """Render a standardized per-page help expander."""
    page_id = str(page_id or "").strip().lower()
    doc = PAGES.get(page_id)
    if not doc:
        return

    title = str(doc.get("title") or "How to read this page").strip()

    with st.expander("How to read this page", expanded=expanded):
        st.markdown(f"#### {title}")

        what = doc.get("what") or []
        if isinstance(what, list) and what:
            st.markdown("**What this page is for**")
            for line in what:
                if str(line).strip():
                    st.markdown(f"- {line}")

        data = doc.get("data") or []
        if isinstance(data, list) and data:
            st.markdown("**What data it uses**")
            for line in data:
                if str(line).strip():
                    st.markdown(f"- {line}")

        key_metrics = doc.get("key_metrics") or []
        if isinstance(key_metrics, list) and key_metrics:
            st.markdown("**Key metrics on this page**")
            for mid in key_metrics:
                mid = str(mid or "").strip()
                if not mid:
                    continue
                mdoc = get_metric_doc(mid)
                if mdoc:
                    st.markdown(f"- **{mdoc.get('name', mid)}** (`{mid}`): {mdoc.get('definition', '')}")
                else:
                    st.markdown(f"- `{mid}`")

        pitfalls = doc.get("pitfalls") or []
        if isinstance(pitfalls, list) and pitfalls:
            st.markdown("**Common pitfalls**")
            for line in pitfalls:
                if str(line).strip():
                    st.markdown(f"- {line}")

        st.caption("Tip: Use the Metrics Glossary for the full definitions + provenance.")
        glossary_link(label="📚 Open Metrics Glossary", key=f"page_help_glossary_{page_id}")


def render_metrics_glossary_page() -> None:
    """Render the Metrics Glossary page."""
    st.title("📚 Metrics Glossary")
    st.caption("Definitions, formulas, provenance, and caveats for Tracey metrics.")

    all_rows: list[dict[str, Any]] = []
    for metric_id, doc in METRICS.items():
        all_rows.append(
            {
                "metric_id": metric_id,
                "name": doc.get("name", metric_id),
                "category": doc.get("category", ""),
                "definition": doc.get("definition", ""),
                "used_in": ", ".join(doc.get("used_in") or []),
            }
        )

    df = pd.DataFrame(all_rows)
    if df.empty:
        st.warning("No metrics have been registered yet.")
        return

    df = df.sort_values(["category", "name"], na_position="last").reset_index(drop=True)

    # Classification specs (derived-label documentation)
    if CLASSIFICATION_SPECS:
        st.markdown("### Classification specs")
        st.caption(
            "Deterministic label definitions used by one or more KPIs (e.g., Content KPIs completion_state). "
            "These are paired with spec-by-example tests in `tests/`."
        )
        for sid, sdoc in CLASSIFICATION_SPECS.items():
            title = str(sdoc.get("name") or sid).strip()
            with st.expander(title, expanded=False):
                md = str(sdoc.get("markdown") or "").strip()
                if md:
                    st.markdown(md)

                applies = sdoc.get("applies_to_metrics") or []
                if isinstance(applies, list) and applies:
                    st.markdown("**Applies to metrics**")
                    for mid in applies:
                        mid = str(mid or "").strip()
                        if not mid:
                            continue
                        mdoc = METRICS.get(mid)
                        label = mdoc.get("name", mid) if isinstance(mdoc, dict) else mid
                        st.markdown(f"- **{label}** (`{mid}`)")

    # Filters
    c1, c2 = st.columns([2, 1])
    with c1:
        q = st.text_input("Search", value="", placeholder="e.g. success, citations, latency")
    with c2:
        categories = sorted(x for x in df["category"].dropna().astype(str).unique() if x.strip())
        selected_categories = st.multiselect("Category", options=categories, default=[])

    view = df.copy()
    if selected_categories:
        view = view[view["category"].isin(selected_categories)]
    if q.strip():
        ql = q.strip().lower()
        view = view[
            view.apply(
                lambda r: ql in str(r.get("metric_id") or "").lower()
                or ql in str(r.get("name") or "").lower()
                or ql in str(r.get("definition") or "").lower(),
                axis=1,
            )
        ]

    st.markdown("### All metrics")
    st.dataframe(
        view[["metric_id", "name", "category", "definition", "used_in"]],
        use_container_width=True,
        hide_index=True,
    )

    # Detail panel
    st.markdown("### Metric details")
    options = view["metric_id"].tolist()
    if not options:
        st.info("No metrics match the current filters.")
        return

    deep_link_metric = _get_query_param("metric")
    if deep_link_metric:
        deep_link_metric = str(deep_link_metric).strip()
    if deep_link_metric and deep_link_metric in METRICS and deep_link_metric not in options:
        # Ensure the metric is selectable even if filters hide it.
        options = [deep_link_metric] + options

    index = 0
    if deep_link_metric and deep_link_metric in options:
        index = max(0, options.index(deep_link_metric))

    selected = st.selectbox(
        "Select a metric",
        options=options,
        index=index,
        format_func=lambda mid: f"{METRICS.get(mid, {}).get('name', mid)}  ·  {mid}",
    )
    with st.container(border=True):
        render_metric_doc(selected, show_id=True)

    # Exports
    st.markdown("### Export")

    export_metric_ids = view["metric_id"].tolist()
    export_metrics = {mid: METRICS[mid] for mid in export_metric_ids if mid in METRICS}
    export_payload = {
        "metrics": export_metrics,
        "pages": PAGES,
        "classification_specs": CLASSIFICATION_SPECS,
    }

    full_json_bytes = json.dumps(export_payload, indent=2, sort_keys=True).encode("utf-8")
    st.download_button(
        "Download glossary registry (JSON)",
        data=full_json_bytes,
        file_name="tracey_metrics_registry.json",
        mime="application/json",
        help="Includes full metric definitions for the currently filtered metric set plus all page docs.",
    )

    # Flat CSV export for easy review in spreadsheets
    flat_rows: list[dict[str, Any]] = []
    for mid in export_metric_ids:
        doc = METRICS.get(mid) or {}
        flat_rows.append(
            {
                "metric_id": mid,
                "name": doc.get("name", mid),
                "category": doc.get("category", ""),
                "definition": doc.get("definition", ""),
                "formula": doc.get("formula", ""),
                "provenance": doc.get("provenance", ""),
                "population": doc.get("population", ""),
                "numerator": doc.get("numerator", ""),
                "denominator": doc.get("denominator", ""),
                "unit": doc.get("unit", ""),
                "method": doc.get("method", ""),
                "used_in": ", ".join(doc.get("used_in") or []),
                "code_refs": ", ".join(doc.get("code_refs") or []),
                "caveats": " | ".join(doc.get("caveats") or []),
                "spec_refs": ", ".join(doc.get("spec_refs") or []),
            }
        )

    flat_df = pd.DataFrame(flat_rows)
    csv_bytes = flat_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download glossary registry (CSV)",
        data=csv_bytes,
        file_name="tracey_metrics_registry.csv",
        mime="text/csv",
        help="Flat, spreadsheet-friendly export of the currently filtered metric set.",
    )
