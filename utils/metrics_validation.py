"""Validation helpers for the metrics/page documentation registry.

This module is intentionally small and dependency-free so it can be used in unit
tests and CI without importing Streamlit.

The intent is to prevent silent drift:
  - KPI cards are added without glossary entries
  - glossary entries are missing required metadata (population, units, etc.)
  - page docs reference non-existent metric IDs
"""

from __future__ import annotations

import re
from typing import Any


_SNAKE_CASE_RE = re.compile(r"^[a-z][a-z0-9_]*$")


REQUIRED_METRIC_FIELDS: tuple[str, ...] = (
    "name",
    "category",
    "definition",
    "formula",
    "provenance",
    "caveats",
    "used_in",
    # Governance fields
    "population",
    "numerator",
    "denominator",
    "unit",
    "method",
    "code_refs",
)




REQUIRED_SPEC_FIELDS: tuple[str, ...] = (
    "name",
    "applies_to_metrics",
    "markdown",
    "code_refs",
    "test_refs",
)
def validate_metrics_registry(
    metrics: dict[str, dict[str, Any]],
    pages: dict[str, dict[str, Any]],
) -> list[str]:
    """Validate the registry and return a list of human-readable errors."""
    errors: list[str] = []

    # -----------------------------
    # Metric schema checks
    # -----------------------------
    for metric_id, doc in (metrics or {}).items():
        mid = str(metric_id or "").strip()
        if not mid:
            errors.append("Metric ID is empty")
            continue
        if not _SNAKE_CASE_RE.match(mid):
            errors.append(f"Metric ID must be snake_case: {mid!r}")

        if not isinstance(doc, dict):
            errors.append(f"Metric {mid!r} doc must be a dict")
            continue

        for k in REQUIRED_METRIC_FIELDS:
            if k not in doc:
                errors.append(f"Metric {mid!r} missing required field: {k}")
                continue
            v = doc.get(k)
            if v is None:
                errors.append(f"Metric {mid!r} field {k!r} must not be None")
                continue
            # Enforce non-empty strings for the key governance fields.
            if k in {"population", "numerator", "denominator", "unit", "method"}:
                if not isinstance(v, str) or not v.strip():
                    errors.append(f"Metric {mid!r} field {k!r} must be a non-empty string")
            if k == "code_refs":
                if not isinstance(v, list) or not v:
                    errors.append(f"Metric {mid!r} field 'code_refs' must be a non-empty list")

    # -----------------------------
    # Page docs checks
    # -----------------------------
    for page_id, pdoc in (pages or {}).items():
        pid = str(page_id or "").strip().lower()
        if not pid:
            errors.append("Page ID is empty")
            continue
        if not isinstance(pdoc, dict):
            errors.append(f"Page {pid!r} doc must be a dict")
            continue

        key_metrics = pdoc.get("key_metrics") or []
        if not isinstance(key_metrics, list):
            errors.append(f"Page {pid!r} key_metrics must be a list")
            continue
        for mid in key_metrics:
            mid = str(mid or "").strip()
            if not mid:
                errors.append(f"Page {pid!r} contains an empty metric ID in key_metrics")
                continue
            if mid not in (metrics or {}):
                errors.append(f"Page {pid!r} references unknown metric_id: {mid!r}")

    return errors



def validate_classification_specs(
    metrics: dict[str, dict[str, Any]],
    classification_specs: dict[str, dict[str, Any]],
) -> list[str]:
    """Validate classification-spec docs and linkage to metrics."""
    errors: list[str] = []

    for spec_id, sdoc in (classification_specs or {}).items():
        sid = str(spec_id or "").strip()
        if not sid:
            errors.append("Classification spec ID is empty")
            continue
        if not isinstance(sdoc, dict):
            errors.append(f"Classification spec {sid!r} doc must be a dict")
            continue

        for k in REQUIRED_SPEC_FIELDS:
            if k not in sdoc:
                errors.append(f"Classification spec {sid!r} missing required field: {k}")

        applies = sdoc.get("applies_to_metrics") or []
        if not isinstance(applies, list) or not applies:
            errors.append(f"Classification spec {sid!r} field 'applies_to_metrics' must be a non-empty list")
            continue

        for mid in applies:
            mid = str(mid or "").strip()
            if not mid:
                errors.append(f"Classification spec {sid!r} has an empty applies_to_metrics entry")
                continue
            if mid not in (metrics or {}):
                errors.append(f"Classification spec {sid!r} references unknown metric_id: {mid!r}")
                continue
            spec_refs = metrics.get(mid, {}).get("spec_refs") or []
            if sid not in spec_refs:
                errors.append(f"Metric {mid!r} must include spec_refs entry for classification spec {sid!r}")

        for field in ("code_refs", "test_refs"):
            vals = sdoc.get(field) or []
            if not isinstance(vals, list) or not vals:
                errors.append(f"Classification spec {sid!r} field {field!r} must be a non-empty list")

        markdown = str(sdoc.get("markdown") or "").strip()
        if not markdown:
            errors.append(f"Classification spec {sid!r} field 'markdown' must be a non-empty string")

    return errors
