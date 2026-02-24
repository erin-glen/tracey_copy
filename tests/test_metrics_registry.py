"""Tests for the metrics/page documentation registry.

These are governance tests: they are meant to fail fast when someone adds a KPI
or references a metric ID without documenting it.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec for {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ROOT = Path(__file__).resolve().parents[1]
metrics_mod = _load_module("metrics_registry", ROOT / "utils" / "metrics_registry.py")
validation_mod = _load_module("metrics_validation", ROOT / "utils" / "metrics_validation.py")

METRICS = metrics_mod.METRICS
PAGES = metrics_mod.PAGES
CLASSIFICATION_SPECS = getattr(metrics_mod, "CLASSIFICATION_SPECS", {})
validate_metrics_registry = validation_mod.validate_metrics_registry
validate_classification_specs = validation_mod.validate_classification_specs


def test_metrics_registry_is_valid() -> None:
    errors = validate_metrics_registry(METRICS, PAGES)
    assert not errors, "\n".join(["Metrics registry validation errors:"] + [f"- {e}" for e in errors])


def test_metric_spec_refs_resolve_to_known_specs() -> None:
    """Governance: any `spec_refs` attached to a metric must exist."""
    if not CLASSIFICATION_SPECS:
        pytest.skip("No CLASSIFICATION_SPECS defined")

    for metric_id, doc in METRICS.items():
        spec_refs = doc.get("spec_refs") or []
        if not spec_refs:
            continue
        assert isinstance(spec_refs, list), f"Metric {metric_id!r} spec_refs must be a list"
        for sid in spec_refs:
            sid = str(sid or "").strip()
            assert sid, f"Metric {metric_id!r} has an empty spec_refs entry"
            assert sid in CLASSIFICATION_SPECS, f"Metric {metric_id!r} references unknown spec_id: {sid!r}"


def test_classification_specs_are_valid() -> None:
    errors = validate_classification_specs(METRICS, CLASSIFICATION_SPECS)
    assert not errors, "\n".join(["Classification spec validation errors:"] + [f"- {e}" for e in errors])


def test_metric_spec_refs_are_bidirectionally_linked() -> None:
    """Governance: metric spec_refs and spec applies_to_metrics should stay in sync."""
    if not CLASSIFICATION_SPECS:
        pytest.skip("No CLASSIFICATION_SPECS defined")

    for metric_id, doc in METRICS.items():
        for sid in doc.get("spec_refs") or []:
            sid = str(sid or "").strip()
            if not sid:
                continue
            applies = CLASSIFICATION_SPECS.get(sid, {}).get("applies_to_metrics") or []
            assert metric_id in applies, (
                f"Metric {metric_id!r} references spec {sid!r} but is missing from "
                f"CLASSIFICATION_SPECS[{sid!r}]['applies_to_metrics']"
            )
