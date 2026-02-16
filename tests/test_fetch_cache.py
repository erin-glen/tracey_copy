"""Tests for fetch_traces_window disk cache — format migration and completeness flag."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from utils.langfuse_api import fetch_traces_window


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_response(status_code: int, json_data: Any = None) -> MagicMock:
    import requests
    resp = MagicMock(spec=requests.Response)
    resp.status_code = status_code
    resp.headers = {}
    resp.text = ""
    if json_data is not None:
        resp.json.return_value = json_data
    else:
        resp.json.side_effect = ValueError("No JSON")
    return resp


def _page_response(traces: list[dict[str, Any]]) -> MagicMock:
    return _make_response(200, json_data={"data": traces})


def _traces(n: int, offset: int = 0) -> list[dict[str, Any]]:
    return [{"id": f"t-{i + offset}", "timestamp": "2025-01-01T00:00:00Z"} for i in range(n)]


_BASE_KWARGS = dict(
    base_url="https://example.com",
    headers={"Authorization": "Basic dGVzdA=="},
    from_iso="2025-01-01T00:00:00Z",
    to_iso="2025-01-02T00:00:00Z",
    envs=None,
    page_size=50,
    page_limit=10,
    max_traces=1000,
    retry=2,
    backoff=0.01,
    http_timeout_s=5,
    use_disk_cache=True,
    max_page_retry_s=2.0,
)


# ---------------------------------------------------------------------------
# Cache write: completeness flag
# ---------------------------------------------------------------------------


class TestCacheWrite:
    @patch("utils.langfuse_api.time_mod.sleep")
    def test_complete_fetch_writes_complete_true(self, mock_sleep, tmp_path):
        """Successful fetch writes cache with complete=True."""
        responses = [_page_response(_traces(3)), _page_response([])]
        with patch("requests.Session") as MockSession:
            MockSession.return_value.get.side_effect = responses
            fetch_traces_window(**_BASE_KWARGS, cache_dir=str(tmp_path))

        # Find the cache file
        cache_files = list(tmp_path.glob("traces_*.json"))
        assert len(cache_files) == 1
        data = json.loads(cache_files[0].read_text())
        assert data["version"] == 2
        assert data["complete"] is True
        assert len(data["data"]) == 3

    @patch("utils.langfuse_api.time_mod.sleep")
    def test_error_fetch_writes_complete_false(self, mock_sleep, tmp_path):
        """Fetch that errors on page 1 writes cache with complete=False."""
        import requests as req_mod
        responses = [_make_response(500, json_data={"error": "boom"})] * 10
        with patch("requests.Session") as MockSession:
            MockSession.return_value.get.side_effect = responses
            fetch_traces_window(**_BASE_KWARGS, cache_dir=str(tmp_path))

        cache_files = list(tmp_path.glob("traces_*.json"))
        assert len(cache_files) == 1
        data = json.loads(cache_files[0].read_text())
        assert data["version"] == 2
        assert data["complete"] is False
        assert data["data"] == []

    @patch("utils.langfuse_api.time_mod.sleep")
    def test_max_traces_hit_writes_complete_false(self, mock_sleep, tmp_path):
        """Hitting max_traces → complete=False (data may be truncated)."""
        responses = [
            _page_response(_traces(50, offset=0)),
            _page_response(_traces(50, offset=50)),
        ]
        kwargs = {**_BASE_KWARGS, "max_traces": 30}
        with patch("requests.Session") as MockSession:
            MockSession.return_value.get.side_effect = responses
            debug: dict[str, Any] = {}
            result = fetch_traces_window(**kwargs, cache_dir=str(tmp_path), debug_out=debug)

        assert len(result) == 30
        cache_files = list(tmp_path.glob("traces_*.json"))
        assert len(cache_files) == 1
        data = json.loads(cache_files[0].read_text())
        assert data["complete"] is False


# ---------------------------------------------------------------------------
# Cache read: format migration
# ---------------------------------------------------------------------------


class TestCacheRead:
    def _write_cache_file(self, tmp_path, content: Any, **kwargs):
        """Write a cache file then call fetch_traces_window to test cache read."""
        # First, do a fetch to discover the cache path
        debug: dict[str, Any] = {}
        with patch("requests.Session") as MockSession:
            MockSession.return_value.get.side_effect = [_page_response([])]
            with patch("utils.langfuse_api.time_mod.sleep"):
                fetch_traces_window(**_BASE_KWARGS, cache_dir=str(tmp_path), debug_out=debug)
        cache_path = debug.get("cache_path")
        assert cache_path, "should have a cache_path"
        # Overwrite with our test content
        from pathlib import Path
        Path(cache_path).write_text(json.dumps(content), encoding="utf-8")
        return cache_path

    def test_reads_v2_complete_cache(self, tmp_path):
        """V2 cache with complete=True → returned without HTTP calls."""
        traces = _traces(5)
        cache_path = self._write_cache_file(tmp_path, {"version": 2, "complete": True, "data": traces})

        with patch("requests.Session") as MockSession:
            mock_get = MockSession.return_value.get
            with patch("utils.langfuse_api.time_mod.sleep"):
                debug: dict[str, Any] = {}
                result = fetch_traces_window(**_BASE_KWARGS, cache_dir=str(tmp_path), debug_out=debug)

        assert len(result) == 5
        assert debug.get("cache_hit") is True
        assert debug.get("cache_version") == 2
        mock_get.assert_not_called()

    def test_ignores_v2_incomplete_cache(self, tmp_path):
        """V2 cache with complete=False → ignored, re-fetches."""
        self._write_cache_file(tmp_path, {"version": 2, "complete": False, "data": _traces(3)})

        responses = [_page_response(_traces(7)), _page_response([])]
        with patch("requests.Session") as MockSession:
            MockSession.return_value.get.side_effect = responses
            with patch("utils.langfuse_api.time_mod.sleep"):
                result = fetch_traces_window(**_BASE_KWARGS, cache_dir=str(tmp_path))

        assert len(result) == 7  # fresh fetch, not the cached 3

    def test_reads_old_format_bare_list(self, tmp_path):
        """Old cache (bare list) → treated as complete, returned."""
        traces = _traces(4)
        self._write_cache_file(tmp_path, traces)

        with patch("requests.Session") as MockSession:
            mock_get = MockSession.return_value.get
            with patch("utils.langfuse_api.time_mod.sleep"):
                debug: dict[str, Any] = {}
                result = fetch_traces_window(**_BASE_KWARGS, cache_dir=str(tmp_path), debug_out=debug)

        assert len(result) == 4
        assert debug.get("cache_hit") is True
        assert debug.get("cache_version") == 1
        mock_get.assert_not_called()

    def test_handles_garbage_cache(self, tmp_path):
        """Corrupt/invalid cache → ignored, re-fetches."""
        self._write_cache_file(tmp_path, "not valid json at all {{{{")

        responses = [_page_response(_traces(2)), _page_response([])]
        with patch("requests.Session") as MockSession:
            MockSession.return_value.get.side_effect = responses
            with patch("utils.langfuse_api.time_mod.sleep"):
                result = fetch_traces_window(**_BASE_KWARGS, cache_dir=str(tmp_path))

        # The garbage file write above used json.dumps(string) which is valid JSON — 
        # so it'll parse as a string, which isn't a list or a version-2 dict → ignored.
        assert len(result) == 2
