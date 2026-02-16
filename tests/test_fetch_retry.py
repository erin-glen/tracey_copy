"""Tests for fetch_traces_window retry logic, 429 handling, and rate limiter integration.

All tests mock HTTP responses — no real API calls.
"""

from __future__ import annotations

import json
import time
from typing import Any
from unittest.mock import MagicMock, patch, PropertyMock

import pytest
import requests

from utils.langfuse_api import fetch_traces_window
from utils.fetch_throttle import TokenBucket, SharedBudget


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_response(status_code: int, json_data: Any = None, headers: dict | None = None, text: str = "") -> MagicMock:
    """Build a mock requests.Response."""
    resp = MagicMock(spec=requests.Response)
    resp.status_code = status_code
    resp.headers = headers or {}
    resp.text = text
    if json_data is not None:
        resp.json.return_value = json_data
    else:
        resp.json.side_effect = ValueError("No JSON")
    return resp


def _page_response(traces: list[dict[str, Any]]) -> MagicMock:
    """Build a 200 response wrapping a page of traces."""
    return _make_response(200, json_data={"data": traces})


def _traces(n: int, offset: int = 0) -> list[dict[str, Any]]:
    """Generate n synthetic trace dicts."""
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
    retry=4,
    backoff=0.01,  # tiny for fast tests
    http_timeout_s=5,
    use_disk_cache=False,
    max_page_retry_s=2.0,  # tight budget for tests
)


# ---------------------------------------------------------------------------
# 5xx retry with exponential backoff
# ---------------------------------------------------------------------------


class TestRetry5xx:
    @patch("utils.langfuse_api.time_mod.sleep")
    def test_retries_on_5xx_then_succeeds(self, mock_sleep):
        """Two 500s followed by a 200 → returns data."""
        responses = [
            _make_response(500, text="Internal Server Error"),
            _make_response(500, text="Internal Server Error"),
            _page_response(_traces(3)),
            _page_response([]),  # end of data
        ]
        with patch("requests.Session") as MockSession:
            MockSession.return_value.get.side_effect = responses
            result = fetch_traces_window(**_BASE_KWARGS)

        assert len(result) == 3
        # Should have slept twice (exponential backoff)
        backoff_calls = [c for c in mock_sleep.call_args_list if c[0][0] > 0]
        assert len(backoff_calls) >= 2

    @patch("utils.langfuse_api.time_mod.sleep")
    def test_5xx_exponential_backoff_delays(self, mock_sleep):
        """Backoff delays increase exponentially."""
        responses = [
            _make_response(500, text="err"),
            _make_response(500, text="err"),
            _make_response(500, text="err"),
            _page_response(_traces(1)),
            _page_response([]),
        ]
        with patch("requests.Session") as MockSession:
            MockSession.return_value.get.side_effect = responses
            fetch_traces_window(**_BASE_KWARGS)

        # Extract the positive sleep calls (backoff, not inter-page)
        sleeps = [c[0][0] for c in mock_sleep.call_args_list if c[0][0] > 0.005]
        assert len(sleeps) >= 2
        # Each sleep should be larger than the previous (exponential)
        for i in range(1, len(sleeps)):
            assert sleeps[i] >= sleeps[i - 1] * 0.8  # allow jitter tolerance

    @patch("utils.langfuse_api.time_mod.sleep")
    def test_5xx_exhausts_retries(self, mock_sleep):
        """All retries fail → returns partial results."""
        responses = [_make_response(500, text="err")] * 10
        with patch("requests.Session") as MockSession:
            MockSession.return_value.get.side_effect = responses
            debug: dict[str, Any] = {}
            result = fetch_traces_window(**_BASE_KWARGS, debug_out=debug)

        assert result == []
        assert "stopped_early_reason" in debug
        reason = debug["stopped_early_reason"]
        assert "500" in reason or "5xx" in reason or "retry budget" in reason


# ---------------------------------------------------------------------------
# 429 rate limit handling
# ---------------------------------------------------------------------------


class TestRetry429:
    @patch("utils.langfuse_api.time_mod.sleep")
    def test_429_with_retry_after_header(self, mock_sleep):
        """429 + Retry-After → sleeps that amount, retries, succeeds."""
        responses = [
            _make_response(429, headers={"Retry-After": "0.1"}),
            _page_response(_traces(2)),
            _page_response([]),
        ]
        with patch("requests.Session") as MockSession:
            MockSession.return_value.get.side_effect = responses
            result = fetch_traces_window(**_BASE_KWARGS)

        assert len(result) == 2
        # Should have slept ~0.1s for the 429
        sleep_args = [c[0][0] for c in mock_sleep.call_args_list]
        assert any(0.05 <= s <= 0.2 for s in sleep_args)

    @patch("utils.langfuse_api.time_mod.sleep")
    def test_429_without_retry_after(self, mock_sleep):
        """429 without header → uses exponential fallback."""
        responses = [
            _make_response(429),
            _page_response(_traces(1)),
            _page_response([]),
        ]
        with patch("requests.Session") as MockSession:
            MockSession.return_value.get.side_effect = responses
            result = fetch_traces_window(**_BASE_KWARGS)

        assert len(result) == 1

    @patch("utils.langfuse_api.time_mod.sleep")
    def test_429_does_not_count_against_retry_limit(self, mock_sleep):
        """Multiple 429s don't exhaust the retry counter (bounded by time)."""
        # 3 × 429 then success — with retry=2 this should still succeed
        # because 429s don't increment `attempts`
        responses = [
            _make_response(429, headers={"Retry-After": "0.01"}),
            _make_response(429, headers={"Retry-After": "0.01"}),
            _make_response(429, headers={"Retry-After": "0.01"}),
            _page_response(_traces(1)),
            _page_response([]),
        ]
        kwargs = {**_BASE_KWARGS, "retry": 2}
        with patch("requests.Session") as MockSession:
            MockSession.return_value.get.side_effect = responses
            result = fetch_traces_window(**kwargs)

        assert len(result) == 1

    @patch("utils.langfuse_api.time_mod.sleep")
    def test_429_propagates_pause_to_rate_limiter(self, mock_sleep):
        """429 calls pause_until() on the shared rate limiter."""
        bucket = MagicMock(spec=TokenBucket)
        bucket.acquire.return_value = True

        responses = [
            _make_response(429, headers={"Retry-After": "5"}),
            _page_response(_traces(1)),
            _page_response([]),
        ]
        with patch("requests.Session") as MockSession:
            MockSession.return_value.get.side_effect = responses
            fetch_traces_window(**_BASE_KWARGS, rate_limiter=bucket)

        bucket.pause_until.assert_called_once()
        pause_arg = bucket.pause_until.call_args[0][0]
        assert pause_arg > time.monotonic() - 10  # sanity check it's a real monotonic value


# ---------------------------------------------------------------------------
# Network errors (ConnectionError, Timeout)
# ---------------------------------------------------------------------------


class TestNetworkErrors:
    @patch("utils.langfuse_api.time_mod.sleep")
    def test_connection_error_retried(self, mock_sleep):
        """ConnectionError → retried, then succeeds."""
        with patch("requests.Session") as MockSession:
            MockSession.return_value.get.side_effect = [
                requests.ConnectionError("refused"),
                _page_response(_traces(2)),
                _page_response([]),
            ]
            result = fetch_traces_window(**_BASE_KWARGS)

        assert len(result) == 2

    @patch("utils.langfuse_api.time_mod.sleep")
    def test_timeout_retried(self, mock_sleep):
        """Timeout → retried, then succeeds."""
        with patch("requests.Session") as MockSession:
            MockSession.return_value.get.side_effect = [
                requests.Timeout("timed out"),
                _page_response(_traces(1)),
                _page_response([]),
            ]
            result = fetch_traces_window(**_BASE_KWARGS)

        assert len(result) == 1

    @patch("utils.langfuse_api.time_mod.sleep")
    def test_network_error_exhausts_retries(self, mock_sleep):
        """Persistent network errors → gives up after retry limit."""
        with patch("requests.Session") as MockSession:
            MockSession.return_value.get.side_effect = requests.ConnectionError("down")
            debug: dict[str, Any] = {}
            result = fetch_traces_window(**_BASE_KWARGS, debug_out=debug)

        assert result == []
        assert "Network error" in debug.get("stopped_early_reason", "")


# ---------------------------------------------------------------------------
# Per-page retry time budget
# ---------------------------------------------------------------------------


class TestPageRetryBudget:
    def test_page_retry_budget_caps_total_time(self):
        """With a tight time budget, retries don't hang."""
        def slow_500(*args, **kwargs):
            time.sleep(0.1)
            return _make_response(500, text="err")

        with patch("requests.Session") as MockSession:
            MockSession.return_value.get.side_effect = slow_500
            t0 = time.monotonic()
            result = fetch_traces_window(**{**_BASE_KWARGS, "max_page_retry_s": 0.5})
            elapsed = time.monotonic() - t0

        assert result == []
        assert elapsed < 3.0  # should finish well under 3s


# ---------------------------------------------------------------------------
# Memory error page-size halving
# ---------------------------------------------------------------------------


class TestMemoryError:
    @patch("utils.langfuse_api.time_mod.sleep")
    def test_memory_error_halves_limit(self, mock_sleep):
        """ClickHouse memory error → page size halved → retry succeeds."""
        responses = [
            _make_response(500, text="Memory limit exceeded for query"),
            _page_response(_traces(2)),
            _page_response([]),
        ]
        with patch("requests.Session") as MockSession:
            mock_get = MockSession.return_value.get
            mock_get.side_effect = responses
            debug: dict[str, Any] = {}
            result = fetch_traces_window(**_BASE_KWARGS, debug_out=debug)

        assert len(result) == 2
        # Second call should have halved limit
        second_call_params = mock_get.call_args_list[1][1].get("params", {})
        assert second_call_params.get("limit", 50) == 25


# ---------------------------------------------------------------------------
# Rate limiter integration
# ---------------------------------------------------------------------------


class TestRateLimiterIntegration:
    @patch("utils.langfuse_api.time_mod.sleep")
    def test_rate_limiter_called_per_request(self, mock_sleep):
        """acquire() is called before each HTTP request."""
        bucket = MagicMock(spec=TokenBucket)
        bucket.acquire.return_value = True

        # page_size=5 so limit=5, each full page triggers next page
        responses = [
            _page_response(_traces(5, offset=0)),
            _page_response(_traces(5, offset=5)),
            _page_response(_traces(2, offset=10)),  # partial → stops
        ]
        with patch("requests.Session") as MockSession:
            MockSession.return_value.get.side_effect = responses
            fetch_traces_window(**{**_BASE_KWARGS, "page_size": 5}, rate_limiter=bucket)

        assert bucket.acquire.call_count == 3  # one per page

    @patch("utils.langfuse_api.time_mod.sleep")
    def test_rate_limiter_timeout_aborts(self, mock_sleep):
        """If acquire() times out, fetch stops gracefully."""
        bucket = MagicMock(spec=TokenBucket)
        bucket.acquire.return_value = False

        with patch("requests.Session") as MockSession:
            MockSession.return_value.get.return_value = _page_response(_traces(5))
            debug: dict[str, Any] = {}
            result = fetch_traces_window(**_BASE_KWARGS, rate_limiter=bucket, debug_out=debug)

        assert result == []
        assert "Rate limiter" in debug.get("stopped_early_reason", "")


# ---------------------------------------------------------------------------
# SharedBudget integration
# ---------------------------------------------------------------------------


class TestBudgetIntegration:
    @patch("utils.langfuse_api.time_mod.sleep")
    def test_budget_limits_total_traces(self, mock_sleep):
        """SharedBudget stops fetch when global limit reached."""
        budget = SharedBudget(total=7)

        # page_size=5 so limit=5, full pages trigger next page
        responses = [
            _page_response(_traces(5, offset=0)),
            _page_response(_traces(5, offset=5)),
            _page_response(_traces(5, offset=10)),
        ]
        with patch("requests.Session") as MockSession:
            MockSession.return_value.get.side_effect = responses
            result = fetch_traces_window(
                **{**_BASE_KWARGS, "max_traces": 999, "page_size": 5},
                budget=budget,
            )

        assert len(result) == 7
        assert budget.exhausted()

    @patch("utils.langfuse_api.time_mod.sleep")
    def test_no_budget_uses_max_traces(self, mock_sleep):
        """Without budget, old max_traces logic still works."""
        # page_size=5 so limit=5, full pages trigger next page
        responses = [
            _page_response(_traces(5, offset=0)),
            _page_response(_traces(5, offset=5)),
            _page_response(_traces(5, offset=10)),
        ]
        kwargs = {**_BASE_KWARGS, "max_traces": 7, "page_size": 5}
        with patch("requests.Session") as MockSession:
            MockSession.return_value.get.side_effect = responses
            result = fetch_traces_window(**kwargs)

        assert len(result) == 7


# ---------------------------------------------------------------------------
# Progress callback
# ---------------------------------------------------------------------------


class TestProgressCallback:
    @patch("utils.langfuse_api.time_mod.sleep")
    def test_on_progress_called_per_page(self, mock_sleep):
        """on_progress is called after each successful page."""
        progress_calls: list[tuple[int, int]] = []

        def on_progress(pages_done: int, traces_so_far: int):
            progress_calls.append((pages_done, traces_so_far))

        # page_size=3 so limit=3, first page is full → triggers page 2
        responses = [
            _page_response(_traces(3, offset=0)),
            _page_response(_traces(2, offset=3)),  # partial → stops after this
        ]
        with patch("requests.Session") as MockSession:
            MockSession.return_value.get.side_effect = responses
            result = fetch_traces_window(
                **{**_BASE_KWARGS, "page_size": 3}, on_progress=on_progress,
            )

        assert len(result) == 5
        assert progress_calls == [(1, 3), (2, 5)]

    @patch("utils.langfuse_api.time_mod.sleep")
    def test_on_progress_exception_ignored(self, mock_sleep):
        """If callback raises, fetch continues."""
        def bad_callback(p, t):
            raise RuntimeError("boom")

        responses = [_page_response(_traces(2)), _page_response([])]
        with patch("requests.Session") as MockSession:
            MockSession.return_value.get.side_effect = responses
            result = fetch_traces_window(**_BASE_KWARGS, on_progress=bad_callback)

        assert len(result) == 2
