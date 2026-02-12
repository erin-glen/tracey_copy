"""Tests for utils.user_segments helpers.

Two independent dimensions:
  - Dimension 1 (acquisition): New vs Returning
  - Dimension 2 (engagement): Engaged vs Not Engaged

Verifies:
- Each dimension is a complete binary partition of known users
- The two dimensions are independent (a New user can be Engaged)
- Daily chart classifications are correct per day
- Date type mismatches (Timestamp vs date) don't break comparisons
- Summary (pie) and daily chart relationship is well-defined
"""

from __future__ import annotations

import datetime
from typing import Any

import pandas as pd
import pytest

from utils.user_segments import (
    UserSegments,
    build_daily_user_segments,
    build_first_seen_lookup,
    classify_user_segments,
    compute_engaged_users,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

START = datetime.date(2024, 6, 1)
END = datetime.date(2024, 6, 30)


def _make_traces(
    rows: list[dict[str, Any]],
) -> pd.DataFrame:
    """Build a minimal trace DataFrame from a list of dicts."""
    df = pd.DataFrame(rows)
    for col in ("user_id", "session_id", "trace_id"):
        if col not in df.columns:
            df[col] = None
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], utc=True).dt.date
    return df


def _make_first_seen(rows: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _engaged_trace_rows(user_id: str, date_str: str) -> list[dict]:
    """Generate trace rows that make a user 'engaged': 2 sessions × 2 prompts."""
    rows = []
    for s in range(2):
        for p in range(2):
            rows.append({
                "user_id": user_id,
                "session_id": f"{user_id}_s{s}",
                "trace_id": f"{user_id}_s{s}_t{p}",
                "date": date_str,
            })
    return rows


# ---------------------------------------------------------------------------
# compute_engaged_users
# ---------------------------------------------------------------------------


class TestComputeEngagedUsers:
    def test_basic_engaged(self):
        """User with 2 sessions × 2 prompts each → engaged."""
        rows = [
            {"user_id": "u1", "session_id": "s1", "trace_id": "t1"},
            {"user_id": "u1", "session_id": "s1", "trace_id": "t2"},
            {"user_id": "u1", "session_id": "s2", "trace_id": "t3"},
            {"user_id": "u1", "session_id": "s2", "trace_id": "t4"},
        ]
        df = _make_traces(rows)
        assert compute_engaged_users(df) == {"u1"}

    def test_not_enough_sessions(self):
        """User with 1 session × 3 prompts → NOT engaged."""
        rows = [
            {"user_id": "u1", "session_id": "s1", "trace_id": f"t{i}"}
            for i in range(5)
        ]
        df = _make_traces(rows)
        assert compute_engaged_users(df) == set()

    def test_not_enough_prompts_per_session(self):
        """User with 3 sessions but each has only 1 prompt → NOT engaged."""
        rows = [
            {"user_id": "u1", "session_id": f"s{i}", "trace_id": f"t{i}"}
            for i in range(3)
        ]
        df = _make_traces(rows)
        assert compute_engaged_users(df) == set()

    def test_restrict_to(self):
        """restrict_to filters out users not in the set."""
        rows = [
            {"user_id": uid, "session_id": f"s{i}", "trace_id": f"{uid}_s{i}_t{j}"}
            for uid in ("u1", "u2")
            for i in range(2)
            for j in range(2)
        ]
        df = _make_traces(rows)
        assert compute_engaged_users(df, restrict_to={"u1"}) == {"u1"}
        assert compute_engaged_users(df, restrict_to={"u3"}) == set()

    def test_empty_df(self):
        df = pd.DataFrame(columns=["user_id", "session_id", "trace_id"])
        assert compute_engaged_users(df) == set()

    def test_missing_columns(self):
        df = pd.DataFrame({"x": [1]})
        assert compute_engaged_users(df) == set()


# ---------------------------------------------------------------------------
# build_first_seen_lookup
# ---------------------------------------------------------------------------


class TestBuildFirstSeenLookup:
    def test_authoritative_source(self):
        df = _make_traces([
            {"user_id": "u1", "date": "2024-06-05", "trace_id": "t1", "session_id": "s1"},
        ])
        fs_df = _make_first_seen([
            {"user_id": "u1", "first_seen": "2024-05-01T00:00:00Z"},
        ])
        lookup, filled = build_first_seen_lookup(df, fs_df, START, END)
        assert lookup["u1"] == datetime.date(2024, 5, 1)
        assert filled == 0

    def test_window_fallback(self):
        df = _make_traces([
            {"user_id": "u1", "date": "2024-06-05", "trace_id": "t1", "session_id": "s1"},
            {"user_id": "u1", "date": "2024-06-10", "trace_id": "t2", "session_id": "s2"},
        ])
        lookup, filled = build_first_seen_lookup(df, None, START, END)
        assert lookup["u1"] == datetime.date(2024, 6, 5)
        assert filled == 1

    def test_mixed_sources(self):
        """u1 from authoritative, u2 from window fallback."""
        df = _make_traces([
            {"user_id": "u1", "date": "2024-06-05", "trace_id": "t1", "session_id": "s1"},
            {"user_id": "u2", "date": "2024-06-10", "trace_id": "t2", "session_id": "s2"},
        ])
        fs_df = _make_first_seen([
            {"user_id": "u1", "first_seen": "2024-04-01T00:00:00Z"},
        ])
        lookup, filled = build_first_seen_lookup(df, fs_df, START, END)
        assert lookup["u1"] == datetime.date(2024, 4, 1)
        assert lookup["u2"] == datetime.date(2024, 6, 10)
        assert filled == 1


# ---------------------------------------------------------------------------
# classify_user_segments — two independent dimensions
# ---------------------------------------------------------------------------


class TestClassifyUserSegments:
    def test_acquisition_dimension_is_partition(self):
        """New + Returning = all known users (no overlap)."""
        rows = (
            _engaged_trace_rows("eng1", "2024-06-10")
            + [
                {"user_id": "new1", "session_id": "sn", "trace_id": "tn", "date": "2024-06-10"},
                {"user_id": "ret1", "session_id": "sr", "trace_id": "tr", "date": "2024-06-10"},
            ]
        )
        df = _make_traces(rows)
        fs_df = _make_first_seen([
            {"user_id": "eng1", "first_seen": "2024-03-01T00:00:00Z"},
            {"user_id": "new1", "first_seen": "2024-06-10T00:00:00Z"},
            {"user_id": "ret1", "first_seen": "2024-05-01T00:00:00Z"},
        ])
        seg = classify_user_segments(df, fs_df, START, END)

        assert seg.new_users & seg.returning_users == set()
        assert seg.new_users | seg.returning_users == {"eng1", "new1", "ret1"}

    def test_engagement_dimension_is_partition(self):
        """Engaged + Not Engaged = all known users (no overlap)."""
        rows = (
            _engaged_trace_rows("eng1", "2024-06-10")
            + [
                {"user_id": "new1", "session_id": "sn", "trace_id": "tn", "date": "2024-06-10"},
                {"user_id": "ret1", "session_id": "sr", "trace_id": "tr", "date": "2024-06-10"},
            ]
        )
        df = _make_traces(rows)
        fs_df = _make_first_seen([
            {"user_id": "eng1", "first_seen": "2024-03-01T00:00:00Z"},
            {"user_id": "new1", "first_seen": "2024-06-10T00:00:00Z"},
            {"user_id": "ret1", "first_seen": "2024-05-01T00:00:00Z"},
        ])
        seg = classify_user_segments(df, fs_df, START, END)

        assert seg.engaged_users & seg.not_engaged_users == set()
        assert seg.engaged_users | seg.not_engaged_users == {"eng1", "new1", "ret1"}

    def test_dimensions_are_independent(self):
        """A new user CAN be engaged. Dimensions don't constrain each other."""
        rows = _engaged_trace_rows("new_eng", "2024-06-10")
        rows.append({"user_id": "ret_not", "session_id": "s", "trace_id": "t", "date": "2024-06-10"})
        df = _make_traces(rows)
        fs_df = _make_first_seen([
            {"user_id": "new_eng", "first_seen": "2024-06-10T00:00:00Z"},
            {"user_id": "ret_not", "first_seen": "2024-05-01T00:00:00Z"},
        ])

        seg = classify_user_segments(df, fs_df, START, END)

        # new_eng is New AND Engaged
        assert "new_eng" in seg.new_users
        assert "new_eng" in seg.engaged_users

        # ret_not is Returning AND Not Engaged
        assert "ret_not" in seg.returning_users
        assert "ret_not" in seg.not_engaged_users

    def test_returning_user_can_be_engaged(self):
        rows = _engaged_trace_rows("ret_eng", "2024-06-10")
        df = _make_traces(rows)
        fs_df = _make_first_seen([
            {"user_id": "ret_eng", "first_seen": "2024-03-01T00:00:00Z"},
        ])
        seg = classify_user_segments(df, fs_df, START, END)

        assert "ret_eng" in seg.returning_users
        assert "ret_eng" in seg.engaged_users


# ---------------------------------------------------------------------------
# build_daily_user_segments — two independent dimension pairs per day
# ---------------------------------------------------------------------------


class TestBuildDailyUserSegments:
    def _simple_setup(self) -> tuple[pd.DataFrame, dict[str, Any], set[str]]:
        """3 users: new (u_new), returning (u_ret), engaged returning (u_eng).

        All active on 2024-06-05 and 2024-06-06.
        u_new's first_seen_date is 2024-06-05.
        """
        rows = []
        for d in ("2024-06-05", "2024-06-06"):
            for uid in ("u_new", "u_ret", "u_eng"):
                rows.append({
                    "user_id": uid,
                    "session_id": f"{uid}_s",
                    "trace_id": f"{uid}_{d}",
                    "date": d,
                })
        df = _make_traces(rows)
        fs = {
            "u_new": datetime.date(2024, 6, 5),
            "u_ret": datetime.date(2024, 5, 1),
            "u_eng": datetime.date(2024, 4, 1),
        }
        engaged = {"u_eng"}
        return df, fs, engaged

    def test_acquisition_dimension_sums_to_total_each_day(self):
        df, fs, engaged = self._simple_setup()
        daily = build_daily_user_segments(df, fs, engaged)
        for _, row in daily.iterrows():
            assert row["new_users"] + row["returning_users"] == 3

    def test_engagement_dimension_sums_to_total_each_day(self):
        df, fs, engaged = self._simple_setup()
        daily = build_daily_user_segments(df, fs, engaged)
        for _, row in daily.iterrows():
            assert row["engaged_users"] + row["not_engaged_users"] == 3

    def test_new_only_on_first_seen_date(self):
        """u_new is New on June 5, Returning on June 6."""
        df, fs, engaged = self._simple_setup()
        daily = build_daily_user_segments(df, fs, engaged)

        day1 = daily[daily["date"] == datetime.date(2024, 6, 5)].iloc[0]
        day2 = daily[daily["date"] == datetime.date(2024, 6, 6)].iloc[0]

        assert day1["new_users"] == 1
        assert day1["returning_users"] == 2

        assert day2["new_users"] == 0
        assert day2["returning_users"] == 3  # u_new is now returning

    def test_engagement_is_constant_across_days(self):
        """u_eng is engaged on both days (engagement doesn't change per day)."""
        df, fs, engaged = self._simple_setup()
        daily = build_daily_user_segments(df, fs, engaged)

        for _, row in daily.iterrows():
            assert row["engaged_users"] == 1
            assert row["not_engaged_users"] == 2

    def test_timestamp_vs_date_no_mismatch(self):
        """Dates stored as Timestamps in df should still compare correctly."""
        rows = [
            {"user_id": "u1", "session_id": "s1", "trace_id": "t1", "date": pd.Timestamp("2024-06-05", tz="UTC")},
        ]
        df = pd.DataFrame(rows)
        fs = {"u1": datetime.date(2024, 6, 5)}

        daily = build_daily_user_segments(df, fs, set())
        assert len(daily) == 1
        assert daily.iloc[0]["new_users"] == 1
        assert daily.iloc[0]["not_engaged_users"] == 1

    def test_empty_dataframe(self):
        df = pd.DataFrame(columns=["date", "user_id", "session_id", "trace_id"])
        result = build_daily_user_segments(df, {}, set())
        assert result.empty


# ---------------------------------------------------------------------------
# Consistency: classify vs daily
# ---------------------------------------------------------------------------


class TestConsistency:
    """Verify that summary (classify) and daily (build_daily) agree."""

    def _build_scenario(self):
        """Multi-user scenario over a week."""
        rows = []
        # new1: first seen June 3 (within range), active June 3-5
        for d in ("2024-06-03", "2024-06-04", "2024-06-05"):
            rows.append({"user_id": "new1", "session_id": "n1s", "trace_id": f"new1_{d}", "date": d})

        # ret1: first seen before start, active June 3-4, NOT engaged
        for d in ("2024-06-03", "2024-06-04"):
            rows.append({"user_id": "ret1", "session_id": "r1s", "trace_id": f"ret1_{d}", "date": d})

        # eng1: first seen before start, active June 3-5, IS engaged
        for d in ("2024-06-03", "2024-06-04", "2024-06-05"):
            rows += _engaged_trace_rows("eng1", d)

        df = _make_traces(rows)
        fs_df = _make_first_seen([
            {"user_id": "new1", "first_seen": "2024-06-03T00:00:00Z"},
            {"user_id": "ret1", "first_seen": "2024-05-01T00:00:00Z"},
            {"user_id": "eng1", "first_seen": "2024-04-15T00:00:00Z"},
        ])
        return df, fs_df

    def test_acquisition_partition_matches_summary(self):
        df, fs_df = self._build_scenario()
        seg = classify_user_segments(df, fs_df, START, END)

        assert seg.new_users & seg.returning_users == set()
        assert seg.new_users | seg.returning_users == {"new1", "ret1", "eng1"}

    def test_engagement_partition_matches_summary(self):
        df, fs_df = self._build_scenario()
        seg = classify_user_segments(df, fs_df, START, END)

        assert seg.engaged_users & seg.not_engaged_users == set()
        assert seg.engaged_users | seg.not_engaged_users == {"new1", "ret1", "eng1"}

    def test_daily_new_sum_leq_summary_new(self):
        """Daily new sum ≤ summary new."""
        df, fs_df = self._build_scenario()
        seg = classify_user_segments(df, fs_df, START, END)
        daily = build_daily_user_segments(df, seg.first_seen_by_user, seg.engaged_users)

        daily_new_sum = int(daily["new_users"].sum())
        summary_new = len(seg.new_users)
        assert daily_new_sum <= summary_new
        # new1 IS active on first_seen_date (June 3)
        assert daily_new_sum == summary_new

    def test_daily_new_sum_less_when_not_active_on_first_seen(self):
        """If a new user is NOT active on their first_seen_date, daily sum < summary."""
        rows = [
            {"user_id": "new1", "session_id": "s1", "trace_id": "t1", "date": "2024-06-05"},
        ]
        df = _make_traces(rows)
        fs_df = _make_first_seen([
            {"user_id": "new1", "first_seen": "2024-06-03T00:00:00Z"},
        ])

        seg = classify_user_segments(df, fs_df, START, END)
        daily = build_daily_user_segments(df, seg.first_seen_by_user, seg.engaged_users)

        assert len(seg.new_users) == 1
        assert int(daily["new_users"].sum()) == 0
        # On June 5, first_seen_date(June 3) != date(June 5) → returning

    def test_pie_matches_summary_both_dimensions(self):
        """Pie values should match summary counts for both dimensions."""
        df, fs_df = self._build_scenario()
        seg = classify_user_segments(df, fs_df, START, END)

        # Dimension 1
        assert len(seg.new_users) + len(seg.returning_users) == 3
        assert seg.new_users == {"new1"}
        assert seg.returning_users == {"ret1", "eng1"}

        # Dimension 2
        assert len(seg.engaged_users) + len(seg.not_engaged_users) == 3
        assert seg.engaged_users == {"eng1"}
        assert seg.not_engaged_users == {"new1", "ret1"}
