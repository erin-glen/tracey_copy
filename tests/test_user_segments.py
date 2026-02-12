"""Tests for utils.user_segments helpers.

Verifies:
- Mutual exclusivity: New ∩ Returning ∩ Engaged = ∅
- Partition: New + Returning + Engaged = Total known users
- Engaged ⊂ users with first_seen < start_date
- Daily new sum == unique new users who were active on their first_seen_date
- Daily segments are mutually exclusive and sum to day total
- Date type mismatches (Timestamp vs date) don't break comparisons
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
# classify_user_segments
# ---------------------------------------------------------------------------


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


class TestClassifyUserSegments:
    def test_mutually_exclusive_and_exhaustive(self):
        """New + Returning + Engaged + Unknown = all active users."""
        rows = [
            # u1: returning, first seen before start
            {"user_id": "u1", "session_id": "s1", "trace_id": "t1", "date": "2024-06-05"},
            # u2: new, first seen within range
            {"user_id": "u2", "session_id": "s2", "trace_id": "t2", "date": "2024-06-15"},
        ]
        # Make u1 engaged (2 sessions × 2 prompts)
        rows += _engaged_trace_rows("u1", "2024-06-05")
        df = _make_traces(rows)
        fs_df = _make_first_seen([
            {"user_id": "u1", "first_seen": "2024-04-01T00:00:00Z"},
            {"user_id": "u2", "first_seen": "2024-06-15T00:00:00Z"},
        ])

        seg = classify_user_segments(df, fs_df, START, END)

        # Mutually exclusive
        assert seg.new_users & seg.returning_users == set()
        assert seg.new_users & seg.engaged_returning_users == set()
        assert seg.returning_users & seg.engaged_returning_users == set()

        # Exhaustive (excluding unknowns)
        known = seg.new_users | seg.returning_users | seg.engaged_returning_users
        assert known == {"u1", "u2"}

        # u1 is engaged (not returning), u2 is new
        assert seg.new_users == {"u2"}
        assert seg.engaged_returning_users == {"u1"}
        assert seg.returning_users == set()  # u1 moved to engaged

    def test_returning_excludes_engaged(self):
        """Returning set must not contain any engaged users."""
        rows = []
        # u1, u2: both returning (first seen before start)
        # u1 is engaged, u2 is not
        rows += _engaged_trace_rows("u1", "2024-06-05")
        rows.append({"user_id": "u2", "session_id": "s_only", "trace_id": "t_only", "date": "2024-06-05"})
        df = _make_traces(rows)
        fs_df = _make_first_seen([
            {"user_id": "u1", "first_seen": "2024-03-01T00:00:00Z"},
            {"user_id": "u2", "first_seen": "2024-04-01T00:00:00Z"},
        ])

        seg = classify_user_segments(df, fs_df, START, END)

        assert "u1" in seg.engaged_returning_users
        assert "u1" not in seg.returning_users
        assert "u2" in seg.returning_users
        assert "u2" not in seg.engaged_returning_users

    def test_new_plus_returning_plus_engaged_equals_total(self):
        """N + R + E = total known users."""
        rows = (
            _engaged_trace_rows("engaged1", "2024-06-10")
            + [
                {"user_id": "new1", "session_id": "sn1", "trace_id": "tn1", "date": "2024-06-10"},
                {"user_id": "ret1", "session_id": "sr1", "trace_id": "tr1", "date": "2024-06-10"},
            ]
        )
        df = _make_traces(rows)
        fs_df = _make_first_seen([
            {"user_id": "engaged1", "first_seen": "2024-03-01T00:00:00Z"},
            {"user_id": "new1", "first_seen": "2024-06-10T00:00:00Z"},
            {"user_id": "ret1", "first_seen": "2024-05-01T00:00:00Z"},
        ])

        seg = classify_user_segments(df, fs_df, START, END)
        total_known = len(seg.new_users) + len(seg.returning_users) + len(seg.engaged_returning_users)
        assert total_known == 3

    def test_engaged_only_from_pre_start_users(self):
        """A new user meeting engagement thresholds must NOT be classified as engaged."""
        rows = _engaged_trace_rows("new_active", "2024-06-10")
        df = _make_traces(rows)
        fs_df = _make_first_seen([
            {"user_id": "new_active", "first_seen": "2024-06-10T00:00:00Z"},
        ])

        seg = classify_user_segments(df, fs_df, START, END)
        assert "new_active" in seg.new_users
        assert "new_active" not in seg.engaged_returning_users


# ---------------------------------------------------------------------------
# build_daily_user_segments
# ---------------------------------------------------------------------------


class TestBuildDailyUserSegments:
    def _simple_setup(self) -> tuple[pd.DataFrame, dict[str, Any], set[str]]:
        """3 users: new (u_new), returning (u_ret), engaged (u_eng).

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

    def test_daily_mutual_exclusivity(self):
        """On each day the three categories should be mutually exclusive."""
        df, fs, engaged = self._simple_setup()
        daily = build_daily_user_segments(df, fs, engaged)

        for _, row in daily.iterrows():
            total = row["new_users"] + row["returning_users"] + row["engaged_users"]
            assert total == 3  # all 3 users active each day

    def test_new_only_on_first_seen_date(self):
        """u_new should be 'new' on June 5 and 'returning' on June 6."""
        df, fs, engaged = self._simple_setup()
        daily = build_daily_user_segments(df, fs, engaged)

        day1 = daily[daily["date"] == datetime.date(2024, 6, 5)].iloc[0]
        day2 = daily[daily["date"] == datetime.date(2024, 6, 6)].iloc[0]

        # Day 1: u_new=new, u_ret=returning, u_eng=engaged
        assert day1["new_users"] == 1
        assert day1["returning_users"] == 1
        assert day1["engaged_users"] == 1

        # Day 2: u_new is now returning (first_seen < today), u_ret=returning, u_eng=engaged
        assert day2["new_users"] == 0
        assert day2["returning_users"] == 2  # u_new + u_ret
        assert day2["engaged_users"] == 1

    def test_daily_new_sum_equals_unique_new_active_on_first_seen(self):
        """Sum of daily new users = number of new users active on their first_seen_date."""
        df, fs, engaged = self._simple_setup()
        daily = build_daily_user_segments(df, fs, engaged)
        # Only u_new is new, active on first_seen_date June 5
        assert int(daily["new_users"].sum()) == 1

    def test_timestamp_vs_date_no_mismatch(self):
        """Dates stored as Timestamps in df should still compare correctly."""
        rows = [
            {"user_id": "u1", "session_id": "s1", "trace_id": "t1", "date": pd.Timestamp("2024-06-05", tz="UTC")},
        ]
        df = pd.DataFrame(rows)
        # Intentionally leave date as Timestamp (don't convert to .date())
        fs = {"u1": datetime.date(2024, 6, 5)}

        daily = build_daily_user_segments(df, fs, set())
        assert len(daily) == 1
        assert daily.iloc[0]["new_users"] == 1

    def test_engaged_user_not_classified_as_new_even_on_first_seen(self):
        """Edge case: engaged user active on first_seen_date.

        first_seen_date < start_date, so they should never be 'new'.
        On their first_seen_date (which is before start_date), they wouldn't
        appear in traces anyway. But if they did, first_seen_date == date
        would trigger is_new. This is correct because first_seen_date IS that
        day. The key invariant is that engaged users have first_seen < start,
        so they can't be new in the summary. In the daily chart, if somehow
        first_seen_date == date AND user is engaged, is_new takes precedence.
        This is an edge case that only arises if traces include pre-start dates.
        """
        rows = [
            {"user_id": "u_eng", "session_id": "s1", "trace_id": "t1", "date": "2024-06-05"},
        ]
        df = _make_traces(rows)
        fs = {"u_eng": datetime.date(2024, 4, 1)}  # well before June 5
        daily = build_daily_user_segments(df, fs, {"u_eng"})

        assert daily.iloc[0]["engaged_users"] == 1
        assert daily.iloc[0]["new_users"] == 0

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

    def test_summary_segments_are_disjoint(self):
        df, fs_df = self._build_scenario()
        seg = classify_user_segments(df, fs_df, START, END)

        all_sets = [seg.new_users, seg.returning_users, seg.engaged_returning_users]
        for i, a in enumerate(all_sets):
            for j, b in enumerate(all_sets):
                if i != j:
                    assert a & b == set(), f"Sets {i} and {j} overlap: {a & b}"

    def test_summary_partition(self):
        df, fs_df = self._build_scenario()
        seg = classify_user_segments(df, fs_df, START, END)

        total_known = seg.new_users | seg.returning_users | seg.engaged_returning_users
        assert total_known == {"new1", "ret1", "eng1"}

    def test_daily_new_sum_leq_summary_new(self):
        """Daily new sum ≤ summary new (equal when every new user is active on first_seen_date)."""
        df, fs_df = self._build_scenario()
        seg = classify_user_segments(df, fs_df, START, END)
        daily = build_daily_user_segments(df, seg.first_seen_by_user, seg.engaged_returning_users)

        daily_new_sum = int(daily["new_users"].sum())
        summary_new = len(seg.new_users)
        assert daily_new_sum <= summary_new
        # In this scenario, new1 IS active on their first_seen_date (June 3)
        assert daily_new_sum == summary_new

    def test_daily_new_sum_less_when_not_active_on_first_seen(self):
        """If a new user is NOT active on their first_seen_date, daily sum < summary."""
        rows = [
            # new1 first seen June 3, but only active on June 5
            {"user_id": "new1", "session_id": "s1", "trace_id": "t1", "date": "2024-06-05"},
        ]
        df = _make_traces(rows)
        fs_df = _make_first_seen([
            {"user_id": "new1", "first_seen": "2024-06-03T00:00:00Z"},
        ])

        seg = classify_user_segments(df, fs_df, START, END)
        daily = build_daily_user_segments(df, seg.first_seen_by_user, seg.engaged_returning_users)

        assert len(seg.new_users) == 1  # new1 is new in summary
        assert int(daily["new_users"].sum()) == 0  # but not "new" on any active day
        # On June 5, first_seen_date(June 3) != date(June 5) → classified as returning

    def test_pie_matches_summary(self):
        """Pie values (unique users) should match summary counts exactly."""
        df, fs_df = self._build_scenario()
        seg = classify_user_segments(df, fs_df, START, END)

        # These are exactly what the pie chart uses
        pie_new = len(seg.new_users)
        pie_returning = len(seg.returning_users)
        pie_engaged = len(seg.engaged_returning_users)

        assert pie_new == 1  # new1
        assert pie_returning == 1  # ret1 (eng1 moved to engaged)
        assert pie_engaged == 1  # eng1
        assert pie_new + pie_returning + pie_engaged == 3
