"""User segmentation helpers – two independent dimensions.

Dimension 1 – Acquisition (New vs Returning)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- **New**: users whose first-ever trace was on or after ``start_date``.
- **Returning**: users whose first-ever trace was before ``start_date``.

Dimension 2 – Engagement (Engaged vs Not Engaged)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- **Engaged**: users with ≥2 sessions, each containing ≥2 prompts, within the
  loaded trace window.  This applies to **all** users (new and returning).
- **Not Engaged**: everyone else.

Each dimension is a simple binary split.  The two dimensions are independent:
a user can be New+Engaged, New+Not-Engaged, Returning+Engaged, or
Returning+Not-Engaged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _clean_id_series(s: pd.Series) -> pd.Series:
    """Coerce to str, strip whitespace, drop blanks."""
    return s.astype(str).str.strip().loc[lambda x: x.ne("")]


def _clean_id_set(df: pd.DataFrame, col: str) -> set[str]:
    """Return the set of non-empty, stripped string IDs in *col*."""
    s = df[col].dropna()
    return set(_clean_id_series(s).unique())


# ---------------------------------------------------------------------------
# Engaged-user detection
# ---------------------------------------------------------------------------

def compute_engaged_users(
    df: pd.DataFrame,
    *,
    min_sessions: int = 2,
    min_prompts_per_session: int = 2,
    restrict_to: set[str] | None = None,
) -> set[str]:
    """Identify users with ≥ *min_sessions* sessions each having ≥ *min_prompts_per_session* prompts.

    Parameters
    ----------
    df:
        Must contain columns ``user_id``, ``session_id``, ``trace_id``.
    restrict_to:
        If given, only users in this set can be returned (e.g. returning
        users only).

    Returns
    -------
    set[str]
        User IDs meeting the engagement criteria.
    """
    required = {"user_id", "session_id", "trace_id"}
    if not required.issubset(df.columns):
        return set()

    base = df.dropna(subset=["user_id", "session_id"]).copy()
    base["user_id"] = _clean_id_series(base["user_id"])
    base["session_id"] = _clean_id_series(base["session_id"])
    base = base[base["user_id"].ne("") & base["session_id"].ne("")]

    if restrict_to is not None:
        base = base[base["user_id"].isin(restrict_to)]
    if base.empty:
        return set()

    session_prompt_counts = (
        base.groupby(["user_id", "session_id"], dropna=True)
        .agg(prompts=("trace_id", "count"))
        .reset_index()
    )
    good_sessions = session_prompt_counts[
        session_prompt_counts["prompts"] >= min_prompts_per_session
    ]
    engaged_counts = (
        good_sessions.groupby("user_id", dropna=True)
        .agg(good_sessions=("session_id", "nunique"))
        .reset_index()
    )
    return set(
        engaged_counts
        .loc[engaged_counts["good_sessions"] >= min_sessions, "user_id"]
        .tolist()
    )


# ---------------------------------------------------------------------------
# First-seen lookup
# ---------------------------------------------------------------------------

def build_first_seen_lookup(
    df: pd.DataFrame,
    user_first_seen_df: pd.DataFrame | None,
    start_date,
    end_date,
) -> tuple[dict[str, Any], int]:
    """Build a ``{user_id: first_seen_date}`` dict for every active user.

    1. Start with the authoritative ``user_first_seen_df`` (from Langfuse).
    2. For any active user missing from that table, fall back to the earliest
       date observed in the loaded trace window.

    Returns
    -------
    (first_seen_by_user, filled_from_window)
        *filled_from_window* counts how many users were back-filled from the
        trace window.
    """
    if "user_id" not in df.columns:
        return {}, 0

    active_users = _clean_id_set(df, "user_id")
    if not active_users:
        return {}, 0

    first_seen_by_user: dict[str, Any] = {}

    # --- authoritative source ---
    if user_first_seen_df is not None and len(user_first_seen_df):
        fs = user_first_seen_df.dropna(subset=["user_id", "first_seen"]).copy()
        fs["user_id"] = _clean_id_series(fs["user_id"])
        fs = fs[fs["user_id"].ne("") & fs["user_id"].isin(active_users)]
        fs_dt = pd.to_datetime(fs["first_seen"], errors="coerce", utc=True)
        fs = fs.assign(first_seen_date=fs_dt.dt.date).dropna(subset=["first_seen_date"])
        first_seen_by_user = {
            str(r["user_id"]): r["first_seen_date"]
            for r in fs.to_dict("records")
        }

    # --- window fall-back ---
    missing = active_users - set(first_seen_by_user)
    filled = 0
    if missing and "date" in df.columns and df["date"].notna().any():
        bud = df.dropna(subset=["user_id", "date"]).copy()
        bud["user_id"] = _clean_id_series(bud["user_id"])
        bud = bud[bud["user_id"].ne("")]
        bud["date"] = pd.to_datetime(bud["date"], utc=True, errors="coerce").dt.date
        bud = bud.dropna(subset=["date"])
        window_min = bud.groupby("user_id", dropna=True)["date"].min().to_dict()
        for u in missing:
            d = window_min.get(u)
            if d is not None:
                first_seen_by_user[u] = d
                filled += 1

    return first_seen_by_user, filled


# ---------------------------------------------------------------------------
# Full segment classification
# ---------------------------------------------------------------------------

@dataclass
class UserSegments:
    """Container for two independent user-segmentation dimensions.

    Dimension 1 (acquisition): ``new_users`` and ``returning_users`` are
    disjoint and together cover all known users.

    Dimension 2 (engagement): ``engaged_users`` and ``not_engaged_users`` are
    disjoint and together cover all known users.

    The two dimensions are independent – a user can be in any combination.
    """

    first_seen_by_user: dict[str, Any] = field(default_factory=dict)
    # Dimension 1: acquisition
    new_users: set[str] = field(default_factory=set)
    returning_users: set[str] = field(default_factory=set)
    # Dimension 2: engagement
    engaged_users: set[str] = field(default_factory=set)
    not_engaged_users: set[str] = field(default_factory=set)
    # Metadata
    unknown_users: set[str] = field(default_factory=set)
    filled_from_window: int = 0


def classify_user_segments(
    df: pd.DataFrame,
    user_first_seen_df: pd.DataFrame | None,
    start_date,
    end_date,
) -> UserSegments:
    """Classify active users along two independent dimensions.

    See module docstring for precise definitions.
    """
    first_seen_by_user, filled = build_first_seen_lookup(
        df, user_first_seen_df, start_date, end_date,
    )

    active_users = _clean_id_set(df, "user_id") if "user_id" in df.columns else set()
    known_users = set(first_seen_by_user)
    unknown = active_users - known_users

    fs_dates = pd.Series(list(first_seen_by_user.values()))
    if fs_dates.empty:
        return UserSegments(
            first_seen_by_user=first_seen_by_user,
            unknown_users=unknown,
            filled_from_window=filled,
        )

    # --- Dimension 1: acquisition ---
    new_mask = (fs_dates >= start_date) & (fs_dates <= end_date)
    returning_mask = fs_dates < start_date

    uids = list(first_seen_by_user.keys())
    new_users = {uids[i] for i, v in enumerate(new_mask) if v}
    returning_users = {uids[i] for i, v in enumerate(returning_mask) if v}

    # --- Dimension 2: engagement (applies to ALL users) ---
    engaged = compute_engaged_users(df)
    not_engaged = known_users - engaged - unknown

    return UserSegments(
        first_seen_by_user=first_seen_by_user,
        new_users=new_users,
        returning_users=returning_users,
        engaged_users=engaged,
        not_engaged_users=not_engaged,
        unknown_users=unknown,
        filled_from_window=filled,
    )


# ---------------------------------------------------------------------------
# Daily breakdown for charts
# ---------------------------------------------------------------------------

def build_daily_user_segments(
    df: pd.DataFrame,
    first_seen_by_user: dict[str, Any],
    engaged_users: set[str],
) -> pd.DataFrame:
    """Produce a per-day DataFrame with two independent dimension pairs.

    Columns returned:
    - ``new_users`` / ``returning_users``  (Dimension 1 – acquisition)
    - ``engaged_users`` / ``not_engaged_users`` (Dimension 2 – engagement)

    Dimension 1 per day
    ~~~~~~~~~~~~~~~~~~~
    - **New** on day *D*: ``first_seen_date == D``
    - **Returning** on day *D*: ``first_seen_date < D``

    Dimension 2 per day
    ~~~~~~~~~~~~~~~~~~~
    - **Engaged** on day *D*: user is in *engaged_users*.
    - **Not Engaged** on day *D*: user is **not** in *engaged_users*.

    Each dimension sums to the total distinct users active that day.
    """
    if "date" not in df.columns or "user_id" not in df.columns:
        return pd.DataFrame()

    base = df.dropna(subset=["date", "user_id"]).copy()
    base["user_id"] = _clean_id_series(base["user_id"])
    base = base[base["user_id"].ne("")]
    base = base.drop_duplicates(subset=["date", "user_id"])

    if base.empty:
        return pd.DataFrame()

    base["first_seen_date"] = base["user_id"].map(first_seen_by_user)
    base = base.dropna(subset=["first_seen_date"])

    # Normalise both to datetime.date to avoid Timestamp-vs-date mismatches.
    base["date"] = pd.to_datetime(base["date"], utc=True, errors="coerce").dt.date
    base["first_seen_date"] = pd.to_datetime(
        pd.Series(base["first_seen_date"].values), utc=True, errors="coerce",
    ).dt.date.values
    base = base.dropna(subset=["date", "first_seen_date"])

    # Dimension 1: acquisition
    base["is_new"] = base["first_seen_date"] == base["date"]
    base["is_returning"] = ~base["is_new"]

    # Dimension 2: engagement (independent of acquisition)
    base["is_engaged"] = base["user_id"].isin(engaged_users)
    base["is_not_engaged"] = ~base["is_engaged"]

    daily = (
        base.groupby("date")
        .agg(
            new_users=("is_new", "sum"),
            returning_users=("is_returning", "sum"),
            engaged_users=("is_engaged", "sum"),
            not_engaged_users=("is_not_engaged", "sum"),
        )
        .reset_index()
        .sort_values("date")
    )
    for col in ("new_users", "returning_users", "engaged_users", "not_engaged_users"):
        daily[col] = pd.to_numeric(daily[col], errors="coerce").fillna(0).astype(int)

    return daily
