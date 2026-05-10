"""
lostping/features/builder.py

Assembles one feature vector per vessel from all upstream scoring modules:
  - gaps.py        → gap_score + gap metadata
  - trajectory.py  → trajectory_score + sub-scores
  - spatial.py     → spatial_score + sub-scores
  - raw AIS df     → vessel metadata (name, type, ping count, track bounds)

Output: a single DataFrame, one row per vessel, ready for models/.
"""

from __future__ import annotations

import pandas as pd

from lostping.features.gaps import compute_gaps_all
from lostping.features.trajectory import compute_trajectory_scores, trajectory_scores_to_df
from lostping.features.spatial import compute_spatial_scores, spatial_scores_to_df


# ── Metadata extract ──────────────────────────────────────────────────────────

def _extract_vessel_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """
    One row per vessel with static/summary fields from the AIS data.
    """
    meta = (
        df.groupby("mmsi", sort=False)
        .agg(
            vessel_name  =("vessel_name",  lambda x: x.dropna().mode().iloc[0] if not x.dropna().empty else "UNKNOWN"),
            vessel_type  =("vessel_type",  lambda x: x.dropna().mode().iloc[0] if not x.dropna().empty else "unknown"),
            ping_count   =("timestamp",    "count"),
            first_seen   =("timestamp",    "min"),
            last_seen    =("timestamp",    "max"),
            lat_min      =("lat",          "min"),
            lat_max      =("lat",          "max"),
            lon_min      =("lon",          "min"),
            lon_max      =("lon",          "max"),
            mean_sog     =("sog",          "mean"),
            max_sog      =("sog",          "max"),
        )
        .reset_index()
    )
    meta["first_seen"] = meta["first_seen"].astype(str)
    meta["last_seen"]  = meta["last_seen"].astype(str)
    meta["mean_sog"]   = meta["mean_sog"].round(2)
    meta["max_sog"]    = meta["max_sog"].round(2)
    return meta


# ── Public API ────────────────────────────────────────────────────────────────

def build_feature_matrix(
    df: pd.DataFrame,
    gap_threshold_hours: float = 3.0,
) -> pd.DataFrame:
    """
    Run all feature extractors and join into one feature matrix.

    Args:
        df:                   Cleaned AIS DataFrame from cleaner.py.
        gap_threshold_hours:  Passed to gap scorer (default 3h).

    Returns:
        DataFrame with one row per vessel. Columns:
            mmsi, vessel_name, vessel_type, ping_count, first_seen, last_seen,
            lat_min, lat_max, lon_min, lon_max, mean_sog, max_sog,
            gap_score, gap_count, max_gap_hours, sts_proximity,
            trajectory_score, speed_variance_score, heading_erraticism_score, reversal_score,
            trajectory_event_count,
            spatial_score, port_loiter_score, open_ocean_stop_score, eez_hug_score,
            spatial_event_count
    """
    print("building feature matrix...")

    # ── Metadata ──────────────────────────────────────────────────────────────
    print("  extracting vessel metadata...", end=" ", flush=True)
    meta = _extract_vessel_metadata(df)
    print(f"{len(meta)} vessels")

    # ── Gap scores ────────────────────────────────────────────────────────────
    print("  computing gap scores...", end=" ", flush=True)
    gap_df = compute_gaps_all(df)
    print(f"{len(gap_df)} vessels scored")

    # ── Trajectory scores ─────────────────────────────────────────────────────
    print("  computing trajectory scores...", end=" ", flush=True)
    traj_results = compute_trajectory_scores(df)
    traj_df = trajectory_scores_to_df(traj_results)
    print(f"{len(traj_df)} vessels scored")

    # ── Spatial scores ────────────────────────────────────────────────────────
    print("  computing spatial scores...", end=" ", flush=True)
    spat_results = compute_spatial_scores(df)
    spat_df = spatial_scores_to_df(spat_results)
    print(f"{len(spat_df)} vessels scored")

    # ── Join ──────────────────────────────────────────────────────────────────
    print("  joining...", end=" ", flush=True)
    feature_matrix = (
        meta
        .merge(gap_df,  on="mmsi", how="left")
        .merge(traj_df, on="mmsi", how="left")
        .merge(spat_df, on="mmsi", how="left")
    )

    # Fill vessels that had no events in a module with 0
    score_cols = [
        "gap_score", "gap_count", "max_gap_hours", "total_gap_hours",
        "trajectory_score", "speed_variance_score", "heading_erraticism_score",
        "reversal_score", "trajectory_event_count",
        "spatial_score", "port_loiter_score", "open_ocean_stop_score",
        "eez_hug_score", "spatial_event_count",
    ]
    for col in score_cols:
        if col in feature_matrix.columns:
            feature_matrix[col] = feature_matrix[col].fillna(0)

    print(f"done — {len(feature_matrix)} vessels, {len(feature_matrix.columns)} columns")
    return feature_matrix


def top_vessels(
    feature_matrix: pd.DataFrame,
    score_col: str = "gap_score",
    n: int = 20,
) -> pd.DataFrame:
    """
    Return top-n vessels sorted by any score column.
    Useful for quick inspection before the risk scorer runs.
    """
    cols = ["mmsi", "vessel_name", "vessel_type", "ping_count", score_col]
    cols = [c for c in cols if c in feature_matrix.columns]
    return (
        feature_matrix[cols]
        .sort_values(score_col, ascending=False)
        .head(n)
        .reset_index(drop=True)
    )