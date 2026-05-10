"""
lostping/models/scorer.py

Computes a final risk score 0–100 for every vessel from the feature matrix.

Formula:
    risk = (0.40 * trajectory_score
          + 0.35 * gap_score
          + 0.25 * spatial_score) * 100

Vessels above risk_threshold are written to the flagged list.
Also attaches a human-readable reason string explaining what drove the score.
"""

from __future__ import annotations

import pandas as pd


# ── Weights (must sum to 1.0) ─────────────────────────────────────────────────
W_TRAJECTORY = 0.30
W_GAP        = 0.25
W_SPATIAL    = 0.20
W_ISOLATION  = 0.15
W_ZSCORE     = 0.10
# Default threshold above which a vessel is considered flagged
DEFAULT_RISK_THRESHOLD = 50


# ── Reason builder ────────────────────────────────────────────────────────────

def _build_reason(row: pd.Series) -> str:
    """
    Return a short human-readable string explaining the top contributors
    to this vessel's risk score.
    """
    parts = []

    # Gap signal
    if row.get("gap_score", 0) >= 0.6:
        hrs = row.get("max_gap_hours", 0)
        parts.append(f"AIS dark {hrs:.1f}h")
    elif row.get("gap_score", 0) >= 0.3:
        parts.append("AIS gaps detected")

    # Trajectory signal
    if row.get("speed_variance_score", 0) >= 0.7:
        parts.append("erratic speed")
    if row.get("heading_erraticism_score", 0) >= 0.7:
        parts.append("erratic heading")
    if row.get("reversal_score", 0) >= 0.67:
        parts.append("speed reversals")

    # Spatial signal
    if row.get("port_loiter_score", 0) >= 0.5:
        parts.append("port loitering")
    if row.get("open_ocean_stop_score", 0) >= 0.5:
        parts.append("open-ocean stop")
    if row.get("eez_hug_score", 0) >= 0.3:
        parts.append("EEZ boundary hugging")

    if not parts:
        return "low-level anomaly across multiple signals"
    return "; ".join(parts)


# ── Public API ────────────────────────────────────────────────────────────────

def compute_risk_scores(feature_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Add a risk_score (0–100) and reason column to the feature matrix.

    Args:
        feature_matrix: Output of builder.build_feature_matrix().

    Returns:
        Same DataFrame with two new columns: risk_score, reason.
        Sorted by risk_score descending.
    """
    fm = feature_matrix.copy()

    traj = fm.get("trajectory_score", pd.Series(0, index=fm.index))
    gap  = fm.get("gap_score",        pd.Series(0, index=fm.index))
    spat = fm.get("spatial_score",    pd.Series(0, index=fm.index))
    iso   = fm.get("isolation_score", pd.Series(0, index=fm.index))
    zsc   = fm.get("zscore_score",    pd.Series(0, index=fm.index))

    fm["risk_score"] = ((W_TRAJECTORY * traj
     + W_GAP        * gap
     + W_SPATIAL    * spat
     + W_ISOLATION  * iso
     + W_ZSCORE     * zsc) * 100
    ).round(1)

    fm["reason"] = fm.apply(_build_reason, axis=1)

    return fm.sort_values("risk_score", ascending=False).reset_index(drop=True)


def get_flagged(
    scored: pd.DataFrame,
    threshold: float = DEFAULT_RISK_THRESHOLD,
) -> pd.DataFrame:
    """
    Return only vessels above the risk threshold.

    Args:
        scored:    Output of compute_risk_scores().
        threshold: Risk score cutoff (0–100). Default 50.

    Returns:
        Filtered DataFrame, sorted by risk_score descending.
    """
    flagged = scored[scored["risk_score"] >= threshold].copy()
    print(f"flagged {len(flagged)} vessels above risk threshold {threshold}")
    return flagged


def score_summary(scored: pd.DataFrame) -> None:
    """Print a quick distribution summary of risk scores."""
    rs = scored["risk_score"]
    print(f"\nrisk score distribution ({len(scored)} vessels):")
    print(f"  max    : {rs.max():.1f}")
    print(f"  p95    : {rs.quantile(0.95):.1f}")
    print(f"  p75    : {rs.quantile(0.75):.1f}")
    print(f"  median : {rs.median():.1f}")
    print(f"  mean   : {rs.mean():.1f}")
    print(f"  min    : {rs.min():.1f}")
    buckets = [0, 25, 50, 75, 90, 100]
    for lo, hi in zip(buckets, buckets[1:]):
        n = ((rs >= lo) & (rs < hi)).sum()
        bar = "█" * (n // 20)
        print(f"  {lo:>3}–{hi:<3}: {n:>5}  {bar}")