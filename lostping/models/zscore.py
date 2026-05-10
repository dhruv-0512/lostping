"""
lostping/models/zscore.py

Z-score baseline anomaly detection.

For each vessel type, computes the mean and std of:
  - mean_sog       (average speed)
  - max_sog        (top speed)
  - speed_variance_score
  - heading_erraticism_score
  - gap_score

Then scores each vessel by how many standard deviations it sits above
its type's mean. Vessels that are statistically extreme for their peer
group get a high zscore_score.

This is the simplest model — no ML, no clustering — just "how weird is
this vessel compared to others of the same type."
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ── Features to z-score ───────────────────────────────────────────────────────
ZSCORE_FEATURES = [
    "mean_sog",
    "max_sog",
    "speed_variance_score",
    "heading_erraticism_score",
    "gap_score",
]

# Clip z-scores at this value before normalising (prevents one extreme
# outlier from compressing everyone else's scores)
ZSCORE_CLIP = 4.0

# Minimum vessels of a type to compute meaningful type-level stats
MIN_TYPE_SIZE = 10


# ── Public API ────────────────────────────────────────────────────────────────

def compute_zscore_scores(feature_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Add a zscore_score (0–1) column to the feature matrix.

    Method:
      1. For each vessel type, compute mean + std of ZSCORE_FEATURES.
      2. For each vessel, compute z = (value - type_mean) / type_std per feature.
      3. Clip z at ZSCORE_CLIP, normalise to [0, 1].
      4. Average across features → zscore_score.
      5. Vessels in small types fall back to global stats.

    Args:
        feature_matrix: Output of builder.build_feature_matrix()
                        (with isolation_score and loiter_score already merged in).

    Returns:
        feature_matrix with new column: zscore_score (0–1).
    """
    fm = feature_matrix.copy()

    # Only use features that actually exist in the matrix
    features = [f for f in ZSCORE_FEATURES if f in fm.columns]

    fm["zscore_score"] = 0.0
    vtype_col = fm["vessel_type"].fillna("unknown")
    type_counts = vtype_col.value_counts()

    # ── Precompute global stats as fallback ───────────────────────────────────
    global_means = fm[features].mean()
    global_stds  = fm[features].std().replace(0, 1)

    def _zscores_for_group(subset: pd.DataFrame, means: pd.Series, stds: pd.Series) -> np.ndarray:
        """Return per-row mean z-score (clipped, normalised) for a subset."""
        z = (subset[features].values - means[features].values) / stds[features].values
        z = np.clip(z, 0, ZSCORE_CLIP)          # only care about above-average anomalies
        z_norm = z / ZSCORE_CLIP                 # normalise to [0, 1]
        return z_norm.mean(axis=1)               # average across features

    # ── Per-type scoring ──────────────────────────────────────────────────────
    for vtype, count in type_counts.items():
        mask = vtype_col == vtype

        if count < MIN_TYPE_SIZE:
            # Use global stats
            scores = _zscores_for_group(fm[mask], global_means, global_stds)
        else:
            type_means = fm[mask][features].mean()
            type_stds  = fm[mask][features].std().replace(0, 1)
            scores     = _zscores_for_group(fm[mask], type_means, type_stds)

        fm.loc[mask, "zscore_score"] = scores

    fm["zscore_score"] = fm["zscore_score"].round(4)
    return fm