"""
lostping/models/isolation.py

Isolation Forest anomaly detection, trained separately per vessel type.

Logic:
  - For each vessel type (cargo, tanker, fishing, etc.) with enough vessels,
    train an Isolation Forest on the feature columns.
  - Vessels that are anomalous *relative to their peer type* score high.
  - A tug behaving like other tugs scores low even if its absolute numbers
    look weird to a global model.
  - Types with too few vessels (<MIN_TYPE_SIZE) fall back to a global model.

Returns an isolation_score 0–1 per vessel (1 = most anomalous).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler


# ── Features fed to the model ─────────────────────────────────────────────────
# These must all exist in the feature matrix from builder.py
ISOLATION_FEATURES = [
    "mean_sog",
    "max_sog",
    "ping_count",
    "speed_variance_score",
    "heading_erraticism_score",
    "reversal_score",
    "gap_score",
    "max_gap_hours",
    "port_loiter_score",
    "open_ocean_stop_score",
    "eez_hug_score",
]

# Minimum vessels of a type needed to train a type-specific model
MIN_TYPE_SIZE = 20

# Isolation Forest hyperparameters
N_ESTIMATORS   = 100
CONTAMINATION  = 0.05   # assume ~5% of vessels are anomalous
RANDOM_STATE   = 42


# ── Core ──────────────────────────────────────────────────────────────────────

def _get_feature_array(df: pd.DataFrame) -> np.ndarray:
    """Extract and impute feature columns as a numpy array."""
    cols = [c for c in ISOLATION_FEATURES if c in df.columns]
    X = df[cols].copy().fillna(0).to_numpy(dtype=float)
    return X


def _train_and_score(X: np.ndarray) -> np.ndarray:
    """
    Fit an Isolation Forest on X, return anomaly scores in [0, 1].
    sklearn returns decision_function scores where more negative = more anomalous.
    We flip and normalise to [0, 1].
    """
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    clf = IsolationForest(
        n_estimators=N_ESTIMATORS,
        contamination=CONTAMINATION,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    clf.fit(X_scaled)

    # decision_function: negative = anomalous, positive = normal
    raw_scores = clf.decision_function(X_scaled)

    # Flip so anomalous = high, then normalise to [0, 1]
    flipped = -raw_scores
    min_s, max_s = flipped.min(), flipped.max()
    if max_s == min_s:
        return np.zeros(len(flipped))
    return (flipped - min_s) / (max_s - min_s)


# ── Public API ────────────────────────────────────────────────────────────────

def compute_isolation_scores(feature_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Isolation Forest anomaly scores per vessel type.

    Args:
        feature_matrix: Output of builder.build_feature_matrix().

    Returns:
        feature_matrix with a new column: isolation_score (0–1).
    """
    fm = feature_matrix.copy()
    fm["isolation_score"] = 0.0

    vessel_types = fm["vessel_type"].fillna("unknown").unique()
    type_counts  = fm["vessel_type"].fillna("unknown").value_counts()

    # Vessels that fall back to global model
    small_type_mask = fm["vessel_type"].fillna("unknown").map(
        lambda t: type_counts.get(t, 0) < MIN_TYPE_SIZE
    )

    print(f"  isolation forest — {len(vessel_types)} vessel types")

    # ── Per-type models ───────────────────────────────────────────────────────
    for vtype in vessel_types:
        mask = fm["vessel_type"].fillna("unknown") == vtype
        n    = mask.sum()

        if n < MIN_TYPE_SIZE:
            continue  # handled by global model below

        X      = _get_feature_array(fm[mask])
        scores = _train_and_score(X)
        fm.loc[mask, "isolation_score"] = scores
        print(f"    {vtype:<20} {n:>5} vessels  scored")

    # ── Global fallback for small types ───────────────────────────────────────
    n_small = small_type_mask.sum()
    if n_small > 0:
        X_global      = _get_feature_array(fm[small_type_mask])
        global_scores = _train_and_score(X_global)
        fm.loc[small_type_mask, "isolation_score"] = global_scores
        print(f"    global fallback      {n_small:>5} vessels  scored")

    fm["isolation_score"] = fm["isolation_score"].round(4)
    return fm