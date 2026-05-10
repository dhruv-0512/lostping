"""
lostping/features/trajectory.py

Detects erratic vessel movement:
  - Speed variance (SOG spikes/drops relative to vessel-type baseline)
  - Heading erraticism (circular variance of COG)
  - Sudden speed reversals (deceleration events)

Returns a trajectory_score 0–1 per vessel + structured events list.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field


# ── Vessel-type speed baselines (knots) ───────────────────────────────────────
# (expected_mean, max_normal) — anything beyond max_normal is flagged
SPEED_BASELINES: dict[str, tuple[float, float]] = {
    "cargo":      (12.0, 22.0),
    "tanker":     (11.0, 18.0),
    "passenger":  (16.0, 28.0),
    "fishing":    (5.0,  12.0),
    "tug":        (6.0,  14.0),
    "pleasure":   (10.0, 30.0),
    "military":   (15.0, 35.0),
    "other":      (8.0,  25.0),
    "unknown":    (8.0,  25.0),
}

# Minimum pings needed to compute meaningful trajectory stats
MIN_PINGS = 10

# Speed-reversal: flag if speed drops by this many knots within one ping interval
REVERSAL_THRESHOLD_KNOTS = 8.0

# Heading change threshold per ping (degrees) to count as erratic step
HEADING_STEP_THRESHOLD_DEG = 45.0


# ── Output types ──────────────────────────────────────────────────────────────

@dataclass
class TrajectoryEvent:
    mmsi: str
    event_type: str          # "speed_spike" | "speed_reversal" | "erratic_heading"
    timestamp: str           # ISO string of the triggering ping
    lat: float
    lon: float
    detail: dict = field(default_factory=dict)


@dataclass
class TrajectoryResult:
    mmsi: str
    trajectory_score: float                    # 0–1
    speed_variance_score: float                # 0–1 sub-score
    heading_erraticism_score: float            # 0–1 sub-score
    reversal_score: float                      # 0–1 sub-score
    events: list[TrajectoryEvent] = field(default_factory=list)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _circular_variance(angles_deg: np.ndarray) -> float:
    """
    Circular variance of a set of angles in [0, 360).
    Returns 0 (no variance) → 1 (maximally spread).
    """
    if len(angles_deg) < 2:
        return 0.0
    rad = np.deg2rad(angles_deg)
    R = np.abs(np.mean(np.exp(1j * rad)))   # mean resultant length
    return float(1.0 - R)                   # 0 = all same direction, 1 = random


def _angular_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Smallest signed difference between consecutive headings (degrees)."""
    diff = (b - a + 180) % 360 - 180
    return np.abs(diff)


def _sigmoid_scale(x: float, midpoint: float, steepness: float = 6.0) -> float:
    """Map x to (0, 1) with a sigmoid centred on midpoint."""
    return float(1 / (1 + np.exp(-steepness * (x - midpoint))))


# ── Per-vessel scoring ────────────────────────────────────────────────────────

def _score_vessel(group: pd.DataFrame, vessel_type: str) -> TrajectoryResult:
    mmsi = str(group["mmsi"].iloc[0])
    events: list[TrajectoryEvent] = []

    baseline_mean, baseline_max = SPEED_BASELINES.get(vessel_type, SPEED_BASELINES["unknown"])

    sog = group["sog"].to_numpy(dtype=float)
    cog = group["cog"].to_numpy(dtype=float)
    timestamps = group["timestamp"].astype(str).to_numpy()
    lats = group["lat"].to_numpy(dtype=float)
    lons = group["lon"].to_numpy(dtype=float)

    # ── 1. Speed variance score ───────────────────────────────────────────────
    sog_std = float(np.std(sog))
    # Normalise by baseline_mean so a tanker and a pleasure craft are comparable
    normalised_std = sog_std / max(baseline_mean, 1.0)
    speed_variance_score = min(_sigmoid_scale(normalised_std, midpoint=1.0), 1.0)

    # Flag individual speed spikes above vessel-type max
    spike_mask = sog > baseline_max
    for idx in np.where(spike_mask)[0]:
        events.append(TrajectoryEvent(
            mmsi=mmsi,
            event_type="speed_spike",
            timestamp=timestamps[idx],
            lat=float(lats[idx]),
            lon=float(lons[idx]),
            detail={"sog_knots": round(float(sog[idx]), 1), "type_max_knots": baseline_max},
        ))

    # ── 2. Speed reversal score ───────────────────────────────────────────────
    sog_diffs = np.abs(np.diff(sog))
    reversal_mask = sog_diffs >= REVERSAL_THRESHOLD_KNOTS
    reversal_count = int(reversal_mask.sum())
    # Normalise: 3+ reversals in a track = clearly suspicious
    reversal_score = min(reversal_count / 3.0, 1.0)

    for idx in np.where(reversal_mask)[0]:
        events.append(TrajectoryEvent(
            mmsi=mmsi,
            event_type="speed_reversal",
            timestamp=timestamps[idx + 1],
            lat=float(lats[idx + 1]),
            lon=float(lons[idx + 1]),
            detail={
                "sog_before": round(float(sog[idx]), 1),
                "sog_after":  round(float(sog[idx + 1]), 1),
                "delta_knots": round(float(sog_diffs[idx]), 1),
            },
        ))

    # ── 3. Heading erraticism score ───────────────────────────────────────────
    # Only use valid COG readings (0–360; 511 = not available)
    valid_cog_mask = (cog >= 0) & (cog <= 360)
    valid_cog = cog[valid_cog_mask]

    if len(valid_cog) >= 4:
        circ_var = _circular_variance(valid_cog)
        # Also count ping-to-ping jumps above threshold
        step_diffs = _angular_diff(valid_cog[:-1], valid_cog[1:])
        erratic_steps = int((step_diffs >= HEADING_STEP_THRESHOLD_DEG).sum())
        erratic_ratio = erratic_steps / max(len(step_diffs), 1)

        # Blend circular variance (global shape) + step ratio (local jumpiness)
        heading_erraticism_score = min(0.6 * circ_var + 0.4 * erratic_ratio, 1.0)

        # Flag clusters of erratic headings
        step_mask = step_diffs >= HEADING_STEP_THRESHOLD_DEG
        # Re-index back into full arrays (valid_cog_mask offsets things, keep it simple:
        # use positions within valid_cog)
        valid_indices = np.where(valid_cog_mask)[0]
        for rel_idx in np.where(step_mask)[0]:
            abs_idx = int(valid_indices[rel_idx + 1])
            events.append(TrajectoryEvent(
                mmsi=mmsi,
                event_type="erratic_heading",
                timestamp=timestamps[abs_idx],
                lat=float(lats[abs_idx]),
                lon=float(lons[abs_idx]),
                detail={
                    "heading_before_deg": round(float(valid_cog[rel_idx]), 1),
                    "heading_after_deg":  round(float(valid_cog[rel_idx + 1]), 1),
                    "delta_deg": round(float(step_diffs[rel_idx]), 1),
                },
            ))
    else:
        heading_erraticism_score = 0.0

    # ── 4. Composite trajectory score ─────────────────────────────────────────
    # Weights: speed variance 40%, heading 40%, reversals 20%
    trajectory_score = min(
        0.40 * speed_variance_score
        + 0.40 * heading_erraticism_score
        + 0.20 * reversal_score,
        1.0,
    )

    return TrajectoryResult(
        mmsi=mmsi,
        trajectory_score=round(trajectory_score, 4),
        speed_variance_score=round(speed_variance_score, 4),
        heading_erraticism_score=round(heading_erraticism_score, 4),
        reversal_score=round(reversal_score, 4),
        events=events,
    )


# ── Public API ────────────────────────────────────────────────────────────────

def compute_trajectory_scores(df: pd.DataFrame) -> list[TrajectoryResult]:
    """
    Compute trajectory anomaly scores for every vessel in df.

    Args:
        df: Cleaned AIS DataFrame from cleaner.py.
            Required columns: mmsi, timestamp, lat, lon, sog, cog, vessel_type

    Returns:
        List of TrajectoryResult, one per vessel with >= MIN_PINGS pings.
    """
    required = {"mmsi", "timestamp", "lat", "lon", "sog", "cog", "vessel_type"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"trajectory.py: missing columns {missing}")

    df = df.sort_values(["mmsi", "timestamp"]).copy()

    results: list[TrajectoryResult] = []

    for mmsi, group in df.groupby("mmsi", sort=False):
        if len(group) < MIN_PINGS:
            continue

        mode = group["vessel_type"].mode()
        vessel_type = str(mode.iloc[0]).lower() if not mode.empty else "unknown"
        result = _score_vessel(group, vessel_type)
        results.append(result)

    return results


def trajectory_scores_to_df(results: list[TrajectoryResult]) -> pd.DataFrame:
    """
    Flatten TrajectoryResult list into a tidy DataFrame for builder.py.

    Columns: mmsi, trajectory_score, speed_variance_score,
             heading_erraticism_score, reversal_score, trajectory_event_count
    """
    rows = [
        {
            "mmsi": r.mmsi,
            "trajectory_score": r.trajectory_score,
            "speed_variance_score": r.speed_variance_score,
            "heading_erraticism_score": r.heading_erraticism_score,
            "reversal_score": r.reversal_score,
            "trajectory_event_count": len(r.events),
        }
        for r in results
    ]
    return pd.DataFrame(rows)