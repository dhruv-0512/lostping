"""
gaps.py — detect and score AIS transmission gaps per vessel

A "gap" is a period where a vessel stops broadcasting — either because
it turned off its transponder (suspicious) or lost satellite coverage
(innocent). Context determines which: gaps near ports or in known
satellite dead zones are less suspicious than gaps mid-ocean near
known ship-to-ship transfer zones.

Output per vessel:
    gap_count       — number of gaps above threshold
    max_gap_hours   — duration of the longest single gap
    total_gap_hours — cumulative silence time
    gap_score       — normalised 0–1 suspicion score
    gap_events      — list of individual gap records (for the report)
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict


# --- thresholds ---------------------------------------------------------------

# A gap shorter than this is normal (satellite coverage holes, brief shutdown)
MIN_GAP_HOURS = 3.0

# A gap longer than this is heavily weighted even if only one occurs
SEVERE_GAP_HOURS = 8.0

# Cap for normalisation — gaps beyond this all score 1.0
SCORE_CAP_HOURS = 24.0


# --- known ship-to-ship (STS) transfer zones ----------------------------------
# (lon_min, lat_min, lon_max, lat_max, label)
# Sources: UN Panel of Experts reports, OFAC advisories
STS_ZONES = [
    (-98.0, 18.0, -84.0, 24.0, "gulf_of_mexico_offshore"),
    (52.0,  24.0,  57.0, 27.0, "strait_of_hormuz"),
    (36.0,  11.0,  45.0, 15.0, "gulf_of_aden"),
    (119.0, 28.0, 124.0, 33.0, "east_china_sea"),
    (27.0,  36.0,  32.0, 42.0, "black_sea"),
]


@dataclass
class GapEvent:
    start:           str    # ISO timestamp
    end:             str    # ISO timestamp
    gap_hours:       float
    start_lat:       float
    start_lon:       float
    end_lat:         float
    end_lon:         float
    near_sts_zone:   bool
    sts_zone_label:  str | None


def compute_gaps(vessel_df: pd.DataFrame) -> dict:
    """
    Compute gap features for a single vessel's track.

    Args:
        vessel_df: DataFrame for one MMSI, sorted by timestamp.

    Returns:
        dict with gap_count, max_gap_hours, total_gap_hours,
        gap_score, gap_events.
    """
    if len(vessel_df) < 2:
        return _empty_result()

    df = vessel_df.sort_values("timestamp").reset_index(drop=True)

    # time delta between consecutive pings in hours
    deltas = df["timestamp"].diff().dt.total_seconds() / 3600
    gap_mask = deltas > MIN_GAP_HOURS

    events = []
    for idx in df.index[gap_mask]:
        prev = df.loc[idx - 1]
        curr = df.loc[idx]
        gap_h = deltas[idx]

        near_sts, sts_label = _check_sts_zone(
            prev["lon"], prev["lat"]
        )

        events.append(GapEvent(
            start          = str(prev["timestamp"]),
            end            = str(curr["timestamp"]),
            gap_hours      = round(gap_h, 2),
            start_lat      = round(float(prev["lat"]), 5),
            start_lon      = round(float(prev["lon"]), 5),
            end_lat        = round(float(curr["lat"]), 5),
            end_lon        = round(float(curr["lon"]), 5),
            near_sts_zone  = near_sts,
            sts_zone_label = sts_label,
        ))

    if not events:
        return _empty_result()

    gap_hours_list  = [e.gap_hours for e in events]
    max_gap         = max(gap_hours_list)
    total_gap       = sum(gap_hours_list)
    sts_count       = sum(1 for e in events if e.near_sts_zone)

    score = _compute_score(max_gap, len(events), sts_count)

    return {
        "gap_count":       len(events),
        "max_gap_hours":   round(max_gap, 2),
        "total_gap_hours": round(total_gap, 2),
        "gap_score":       round(score, 4),
        "gap_events":      [asdict(e) for e in events],
    }


def compute_gaps_all(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run compute_gaps over every vessel in the DataFrame.

    Returns a summary DataFrame with one row per MMSI.
    """
    results = []
    for mmsi, group in df.groupby("mmsi"):
        res = compute_gaps(group)
        res["mmsi"] = mmsi
        results.append(res)

    summary = pd.DataFrame(results)

    # drop gap_events list column for the summary table
    # (kept inside full result dict for reporting)
    return summary[["mmsi", "gap_count", "max_gap_hours",
                     "total_gap_hours", "gap_score"]]


# --- internals ----------------------------------------------------------------

def _check_sts_zone(lon: float, lat: float) -> tuple[bool, str | None]:
    """Return (is_near_sts, zone_label) for a coordinate."""
    for lon_min, lat_min, lon_max, lat_max, label in STS_ZONES:
        if lon_min <= lon <= lon_max and lat_min <= lat <= lat_max:
            return True, label
    return False, None


def _compute_score(
    max_gap: float,
    gap_count: int,
    sts_count: int,
) -> float:
    """
    Combine gap features into a 0–1 suspicion score.

    Logic:
      - base score from max_gap normalised to SCORE_CAP_HOURS
      - bonus for multiple gaps (each additional gap adds weight)
      - bonus for gaps near STS zones (major red flag)
    """
    base       = min(max_gap / SCORE_CAP_HOURS, 1.0)
    multi_bonus = min((gap_count - 1) * 0.05, 0.20)   # up to +0.20
    sts_bonus   = min(sts_count * 0.15, 0.30)          # up to +0.30

    # severe gap bump
    if max_gap >= SEVERE_GAP_HOURS:
        base = min(base * 1.3, 1.0)

    return min(base + multi_bonus + sts_bonus, 1.0)


def _empty_result() -> dict:
    return {
        "gap_count":       0,
        "max_gap_hours":   0.0,
        "total_gap_hours": 0.0,
        "gap_score":       0.0,
        "gap_events":      [],
    }