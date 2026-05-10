"""
lostping/report/json_report.py

Serialises the flagged vessel list + all associated events into a structured
JSON report at outputs/reports/flagged.json.

Structure:
{
  "generated_at": "2024-01-03T...",
  "source_file":  "ais-2024-01-03.csv",
  "vessels_analysed": 4621,
  "vessels_flagged":  34,
  "risk_threshold":   50,
  "vessels": [
    {
      "mmsi":            "368120080",
      "vessel_name":     "JET 1",
      "vessel_type":     "50",
      "risk_score":      73.3,
      "isolation_score": 1.0,
      "loiter_score":    1.0,
      "zscore_score":    0.45,
      "gap_score":       0.82,
      "trajectory_score":0.91,
      "spatial_score":   0.72,
      "reason":          "AIS dark 14.8h; erratic speed; ...",
      "ping_count":      43,
      "first_seen":      "2024-01-03 00:12:00",
      "last_seen":       "2024-01-03 23:45:00",
      "track_bounds":    {"lat_min": ..., "lat_max": ..., "lon_min": ..., "lon_max": ...},
      "gap_events":      [...],
      "spatial_events":  [...],
      "loiter_clusters": [...]
    },
    ...
  ]
}
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from dataclasses import asdict

import pandas as pd

from lostping.models.loiter import compute_loiter_scores, LoiterResult
from lostping.features.gaps import compute_gaps_all
from lostping.features.spatial import compute_spatial_scores


# ── Score columns to include per vessel ───────────────────────────────────────
SCORE_COLS = [
    "risk_score", "isolation_score", "loiter_score", "zscore_score",
    "gap_score", "trajectory_score", "spatial_score",
    "speed_variance_score", "heading_erraticism_score", "reversal_score",
    "port_loiter_score", "open_ocean_stop_score", "eez_hug_score",
    "max_gap_hours", "gap_count", "cluster_count", "max_cluster_ratio",
    "reason",
]

METADATA_COLS = [
    "vessel_name", "vessel_type", "ping_count",
    "first_seen", "last_seen",
    "lat_min", "lat_max", "lon_min", "lon_max",
    "mean_sog", "max_sog",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_float(val) -> float | None:
    try:
        f = float(val)
        return round(f, 4) if not pd.isna(f) else None
    except (TypeError, ValueError):
        return None


def _safe_int(val) -> int | None:
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


def _vessel_dict(row: pd.Series) -> dict:
    d = {"mmsi": str(row["mmsi"])}

    for col in METADATA_COLS:
        val = row.get(col)
        if col in ("ping_count",):
            d[col] = _safe_int(val)
        elif col in ("vessel_name", "vessel_type", "first_seen", "last_seen"):
            d[col] = str(val) if val is not None else None
        else:
            d[col] = _safe_float(val)

    d["track_bounds"] = {
        "lat_min": _safe_float(row.get("lat_min")),
        "lat_max": _safe_float(row.get("lat_max")),
        "lon_min": _safe_float(row.get("lon_min")),
        "lon_max": _safe_float(row.get("lon_max")),
    }
    # Remove duplicate top-level bound keys
    for k in ("lat_min", "lat_max", "lon_min", "lon_max"):
        d.pop(k, None)

    for col in SCORE_COLS:
        val = row.get(col)
        if col in ("reason",):
            d[col] = str(val) if val is not None else ""
        elif col in ("gap_count", "cluster_count"):
            d[col] = _safe_int(val)
        else:
            d[col] = _safe_float(val)

    # Placeholder lists — populated by _attach_events
    d["gap_events"]      = []
    d["spatial_events"]  = []
    d["loiter_clusters"] = []

    return d


def _attach_events(
    vessels: list[dict],
    df: pd.DataFrame,
) -> list[dict]:
    """
    Re-run gap, spatial, loiter extractors on the flagged subset only
    and attach their event lists to each vessel dict.
    """
    flagged_mmsis = {v["mmsi"] for v in vessels}
    subset = df[df["mmsi"].isin(flagged_mmsis)].copy()

    if subset.empty:
        return vessels

    # ── Gap events ────────────────────────────────────────────────────────────
    # compute_gaps_all doesn't expose events directly; import compute_gaps
    from lostping.features.gaps import compute_gaps

    gap_events_by_mmsi: dict[str, list] = {}
    for mmsi, group in subset.groupby("mmsi"):
        result = compute_gaps(group)
        gap_events_by_mmsi[str(mmsi)] = result.get("gap_events", [])

    # ── Spatial events ────────────────────────────────────────────────────────
    spat_results = compute_spatial_scores(subset)
    spat_events_by_mmsi = {
        r.mmsi: [asdict(e) for e in r.events]
        for r in spat_results
    }

    # ── Loiter clusters ───────────────────────────────────────────────────────
    loiter_results = compute_loiter_scores(subset)
    loiter_by_mmsi = {
        r.mmsi: [asdict(c) for c in r.clusters]
        for r in loiter_results
    }

    # ── Attach ────────────────────────────────────────────────────────────────
    for v in vessels:
        mmsi = v["mmsi"]
        v["gap_events"]      = gap_events_by_mmsi.get(mmsi, [])
        v["spatial_events"]  = spat_events_by_mmsi.get(mmsi, [])
        v["loiter_clusters"] = loiter_by_mmsi.get(mmsi, [])

    return vessels


# ── Public API ────────────────────────────────────────────────────────────────

def write_json_report(
    flagged: pd.DataFrame,
    df: pd.DataFrame,
    out_path: str = "outputs/reports/flagged.json",
    source_file: str = "",
    total_vessels: int = 0,
    risk_threshold: float = 50.0,
) -> str:
    """
    Write a structured JSON report for all flagged vessels.

    Args:
        flagged:        Output of scorer.get_flagged() — scored + filtered DataFrame.
        df:             Full cleaned AIS DataFrame (needed to pull event details).
        out_path:       Where to write the JSON file.
        source_file:    Name of the source CSV (for metadata).
        total_vessels:  Total vessels analysed before filtering.
        risk_threshold: The threshold used to produce the flagged list.

    Returns:
        Absolute path of the written file.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    vessels = [_vessel_dict(row) for _, row in flagged.iterrows()]
    vessels = _attach_events(vessels, df)

    report = {
        "generated_at":     datetime.now(timezone.utc).isoformat(),
        "source_file":      source_file,
        "vessels_analysed": int(total_vessels),
        "vessels_flagged":  len(vessels),
        "risk_threshold":   float(risk_threshold),
        "vessels":          vessels,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"wrote {len(vessels)} flagged vessels → {out_path}")
    return os.path.abspath(out_path)