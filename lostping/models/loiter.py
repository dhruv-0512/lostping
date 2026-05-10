"""
lostping/models/loiter.py

Detects vessels loitering — staying in or repeatedly returning to the same
small patch of ocean.

Method:
  - For each vessel, run DBSCAN on (lat, lon) ping coordinates.
  - A vessel with a large dense cluster relative to its total pings is loitering.
  - Cluster centre, radius, and dwell time are recorded as events.
  - Works in open ocean (no port proximity filter) — catches behaviour
    that spatial.py misses because it's not near any known reference point.

Returns a loiter_score 0–1 per vessel + cluster event list.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from sklearn.cluster import DBSCAN


# ── DBSCAN parameters ─────────────────────────────────────────────────────────
# epsilon in degrees (~0.05 deg ≈ 3 nautical miles at mid-latitudes)
DBSCAN_EPS_DEG   = 0.05
DBSCAN_MIN_PINGS = 8     # minimum pings to form a cluster

# A cluster must contain at least this fraction of the vessel's total pings
# to be considered a loiter (filters out brief stops)
LOITER_CLUSTER_RATIO = 0.20

# Minimum pings per vessel to bother running DBSCAN
MIN_VESSEL_PINGS = 15


# ── Output types ──────────────────────────────────────────────────────────────

@dataclass
class LoiterCluster:
    mmsi: str
    cluster_id: int
    centre_lat: float
    centre_lon: float
    radius_nm: float          # approximate radius of cluster in nautical miles
    ping_count: int           # pings in this cluster
    cluster_ratio: float      # fraction of vessel's total pings in cluster
    first_seen: str
    last_seen: str
    dwell_hours: float        # time from first to last ping in cluster


@dataclass
class LoiterResult:
    mmsi: str
    loiter_score: float
    cluster_count: int
    max_cluster_ratio: float
    clusters: list[LoiterCluster] = field(default_factory=list)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _deg_to_nm(deg: float) -> float:
    """Approximate degrees to nautical miles (at mid-latitudes)."""
    return deg * 60.0


def _score_from_clusters(clusters: list[LoiterCluster], total_pings: int) -> float:
    """
    Compute loiter_score 0–1 from cluster list.

    Factors:
      - Max cluster ratio (what fraction of pings are in the densest cluster)
      - Number of distinct clusters (multiple loiter spots = more suspicious)
      - Dwell time of the largest cluster
    """
    if not clusters:
        return 0.0

    max_ratio  = max(c.cluster_ratio for c in clusters)
    n_clusters = len(clusters)
    max_dwell  = max(c.dwell_hours for c in clusters)

    # Base score from cluster density ratio
    base = min(max_ratio / 0.5, 1.0)   # 50% of pings in one spot → saturates

    # Bonus for multiple loiter locations (up to +0.20)
    multi_bonus = min((n_clusters - 1) * 0.10, 0.20)

    # Bonus for long dwell time (up to +0.20, saturates at 6h)
    dwell_bonus = min(max_dwell / 6.0, 1.0) * 0.20

    return min(base + multi_bonus + dwell_bonus, 1.0)


# ── Per-vessel scoring ────────────────────────────────────────────────────────

def _score_vessel(group: pd.DataFrame) -> LoiterResult:
    mmsi       = str(group["mmsi"].iloc[0])
    total_n    = len(group)
    coords     = group[["lat", "lon"]].to_numpy(dtype=float)
    timestamps = pd.to_datetime(group["timestamp"])

    db = DBSCAN(eps=DBSCAN_EPS_DEG, min_samples=DBSCAN_MIN_PINGS, algorithm="ball_tree", metric="haversine")
    # DBSCAN haversine expects radians
    coords_rad = np.deg2rad(coords)
    labels     = db.fit_predict(coords_rad)

    clusters: list[LoiterCluster] = []
    unique_labels = set(labels) - {-1}   # -1 = noise

    for cid in unique_labels:
        mask        = labels == cid
        cluster_pts = coords[mask]
        cluster_ts  = timestamps[mask]
        n_cluster   = mask.sum()
        ratio       = n_cluster / total_n

        if ratio < LOITER_CLUSTER_RATIO:
            continue   # too small to call a loiter

        centre_lat = float(cluster_pts[:, 0].mean())
        centre_lon = float(cluster_pts[:, 1].mean())

        # Approximate radius: mean distance from centre
        lat_diffs = cluster_pts[:, 0] - centre_lat
        lon_diffs = cluster_pts[:, 1] - centre_lon
        dists_deg = np.sqrt(lat_diffs**2 + lon_diffs**2)
        radius_nm = float(_deg_to_nm(dists_deg.mean()))

        first_ts   = cluster_ts.min()
        last_ts    = cluster_ts.max()
        dwell_hrs  = (last_ts - first_ts).total_seconds() / 3600.0

        clusters.append(LoiterCluster(
            mmsi=mmsi,
            cluster_id=int(cid),
            centre_lat=round(centre_lat, 5),
            centre_lon=round(centre_lon, 5),
            radius_nm=round(radius_nm, 2),
            ping_count=int(n_cluster),
            cluster_ratio=round(ratio, 4),
            first_seen=str(first_ts),
            last_seen=str(last_ts),
            dwell_hours=round(dwell_hrs, 2),
        ))

    loiter_score = _score_from_clusters(clusters, total_n)

    return LoiterResult(
        mmsi=mmsi,
        loiter_score=round(loiter_score, 4),
        cluster_count=len(clusters),
        max_cluster_ratio=round(max((c.cluster_ratio for c in clusters), default=0.0), 4),
        clusters=clusters,
    )


# ── Public API ────────────────────────────────────────────────────────────────

def compute_loiter_scores(df: pd.DataFrame) -> list[LoiterResult]:
    """
    Run DBSCAN loiter detection on every vessel in df.

    Args:
        df: Cleaned AIS DataFrame. Required columns: mmsi, timestamp, lat, lon.

    Returns:
        List of LoiterResult, one per vessel with >= MIN_VESSEL_PINGS pings.
    """
    required = {"mmsi", "timestamp", "lat", "lon"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"loiter.py: missing columns {missing}")

    df = df.sort_values(["mmsi", "timestamp"]).copy()

    results: list[LoiterResult] = []
    for _, group in df.groupby("mmsi", sort=False):
        if len(group) < MIN_VESSEL_PINGS:
            continue
        results.append(_score_vessel(group))

    return results


def loiter_scores_to_df(results: list[LoiterResult]) -> pd.DataFrame:
    """
    Flatten LoiterResult list into a tidy DataFrame for merging.

    Columns: mmsi, loiter_score, cluster_count, max_cluster_ratio
    """
    return pd.DataFrame([
        {
            "mmsi":               r.mmsi,
            "loiter_score":       r.loiter_score,
            "cluster_count":      r.cluster_count,
            "max_cluster_ratio":  r.max_cluster_ratio,
        }
        for r in results
    ])