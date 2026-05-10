"""
lostping/features/spatial.py

Spatial anomaly detection:
  - Port proximity: vessel loitering near a major port without docking
  - Anchorage avoidance: vessel passes close to port but doesn't stop
  - EEZ boundary hugging: vessel tracks along a boundary (common in illegal fishing)
  - Open-ocean stop: vessel goes near-zero speed far from any port or anchorage

Returns a spatial_score 0–1 per vessel + structured events list.
"""

from __future__ import annotations

import math
import numpy as np
import pandas as pd
from dataclasses import dataclass, field


# ── Major ports (Gulf of Mexico + surrounding, covers Marine Cadastre Zone 1) ─
# (name, lat, lon, radius_nm) — radius = normal operational zone
MAJOR_PORTS: list[tuple[str, float, float, float]] = [
    ("Houston",           29.7355, -95.0155, 15.0),
    ("New Orleans",       29.9511, -90.0715, 12.0),
    ("Tampa",             27.9506, -82.4572, 10.0),
    ("Mobile",            30.6954, -88.0430, 10.0),
    ("Corpus Christi",    27.8006, -97.3964, 10.0),
    ("Beaumont",          29.9552, -94.1044,  8.0),
    ("Pascagoula",        30.3460, -88.5561,  8.0),
    ("Port Fourchon",     29.1131, -90.2043,  8.0),
    ("Pensacola",         30.4013, -87.2169,  8.0),
    ("Veracruz",          19.2000, -96.1333, 10.0),
    ("Tampico",           22.2000, -97.8500,  8.0),
    ("Coatzacoalcos",     18.1500, -94.4333,  8.0),
    ("Havana",            23.1370, -82.3580, 10.0),
    ("Miami",             25.7742, -80.1936, 12.0),
    ("Port Everglades",   26.0820, -80.1180, 10.0),
    ("Jacksonville",      30.3322, -81.6557, 10.0),
    ("Savannah",          32.0835, -81.0998, 10.0),
    ("Charleston",        32.7765, -79.9311, 10.0),
]

# ── Known anchorage / waiting areas (vessels legitimately slow here) ───────────
ANCHORAGE_ZONES: list[tuple[str, float, float, float]] = [
    ("Houston Ship Channel Anchorage", 29.6200, -94.9800, 5.0),
    ("Mississippi River Anchorage",    28.9000, -89.4000, 8.0),
    ("Tampa Bay Anchorage",            27.6000, -82.6500, 6.0),
    ("Gulf of Mexico Deepwater Anch.", 27.0000, -90.0000, 10.0),
]

# ── EEZ boundary segments (simplified as lat/lon bands) ──────────────────────
# Each entry: (name, lat_min, lat_max, lon_min, lon_max, boundary_type)
# "hug" = vessel tracking along this band is suspicious
EEZ_BOUNDARIES: list[dict] = [
    # US–Mexico maritime boundary in Gulf
    {"name": "US-Mexico Gulf EEZ",    "lat_min": 25.5, "lat_max": 26.5, "lon_min": -97.5, "lon_max": -90.0, "type": "international"},
    # Cuba EEZ northern edge
    {"name": "Cuba EEZ North",         "lat_min": 23.0, "lat_max": 24.5, "lon_min": -85.0, "lon_max": -74.0, "type": "international"},
    # Florida Straits
    {"name": "Florida Straits",        "lat_min": 24.0, "lat_max": 25.5, "lon_min": -81.0, "lon_max": -79.5, "type": "chokepoint"},
    # Yucatan Channel
    {"name": "Yucatan Channel",        "lat_min": 21.0, "lat_max": 22.5, "lon_min": -87.5, "lon_max": -85.5, "type": "chokepoint"},
]

# Thresholds
NEAR_PORT_NM          = 20.0   # within this = "near port"
PORT_LOITER_SPEED_KT  = 2.0    # below this while near port = loitering/waiting
OPEN_OCEAN_MIN_NM     = 50.0   # farther than this from any port = open ocean
OPEN_OCEAN_STOP_KT    = 1.0    # below this in open ocean = suspicious stop
EEZ_HUG_TOLERANCE_DEG = 0.4    # lat/lon degrees — how close to boundary counts
EEZ_HUG_MIN_PINGS     = 5      # need at least this many consecutive pings near boundary
MIN_PINGS             = 10


# ── Geometry helpers ──────────────────────────────────────────────────────────

def _haversine_nm(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in nautical miles."""
    R = 3440.065  # Earth radius in nm
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))


def _nearest_port_dist(lat: float, lon: float) -> tuple[str, float]:
    """Return (port_name, distance_nm) for the closest major port."""
    best_name, best_dist = "none", float("inf")
    for name, plat, plon, _ in MAJOR_PORTS:
        d = _haversine_nm(lat, lon, plat, plon)
        if d < best_dist:
            best_dist, best_name = d, name
    return best_name, best_dist


def _in_anchorage(lat: float, lon: float) -> bool:
    for _, alat, alon, radius in ANCHORAGE_ZONES:
        if _haversine_nm(lat, lon, alat, alon) <= radius:
            return True
    return False


def _near_eez(lat: float, lon: float) -> list[str]:
    """Return names of EEZ boundaries this point is close to."""
    matched = []
    for eez in EEZ_BOUNDARIES:
        lat_near = abs(lat - np.clip(lat, eez["lat_min"], eez["lat_max"])) < EEZ_HUG_TOLERANCE_DEG
        lon_near = abs(lon - np.clip(lon, eez["lon_min"], eez["lon_max"])) < EEZ_HUG_TOLERANCE_DEG
        in_lon_band = eez["lon_min"] - EEZ_HUG_TOLERANCE_DEG <= lon <= eez["lon_max"] + EEZ_HUG_TOLERANCE_DEG
        in_lat_band = eez["lat_min"] - EEZ_HUG_TOLERANCE_DEG <= lat <= eez["lat_max"] + EEZ_HUG_TOLERANCE_DEG
        if (lat_near or lon_near) and in_lon_band and in_lat_band:
            matched.append(eez["name"])
    return matched


# ── Output types ──────────────────────────────────────────────────────────────

@dataclass
class SpatialEvent:
    mmsi: str
    event_type: str      # "port_loiter" | "open_ocean_stop" | "eez_hugging" | "anchorage_avoidance"
    timestamp: str
    lat: float
    lon: float
    detail: dict = field(default_factory=dict)


@dataclass
class SpatialResult:
    mmsi: str
    spatial_score: float
    port_loiter_score: float
    open_ocean_stop_score: float
    eez_hug_score: float
    events: list[SpatialEvent] = field(default_factory=list)


# ── Per-vessel scoring ────────────────────────────────────────────────────────

def _score_vessel(group: pd.DataFrame) -> SpatialResult:
    mmsi = str(group["mmsi"].iloc[0])
    events: list[SpatialEvent] = []

    lats       = group["lat"].to_numpy(dtype=float)
    lons       = group["lon"].to_numpy(dtype=float)
    sogs       = group["sog"].to_numpy(dtype=float)
    timestamps = group["timestamp"].astype(str).to_numpy()
    n          = len(group)

    port_loiter_pings    = 0
    open_ocean_stop_pings = 0
    eez_hug_streak        = 0
    max_eez_streak        = 0
    last_eez_names: list[str] = []

    for i in range(n):
        lat, lon, sog, ts = lats[i], lons[i], sogs[i], timestamps[i]
        port_name, dist_nm = _nearest_port_dist(lat, lon)
        in_anch = _in_anchorage(lat, lon)
        eez_names = _near_eez(lat, lon)

        # ── Port loitering: slow, near port, not in anchorage ─────────────────
        if dist_nm <= NEAR_PORT_NM and sog <= PORT_LOITER_SPEED_KT and not in_anch:
            port_loiter_pings += 1
            if port_loiter_pings in (1, 10, 30):   # event at onset + milestones
                events.append(SpatialEvent(
                    mmsi=mmsi, event_type="port_loiter",
                    timestamp=ts, lat=lat, lon=lon,
                    detail={"nearest_port": port_name, "dist_nm": round(dist_nm, 1), "sog_kt": round(sog, 1)},
                ))

        # ── Open-ocean stop: near-zero speed far from any port ────────────────
        if dist_nm >= OPEN_OCEAN_MIN_NM and sog <= OPEN_OCEAN_STOP_KT:
            open_ocean_stop_pings += 1
            if open_ocean_stop_pings == 1:
                events.append(SpatialEvent(
                    mmsi=mmsi, event_type="open_ocean_stop",
                    timestamp=ts, lat=lat, lon=lon,
                    detail={"nearest_port": port_name, "dist_nm": round(dist_nm, 1), "sog_kt": round(sog, 1)},
                ))

        # ── EEZ boundary hugging ───────────────────────────────────────────────
        if eez_names:
            eez_hug_streak += 1
            last_eez_names = eez_names
            max_eez_streak = max(max_eez_streak, eez_hug_streak)
        else:
            if eez_hug_streak >= EEZ_HUG_MIN_PINGS:
                events.append(SpatialEvent(
                    mmsi=mmsi, event_type="eez_hugging",
                    timestamp=timestamps[i - 1], lat=lats[i - 1], lon=lons[i - 1],
                    detail={"boundary": last_eez_names, "consecutive_pings": eez_hug_streak},
                ))
            eez_hug_streak = 0

    # Catch streak that runs to end of track
    if eez_hug_streak >= EEZ_HUG_MIN_PINGS:
        events.append(SpatialEvent(
            mmsi=mmsi, event_type="eez_hugging",
            timestamp=timestamps[-1], lat=lats[-1], lon=lons[-1],
            detail={"boundary": last_eez_names, "consecutive_pings": eez_hug_streak},
        ))

    # ── Sub-scores (0–1) ──────────────────────────────────────────────────────
    port_loiter_score     = min(port_loiter_pings / max(n * 0.2, 1), 1.0)
    open_ocean_stop_score = min(open_ocean_stop_pings / 5.0, 1.0)
    eez_hug_score         = min(max_eez_streak / 20.0, 1.0)

    spatial_score = min(
        0.40 * port_loiter_score
        + 0.35 * open_ocean_stop_score
        + 0.25 * eez_hug_score,
        1.0,
    )

    return SpatialResult(
        mmsi=mmsi,
        spatial_score=round(spatial_score, 4),
        port_loiter_score=round(port_loiter_score, 4),
        open_ocean_stop_score=round(open_ocean_stop_score, 4),
        eez_hug_score=round(eez_hug_score, 4),
        events=events,
    )


# ── Public API ────────────────────────────────────────────────────────────────

def compute_spatial_scores(df: pd.DataFrame) -> list[SpatialResult]:
    """
    Compute spatial anomaly scores for every vessel in df.

    Args:
        df: Cleaned AIS DataFrame. Required columns:
            mmsi, timestamp, lat, lon, sog

    Returns:
        List of SpatialResult, one per vessel with >= MIN_PINGS pings.
    """
    required = {"mmsi", "timestamp", "lat", "lon", "sog"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"spatial.py: missing columns {missing}")

    df = df.sort_values(["mmsi", "timestamp"]).copy()

    results: list[SpatialResult] = []
    for _, group in df.groupby("mmsi", sort=False):
        if len(group) < MIN_PINGS:
            continue
        results.append(_score_vessel(group))

    return results


def spatial_scores_to_df(results: list[SpatialResult]) -> pd.DataFrame:
    """
    Flatten SpatialResult list into a tidy DataFrame for builder.py.

    Columns: mmsi, spatial_score, port_loiter_score,
             open_ocean_stop_score, eez_hug_score, spatial_event_count
    """
    return pd.DataFrame([
        {
            "mmsi": r.mmsi,
            "spatial_score": r.spatial_score,
            "port_loiter_score": r.port_loiter_score,
            "open_ocean_stop_score": r.open_ocean_stop_score,
            "eez_hug_score": r.eez_hug_score,
            "spatial_event_count": len(r.events),
        }
        for r in results
    ])