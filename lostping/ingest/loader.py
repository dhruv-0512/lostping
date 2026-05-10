"""
loader.py — parse and filter raw Marine Cadastre AIS CSV files

Marine Cadastre 2024 schema:
    mmsi, base_date_time, longitude, latitude, sog, cog, heading,
    vessel_name, imo, call_sign, vessel_type, status, length, width,
    draft, cargo, transceiver
"""

import pandas as pd
import pathlib
from tqdm import tqdm

COLUMN_MAP = {
    "mmsi":            "mmsi",
    "base_date_time":  "timestamp",
    "longitude":       "lon",
    "latitude":        "lat",
    "sog":             "sog",
    "cog":             "cog",
    "heading":         "heading",
    "vessel_name":     "vessel_name",
    "vessel_type":     "vessel_type",
    "status":          "status",
    "length":          "length",
    "draft":           "draft",
}

KEEP_COLS = list(COLUMN_MAP.keys())

# Marine Cadastre vessel_type codes worth tracking
# Full list: https://marinecadastre.gov/ais/  (Vessel Codes link)
VESSEL_TYPE_LABELS = {
    30: "fishing",
    31: "towing",
    32: "towing_large",
    37: "pleasure",
    52: "tug",
    60: "passenger",
    70: "cargo",
    71: "cargo",
    72: "cargo",
    73: "cargo",
    74: "cargo",
    80: "tanker",
    81: "tanker",
    82: "tanker",
    83: "tanker",
    84: "tanker",
    1001: "fishing",
    1016: "cargo",
    1017: "tanker",
}

# Bounding boxes for named region presets [lon_min, lat_min, lon_max, lat_max]
REGION_PRESETS = {
    "gulf_of_mexico":   (-97.5, 18.0, -80.5, 30.5),
    "east_coast":       (-82.0, 25.0, -65.0, 45.0),
    "west_coast":       (-130.0, 30.0, -116.0, 49.0),
    "great_lakes":      (-92.0, 41.0, -76.0, 49.0),
    "chesapeake":       (-77.5, 36.5, -75.0, 39.5),
}


def load_csv(
    path: str | pathlib.Path,
    region: str | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    vessel_types: list[int] | None = None,
    chunksize: int = 200_000,
) -> pd.DataFrame:
    """
    Load one Marine Cadastre daily AIS CSV into a clean DataFrame.

    Args:
        path:         Path to the .csv file
        region:       Named region preset string (see REGION_PRESETS)
        bbox:         Manual bounding box (lon_min, lat_min, lon_max, lat_max)
                      overrides region if both provided
        vessel_types: List of vessel_type codes to keep. None = keep all.
        chunksize:    Rows per chunk (tune based on available RAM)

    Returns:
        DataFrame with cleaned, filtered vessel pings.
    """
    path = pathlib.Path(path)
    if not path.exists():
        raise FileNotFoundError(f"AIS file not found: {path}")

    bounds = _resolve_bounds(region, bbox)

    chunks = []
    total_rows = 0
    kept_rows = 0

    reader = pd.read_csv(
        path,
        usecols=KEEP_COLS,
        dtype={
            "mmsi":        "Int64",
            "vessel_type": "Int64",
            "status":      "Int64",
            "sog":         float,
            "cog":         float,
            "heading":     float,
            "lon":         float,
            "lat":         float,
        },
        parse_dates=["base_date_time"],
        chunksize=chunksize,
        low_memory=False,
    )

    for chunk in tqdm(reader, desc=f"loading {path.name}", unit="chunk"):
        total_rows += len(chunk)

        chunk = chunk.rename(columns=COLUMN_MAP)

        # drop rows with missing critical fields
        chunk = chunk.dropna(subset=["mmsi", "timestamp", "lat", "lon"])

        # spatial filter
        if bounds:
            lon_min, lat_min, lon_max, lat_max = bounds
            chunk = chunk[
                (chunk["lon"] >= lon_min) & (chunk["lon"] <= lon_max) &
                (chunk["lat"] >= lat_min) & (chunk["lat"] <= lat_max)
            ]

        # vessel type filter
        if vessel_types:
            chunk = chunk[chunk["vessel_type"].isin(vessel_types)]

        # drop obvious garbage
        chunk = chunk[
            (chunk["sog"] >= 0) & (chunk["sog"] <= 60) &   # knots — nothing goes faster
            (chunk["lat"].between(-90, 90)) &
            (chunk["lon"].between(-180, 180))
        ]

        kept_rows += len(chunk)
        if len(chunk) > 0:
            chunks.append(chunk)

    if not chunks:
        raise ValueError(
            f"No data survived filtering. Check your region/bbox. "
            f"Total rows scanned: {total_rows:,}"
        )

    df = pd.concat(chunks, ignore_index=True)
    df = df.sort_values(["mmsi", "timestamp"]).reset_index(drop=True)

    # map numeric vessel_type to readable label
    df["vessel_type_label"] = (
        df["vessel_type"]
        .map(VESSEL_TYPE_LABELS)
        .fillna("other")
    )

    print(
        f"\nloaded {path.name}\n"
        f"  total rows scanned : {total_rows:,}\n"
        f"  rows after filter  : {kept_rows:,}\n"
        f"  unique vessels     : {df['mmsi'].nunique():,}\n"
        f"  date range         : {df['timestamp'].min()} → {df['timestamp'].max()}\n"
        f"  vessel types       : {df['vessel_type_label'].value_counts().to_dict()}"
    )

    return df


def load_multiple(
    paths: list[str | pathlib.Path],
    **kwargs,
) -> pd.DataFrame:
    """
    Load and concatenate multiple daily AIS CSV files.
    All kwargs passed through to load_csv.
    """
    frames = [load_csv(p, **kwargs) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["mmsi", "timestamp"]).reset_index(drop=True)
    df = df.drop_duplicates(subset=["mmsi", "timestamp"])
    return df


def _resolve_bounds(
    region: str | None,
    bbox: tuple | None,
) -> tuple | None:
    if bbox:
        return bbox
    if region:
        if region not in REGION_PRESETS:
            raise ValueError(
                f"Unknown region '{region}'. "
                f"Available: {list(REGION_PRESETS.keys())}"
            )
        return REGION_PRESETS[region]
    return None