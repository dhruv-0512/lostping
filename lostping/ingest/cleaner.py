"""
cleaner.py — deduplicate and sanity-check a loaded AIS DataFrame

Call this after load_csv() / load_multiple() before passing to features/.
"""

import pandas as pd


# MMSI ranges that are not real vessels
# 0        = unknown / not set
# 9xxxxxxx = base stations, aids to navigation, search & rescue
# 111xxxxx = SAR aircraft
INVALID_MMSI_PREFIXES = ("0", "111", "99", "98", "97")

# Minimum pings for a vessel track to be worth analysing
MIN_PINGS = 10


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run all cleaning steps in order. Returns a clean DataFrame.
    """
    df = _drop_invalid_mmsi(df)
    df = _drop_duplicate_pings(df)
    df = _drop_sparse_tracks(df)
    df = _fix_dtypes(df)

    print(
        f"clean complete\n"
        f"  vessels remaining : {df['mmsi'].nunique():,}\n"
        f"  total pings       : {len(df):,}"
    )

    return df


def _drop_invalid_mmsi(df: pd.DataFrame) -> pd.DataFrame:
    """Remove base stations, SAR aircraft, and unset MMSIs."""
    mmsi_str = df["mmsi"].astype(str)
    mask = ~mmsi_str.str.startswith(INVALID_MMSI_PREFIXES)
    dropped = (~mask).sum()
    if dropped:
        print(f"  dropped {dropped:,} rows with invalid MMSI")
    return df[mask].copy()


def _drop_duplicate_pings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Same vessel, same timestamp — keep first occurrence.
    Marine Cadastre sometimes has duplicates when two receivers
    pick up the same broadcast.
    """
    before = len(df)
    df = df.drop_duplicates(subset=["mmsi", "timestamp"], keep="first")
    dropped = before - len(df)
    if dropped:
        print(f"  dropped {dropped:,} duplicate pings")
    return df


def _drop_sparse_tracks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove vessels with fewer than MIN_PINGS — too little data
    to compute meaningful trajectory features or detect gaps.
    """
    counts = df.groupby("mmsi")["timestamp"].count()
    valid_mmsi = counts[counts >= MIN_PINGS].index
    before = df["mmsi"].nunique()
    df = df[df["mmsi"].isin(valid_mmsi)].copy()
    dropped = before - df["mmsi"].nunique()
    if dropped:
        print(f"  dropped {dropped:,} vessels with < {MIN_PINGS} pings")
    return df


def _fix_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure timestamps are sorted and dtypes are correct."""
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=False)
    df = df.sort_values(["mmsi", "timestamp"]).reset_index(drop=True)
    df["mmsi"] = df["mmsi"].astype(str)   # string from here on — easier to handle
    return df