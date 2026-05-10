"""
lostping/cli.py

Command-line interface for lostping.

Commands:
    lostping ingest  --input <file_or_dir> --region <region> --out <dir>
    lostping detect  --input <dir> --risk-threshold <n> --out <dir>
    lostping map     --report <flagged.json> --input <processed_dir> --out <file.html>
    lostping run     --input <file_or_dir> --region <region> --out <dir>
"""

from __future__ import annotations

import argparse
import os
import sys
import time


# ── Helpers ───────────────────────────────────────────────────────────────────

def _find_csvs(path: str) -> list[str]:
    if os.path.isfile(path):
        return [path]
    return sorted(
        os.path.join(path, f)
        for f in os.listdir(path)
        if f.endswith(".csv")
    )


def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _elapsed(t0: float) -> str:
    s = time.time() - t0
    return f"{s:.1f}s" if s < 60 else f"{s/60:.1f}min"


# ── Commands ──────────────────────────────────────────────────────────────────

def cmd_ingest(args: argparse.Namespace) -> None:
    from lostping.ingest.loader import load_csv, load_multiple
    from lostping.ingest.cleaner import clean
    import pandas as pd

    t0 = time.time()
    csvs = _find_csvs(args.input)
    if not csvs:
        print(f"error: no CSV files found at {args.input}", file=sys.stderr)
        sys.exit(1)

    print(f"ingesting {len(csvs)} file(s)...")
    if len(csvs) == 1:
        df = load_csv(csvs[0], region=args.region)
    else:
        df = load_multiple(csvs, region=args.region)

    df = clean(df)

    out_dir  = _ensure_dir(args.out)
    out_file = os.path.join(out_dir, "ais_clean.parquet")
    df.to_parquet(out_file, index=False)
    print(f"saved {len(df):,} pings → {out_file}  ({_elapsed(t0)})")


def cmd_detect(args: argparse.Namespace) -> None:
    import pandas as pd
    from lostping.features.builder import build_feature_matrix
    from lostping.models.isolation import compute_isolation_scores
    from lostping.models.loiter import compute_loiter_scores, loiter_scores_to_df
    from lostping.models.zscore import compute_zscore_scores
    from lostping.models.scorer import compute_risk_scores, get_flagged, score_summary
    from lostping.report.json_report import write_json_report

    t0 = time.time()

    # Load processed parquet if available, else raw CSVs
    parquet = os.path.join(args.input, "ais_clean.parquet")
    if os.path.exists(parquet):
        print(f"loading {parquet}...")
        df = pd.read_parquet(parquet)
    else:
        from lostping.ingest.loader import load_csv, load_multiple
        from lostping.ingest.cleaner import clean
        csvs = _find_csvs(args.input)
        if not csvs:
            print(f"error: no data found at {args.input}", file=sys.stderr)
            sys.exit(1)
        df = clean(load_multiple(csvs) if len(csvs) > 1 else load_csv(csvs[0]))

    fm = build_feature_matrix(df)
    fm = compute_isolation_scores(fm)

    loiter_df = loiter_scores_to_df(compute_loiter_scores(df))
    fm = fm.merge(loiter_df, on="mmsi", how="left")
    fm["loiter_score"] = fm["loiter_score"].fillna(0)

    fm     = compute_zscore_scores(fm)
    scored = compute_risk_scores(fm)
    score_summary(scored)

    flagged = get_flagged(scored, threshold=args.risk_threshold)

    out_dir  = _ensure_dir(args.out)
    src_name = os.path.basename(parquet if os.path.exists(parquet) else args.input)
    json_path = os.path.join(out_dir, "flagged.json")

    write_json_report(
        flagged=flagged,
        df=df,
        out_path=json_path,
        source_file=src_name,
        total_vessels=len(fm),
        risk_threshold=args.risk_threshold,
    )
    print(f"detect complete ({_elapsed(t0)})")


def cmd_map(args: argparse.Namespace) -> None:
    import pandas as pd
    from lostping.report.map import build_map
    from lostping.report.html_report import write_html_report

    t0 = time.time()

    # Load AIS data for tracks
    parquet = os.path.join(args.input, "ais_clean.parquet")
    if os.path.exists(parquet):
        df = pd.read_parquet(parquet)
    else:
        from lostping.ingest.loader import load_csv, load_multiple
        from lostping.ingest.cleaner import clean
        csvs = _find_csvs(args.input)
        df = clean(load_multiple(csvs) if len(csvs) > 1 else load_csv(csvs[0]))

    out_dir   = _ensure_dir(os.path.dirname(args.out) or args.out)
    map_path  = args.out if args.out.endswith(".html") else os.path.join(args.out, "map.html")
    html_path = map_path.replace("map.html", "report.html").replace(".html", "_report.html")

    build_map(report_path=args.report, df=df, out_path=map_path)
    write_html_report(report_path=args.report, out_path=html_path, map_path=map_path)
    print(f"map + report complete ({_elapsed(t0)})")


def cmd_run(args: argparse.Namespace) -> None:
    """Run the full pipeline: ingest → detect → map."""
    import pandas as pd
    from lostping.ingest.loader import load_csv, load_multiple
    from lostping.ingest.cleaner import clean
    from lostping.features.builder import build_feature_matrix
    from lostping.models.isolation import compute_isolation_scores
    from lostping.models.loiter import compute_loiter_scores, loiter_scores_to_df
    from lostping.models.zscore import compute_zscore_scores
    from lostping.models.scorer import compute_risk_scores, get_flagged, score_summary
    from lostping.report.json_report import write_json_report
    from lostping.report.map import build_map
    from lostping.report.html_report import write_html_report

    t0   = time.time()
    csvs = _find_csvs(args.input)
    if not csvs:
        print(f"error: no CSV files found at {args.input}", file=sys.stderr)
        sys.exit(1)

    print(f"=== lostping run — {len(csvs)} file(s) ===\n")

    # ── Ingest ────────────────────────────────────────────────────────────────
    region = getattr(args, "region", None)
    df = clean(
        load_multiple(csvs, region=region) if len(csvs) > 1
        else load_csv(csvs[0], region=region)
    )

    # ── Detect ────────────────────────────────────────────────────────────────
    fm = build_feature_matrix(df)
    fm = compute_isolation_scores(fm)

    loiter_df = loiter_scores_to_df(compute_loiter_scores(df))
    fm = fm.merge(loiter_df, on="mmsi", how="left")
    fm["loiter_score"] = fm["loiter_score"].fillna(0)

    fm     = compute_zscore_scores(fm)
    scored = compute_risk_scores(fm)
    score_summary(scored)

    threshold = getattr(args, "risk_threshold", 50)
    flagged   = get_flagged(scored, threshold=threshold)

    # ── Reports ───────────────────────────────────────────────────────────────
    out_dir   = _ensure_dir(args.out)
    src_name  = os.path.basename(csvs[0])
    json_path = os.path.join(out_dir, "flagged.json")
    map_path  = os.path.join(out_dir, "map.html")
    html_path = os.path.join(out_dir, "report.html")

    write_json_report(
        flagged=flagged, df=df, out_path=json_path,
        source_file=src_name, total_vessels=len(fm), risk_threshold=threshold,
    )
    build_map(report_path=json_path, df=df, out_path=map_path)
    write_html_report(report_path=json_path, out_path=html_path, map_path=map_path)

    print(f"\n=== done in {_elapsed(t0)} ===")
    print(f"  flagged.json → {json_path}")
    print(f"  map.html     → {map_path}")
    print(f"  report.html  → {html_path}")


# ── Argument parser ───────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lostping",
        description="AIS vessel anomaly detection — find ships behaving suspiciously.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── ingest ────────────────────────────────────────────────────────────────
    p_ingest = sub.add_parser("ingest", help="Parse and clean raw AIS CSV files")
    p_ingest.add_argument("--input",  required=True, help="Path to CSV file or directory")
    p_ingest.add_argument("--region", default=None,  help="Region preset (e.g. gulf_of_mexico)")
    p_ingest.add_argument("--out",    default="data/processed", help="Output directory")

    # ── detect ────────────────────────────────────────────────────────────────
    p_detect = sub.add_parser("detect", help="Run anomaly detection on processed data")
    p_detect.add_argument("--input",          required=True, help="Processed data directory")
    p_detect.add_argument("--risk-threshold", type=float, default=50.0, help="Flag vessels above this score (default 50)")
    p_detect.add_argument("--out",            default="outputs/reports", help="Output directory")

    # ── map ───────────────────────────────────────────────────────────────────
    p_map = sub.add_parser("map", help="Generate map and HTML report from flagged.json")
    p_map.add_argument("--report", required=True, help="Path to flagged.json")
    p_map.add_argument("--input",  required=True, help="Processed data directory (for tracks)")
    p_map.add_argument("--out",    default="outputs/maps/map.html", help="Output HTML map path")

    # ── run ───────────────────────────────────────────────────────────────────
    p_run = sub.add_parser("run", help="Full pipeline: ingest → detect → map")
    p_run.add_argument("--input",          required=True, help="Path to CSV file or directory")
    p_run.add_argument("--region",         default=None,  help="Region preset (e.g. gulf_of_mexico)")
    p_run.add_argument("--risk-threshold", type=float, default=50.0, help="Flag threshold (default 50)")
    p_run.add_argument("--out",            default="outputs", help="Output directory")

    return parser


def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()

    # Normalise hyphenated arg name
    if hasattr(args, "risk_threshold") is False and hasattr(args, "risk-threshold"):
        args.risk_threshold = getattr(args, "risk-threshold")

    dispatch = {
        "ingest": cmd_ingest,
        "detect": cmd_detect,
        "map":    cmd_map,
        "run":    cmd_run,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()