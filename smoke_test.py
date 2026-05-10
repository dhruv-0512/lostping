from lostping.ingest.loader import load_csv
from lostping.ingest.cleaner import clean
from lostping.features.builder import build_feature_matrix
from lostping.models.isolation import compute_isolation_scores
from lostping.models.loiter import compute_loiter_scores, loiter_scores_to_df
from lostping.models.zscore import compute_zscore_scores
from lostping.models.scorer import compute_risk_scores, get_flagged, score_summary

df = clean(load_csv("data/raw/ais-2024-01-03.csv", region="gulf_of_mexico"))
fm = build_feature_matrix(df)
fm = compute_isolation_scores(fm)

loiter_results = compute_loiter_scores(df)
loiter_df = loiter_scores_to_df(loiter_results)
fm = fm.merge(loiter_df, on="mmsi", how="left")
fm["loiter_score"] = fm["loiter_score"].fillna(0)

fm = compute_zscore_scores(fm)
scored = compute_risk_scores(fm)
score_summary(scored)

flagged = get_flagged(scored, threshold=50)
print(flagged[["mmsi", "vessel_name", "risk_score", "isolation_score", "loiter_score", "zscore_score", "reason"]].head(15).to_string())
from lostping.report.json_report import write_json_report

# write_json_report(
#     flagged=flagged,
#     df=df,
#     out_path="outputs/reports/flagged.json",
#     source_file="ais-2024-01-03.csv",
#     total_vessels=len(fm),
#     risk_threshold=50,
# )
# from lostping.report.map import build_map

# build_map(
#     report_path="outputs/reports/flagged.json",
#     df=df,
#     out_path="outputs/maps/gulf.html",
# )
from lostping.report.html_report import write_html_report

write_html_report(
    report_path="outputs/reports/flagged.json",
    out_path="outputs/reports/report.html",
    map_path="outputs/maps/gulf.html",
)