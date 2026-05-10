# lostping

**AIS vessel anomaly detection — find ships behaving suspiciously.**

Ingests raw AIS tracking data, detects vessels that go dark, loiter, or move erratically, scores them 0–100, and outputs an interactive map + JSON + HTML report.

Real-world use cases: sanctions evasion detection, smuggling, illegal fishing, ship-to-ship transfer monitoring.

---

## What it detects

| Signal | Method | Weight |
|--------|--------|--------|
| AIS blackout | Time gaps between pings + STS zone proximity | 25% |
| Erratic movement | Speed variance, heading circular variance, speed reversals | 30% |
| Spatial anomaly | Port loitering, open-ocean stops, EEZ boundary hugging | 20% |
| ML anomaly | Isolation Forest per vessel type | 15% |
| Statistical baseline | Z-score vs vessel type peers | 10% |

---

## Quickstart

```bash
# 1. Install
git clone https://github.com/dhruv-0512/lostping
cd lostping
pip install -e .

# 2. Get data — Marine Cadastre (NOAA), free, no account needed
# https://marinecadastre.gov/ais/
# Download a daily .csv.zst file, decompress:
zstd -d AIS_2024_01_03.csv.zst
mv AIS_2024_01_03.csv data/raw/

# 3. Run full pipeline
lostping run --input data/raw/AIS_2024_01_03.csv --region gulf_of_mexico --out outputs/

# 4. Open outputs
#   outputs/map.html      — interactive vessel map
#   outputs/report.html   — sortable summary table
#   outputs/flagged.json  — structured data for downstream tools
```

---

## Commands

```bash
# Ingest only — parse + clean, save to parquet
lostping ingest --input data/raw/ --region gulf_of_mexico --out data/processed/

# Detect only — run models on processed data
lostping detect --input data/processed/ --risk-threshold 60 --out outputs/reports/

# Map only — generate map + HTML from existing flagged.json
lostping map --report outputs/reports/flagged.json --input data/processed/ --out outputs/maps/map.html

# Full pipeline in one shot
lostping run --input data/raw/AIS_2024_01_03.csv --region gulf_of_mexico --out outputs/
```

---

## Data source

**Marine Cadastre (NOAA)** — [marinecadastre.gov/ais](https://marinecadastre.gov/ais/)

- Free, no account needed
- Daily `.csv.zst` files (Zstandard compressed)
- Decompress with `zstd -d filename.csv.zst`
- Recommended: 3–5 days from Gulf of Mexico zone for testing

---

## Output

### `flagged.json`
```json
{
  "vessels_analysed": 4621,
  "vessels_flagged": 34,
  "vessels": [
    {
      "mmsi": "368120080",
      "vessel_name": "JET 1",
      "risk_score": 73.3,
      "reason": "AIS dark 14.8h; erratic speed; port loitering",
      "gap_events": [...],
      "spatial_events": [...],
      "loiter_clusters": [...]
    }
  ]
}
```

### `map.html`
Interactive Folium map on dark CartoDB tiles. Vessel tracks coloured by risk score, click any marker for full breakdown.

### `report.html`
Sortable dark-theme summary table with signal pills, score bars, and event counts per vessel.

---

## Pipeline

```
Raw AIS CSV (Marine Cadastre)
        ↓
ingest/loader.py      — chunk parse, region filter, column rename
ingest/cleaner.py     — drop invalid MMSIs, dedupe, drop sparse tracks
        ↓
features/gaps.py      — AIS silence detection + STS zone proximity
features/trajectory.py — speed variance, heading erraticism
features/spatial.py   — port proximity, EEZ boundary checks
features/builder.py   — assemble feature matrix (one row per vessel)
        ↓
models/isolation.py   — Isolation Forest per vessel type
models/loiter.py      — DBSCAN loitering cluster detection
models/zscore.py      — speed/heading z-score vs type baseline
models/scorer.py      — weighted risk score 0–100 + reason string
        ↓
report/json_report.py — flagged.json
report/map.py         — Folium interactive HTML map
report/html_report.py — styled HTML summary table
        ↓
cli.py                — lostping ingest / detect / map / run
```

---

## Performance

One day of Gulf of Mexico AIS data:
- **7.3M** raw pings processed
- **4,621** unique vessels scored
- **34** flagged above threshold 50
- Runs in ~3 minutes on a laptop

---

## Requirements

- Python 3.10+
- pandas, numpy, scikit-learn, folium, pyarrow, zstandard

All installed automatically via `pip install -e .`

---

## Project structure

```
lostping/
├── lostping/
│   ├── cli.py
│   ├── ingest/
│   │   ├── loader.py
│   │   └── cleaner.py
│   ├── features/
│   │   ├── gaps.py
│   │   ├── trajectory.py
│   │   ├── spatial.py
│   │   └── builder.py
│   ├── models/
│   │   ├── isolation.py
│   │   ├── loiter.py
│   │   ├── zscore.py
│   │   └── scorer.py
│   └── report/
│       ├── json_report.py
│       ├── map.py
│       └── html_report.py
├── data/
│   ├── raw/
│   └── processed/
├── outputs/
│   ├── maps/
│   └── reports/
├── pyproject.toml
└── README.md
```