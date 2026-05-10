"""
lostping/report/html_report.py

Generates a styled, self-contained HTML summary report from flagged.json.
No external dependencies — pure HTML/CSS/JS, opens in any browser.

Features:
  - Summary stats header
  - Sortable table of flagged vessels
  - Risk score bar per vessel
  - Colour-coded risk level badges
  - Expandable signal breakdown per vessel
  - Links to map
"""

from __future__ import annotations

import json
import os
from datetime import datetime


# ── Helpers ───────────────────────────────────────────────────────────────────

def _risk_colour(score: float) -> str:
    if score >= 80: return "#8B0000"
    if score >= 70: return "#FF0000"
    if score >= 60: return "#FF6600"
    return "#FFD700"


def _risk_label(score: float) -> str:
    if score >= 80: return "CRITICAL"
    if score >= 70: return "HIGH"
    if score >= 60: return "ELEVATED"
    return "MODERATE"


def _score_bar(score: float, colour: str) -> str:
    pct = min(score, 100)
    return f"""
    <div style="background:#2a2a2a; border-radius:3px; height:8px; width:100px; display:inline-block; vertical-align:middle;">
      <div style="background:{colour}; width:{pct}px; height:8px; border-radius:3px;"></div>
    </div>
    <span style="color:{colour}; font-weight:bold; margin-left:6px;">{score}</span>
    """


def _signal_pills(v: dict) -> str:
    pills = []
    scores = {
        "gap":      ("gap_score",          "#e74c3c"),
        "traj":     ("trajectory_score",   "#e67e22"),
        "spatial":  ("spatial_score",      "#f39c12"),
        "iso":      ("isolation_score",    "#9b59b6"),
        "zscore":   ("zscore_score",       "#3498db"),
        "loiter":   ("loiter_score",       "#1abc9c"),
    }
    for label, (col, colour) in scores.items():
        val = v.get(col) or 0
        opacity = max(0.2, float(val))
        pills.append(
            f'<span style="background:{colour}; opacity:{opacity:.2f}; '
            f'color:white; padding:2px 7px; border-radius:10px; '
            f'font-size:11px; margin:1px; display:inline-block;">'
            f'{label} {float(val):.2f}</span>'
        )
    return " ".join(pills)


def _vessel_row(v: dict, idx: int) -> str:
    colour   = _risk_colour(v["risk_score"])
    label    = _risk_label(v["risk_score"])
    name     = v.get("vessel_name") or "UNKNOWN"
    vtype    = v.get("vessel_type") or "?"
    pings    = v.get("ping_count") or "?"
    reason   = (v.get("reason") or "").replace(";", " &bull;")
    first    = str(v.get("first_seen") or "")[:19]
    last     = str(v.get("last_seen") or "")[:19]
    gap_n    = len(v.get("gap_events", []))
    spat_n   = len(v.get("spatial_events", []))
    loit_n   = len(v.get("loiter_clusters", []))

    row_bg = "#1a1a1a" if idx % 2 == 0 else "#1f1f1f"

    return f"""
    <tr style="background:{row_bg}; border-bottom:1px solid #2a2a2a;">
      <td style="padding:10px 12px; color:{colour}; font-weight:bold; font-size:13px; white-space:nowrap;">
        <span style="background:{colour}; color:black; padding:2px 6px; border-radius:3px; font-size:10px; margin-right:6px;">{label}</span>
        {name}
      </td>
      <td style="padding:10px 8px; color:#aaa; font-size:12px;">{v['mmsi']}</td>
      <td style="padding:10px 8px; color:#aaa; font-size:12px;">{vtype}</td>
      <td style="padding:10px 8px;">{_score_bar(v['risk_score'], colour)}</td>
      <td style="padding:10px 8px; color:#ccc; font-size:11px;">{_signal_pills(v)}</td>
      <td style="padding:10px 8px; color:#888; font-size:11px; max-width:280px;">{reason}</td>
      <td style="padding:10px 8px; color:#666; font-size:11px; white-space:nowrap;">{first}<br>{last}</td>
      <td style="padding:10px 8px; color:#aaa; font-size:11px; text-align:center;">
        {pings}<br>
        <span style="color:#e74c3c;">G:{gap_n}</span>
        <span style="color:#f39c12; margin:0 3px;">S:{spat_n}</span>
        <span style="color:#9b59b6;">L:{loit_n}</span>
      </td>
    </tr>
    """


# ── Public API ────────────────────────────────────────────────────────────────

def write_html_report(
    report_path: str,
    out_path: str = "outputs/reports/report.html",
    map_path: str | None = None,
) -> str:
    """
    Generate a styled HTML summary report from flagged.json.

    Args:
        report_path: Path to flagged.json.
        out_path:    Where to write the HTML report.
        map_path:    Optional path to the Folium map HTML (for link).

    Returns:
        Absolute path of the written file.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(report_path, encoding="utf-8") as f:
        report = json.load(f)

    vessels  = report["vessels"]
    n_flag   = report["vessels_flagged"]
    n_total  = report["vessels_analysed"]
    src      = report["source_file"]
    gen_at   = report["generated_at"][:19].replace("T", " ")
    threshold = report["risk_threshold"]

    # Score distribution for header stats
    scores = [v["risk_score"] for v in vessels]
    max_score  = max(scores) if scores else 0
    mean_score = sum(scores) / len(scores) if scores else 0

    map_link = ""
    if map_path and os.path.exists(map_path):
        map_rel = os.path.relpath(map_path, os.path.dirname(out_path))
        map_link = f'<a href="{map_rel}" style="color:#3498db; text-decoration:none;">🗺 Open Interactive Map</a>'

    rows_html = "".join(_vessel_row(v, i) for i, v in enumerate(vessels))

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>lostping — {src}</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ background: #111; color: #ddd; font-family: 'Courier New', monospace; }}
    .header {{ background: #0a0a0a; border-bottom: 1px solid #222; padding: 20px 32px; }}
    .header h1 {{ font-size: 22px; letter-spacing: 3px; color: #fff; }}
    .header h1 span {{ color: #FF6600; }}
    .meta {{ color: #555; font-size: 12px; margin-top: 4px; }}
    .stats {{ display: flex; gap: 24px; margin-top: 16px; flex-wrap: wrap; }}
    .stat {{ background: #1a1a1a; border: 1px solid #2a2a2a; border-radius: 6px; padding: 12px 20px; min-width: 120px; }}
    .stat-val {{ font-size: 28px; font-weight: bold; color: #FF6600; }}
    .stat-label {{ font-size: 11px; color: #666; margin-top: 2px; letter-spacing: 1px; }}
    .map-link {{ margin-top: 12px; font-size: 13px; }}
    .table-wrap {{ padding: 24px 32px; overflow-x: auto; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    thead tr {{ background: #0d0d0d; border-bottom: 2px solid #333; }}
    thead th {{ padding: 10px 12px; text-align: left; color: #666; font-size: 11px;
                letter-spacing: 1px; font-weight: normal; white-space: nowrap; cursor: pointer; }}
    thead th:hover {{ color: #aaa; }}
    tbody tr:hover {{ background: #252525 !important; }}
    .footer {{ padding: 20px 32px; color: #333; font-size: 11px; border-top: 1px solid #1a1a1a; }}
  </style>
</head>
<body>

<div class="header">
  <h1>lost<span>ping</span></h1>
  <div class="meta">generated {gen_at} UTC &nbsp;|&nbsp; source: {src} &nbsp;|&nbsp; threshold: {threshold}</div>
  <div class="stats">
    <div class="stat"><div class="stat-val">{n_flag}</div><div class="stat-label">FLAGGED</div></div>
    <div class="stat"><div class="stat-val">{n_total}</div><div class="stat-label">ANALYSED</div></div>
    <div class="stat"><div class="stat-val">{max_score:.0f}</div><div class="stat-label">MAX SCORE</div></div>
    <div class="stat"><div class="stat-val">{mean_score:.1f}</div><div class="stat-label">MEAN SCORE</div></div>
    <div class="stat"><div class="stat-val">{n_flag/n_total*100:.1f}%</div><div class="stat-label">FLAG RATE</div></div>
  </div>
  <div class="map-link">{map_link}</div>
</div>

<div class="table-wrap">
  <table id="report-table">
    <thead>
      <tr>
        <th onclick="sortTable(0)">VESSEL ↕</th>
        <th onclick="sortTable(1)">MMSI ↕</th>
        <th onclick="sortTable(2)">TYPE ↕</th>
        <th onclick="sortTable(3)">RISK SCORE ↕</th>
        <th>SIGNALS</th>
        <th>REASON</th>
        <th onclick="sortTable(6)">TIMESPAN ↕</th>
        <th onclick="sortTable(7)">PINGS / EVENTS ↕</th>
      </tr>
    </thead>
    <tbody>
      {rows_html}
    </tbody>
  </table>
</div>

<div class="footer">
  lostping &nbsp;|&nbsp; {n_flag} vessels flagged from {n_total} analysed &nbsp;|&nbsp; {src}
</div>

<script>
function sortTable(col) {{
  const table = document.getElementById("report-table");
  const rows  = Array.from(table.querySelectorAll("tbody tr"));
  const asc   = table.dataset.sortCol == col && table.dataset.sortDir == "asc";
  rows.sort((a, b) => {{
    const ta = a.cells[col]?.innerText.trim() || "";
    const tb = b.cells[col]?.innerText.trim() || "";
    const na = parseFloat(ta), nb = parseFloat(tb);
    if (!isNaN(na) && !isNaN(nb)) return asc ? na - nb : nb - na;
    return asc ? ta.localeCompare(tb) : tb.localeCompare(ta);
  }});
  rows.forEach(r => table.querySelector("tbody").appendChild(r));
  table.dataset.sortCol = col;
  table.dataset.sortDir = asc ? "desc" : "asc";
}}
</script>

</body>
</html>"""

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"html report saved → {out_path}")
    return os.path.abspath(out_path)