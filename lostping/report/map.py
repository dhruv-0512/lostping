"""
lostping/report/map.py

Generates an interactive Folium HTML map from flagged.json.

For each flagged vessel:
  - Draws the full AIS track as a polyline, coloured by risk score
  - Places a marker at the vessel's last known position
  - Popup shows vessel name, MMSI, risk score, and reason
  - Gap events marked with a red circle
  - Spatial events marked with an orange circle
  - Loiter clusters marked with a purple circle showing cluster radius

Risk score colour scale:
  50–60  → yellow
  60–70  → orange
  70–80  → red
  80+    → dark red
"""

from __future__ import annotations

import json
import os

import folium
from folium import plugins
import pandas as pd


# ── Colour helpers ────────────────────────────────────────────────────────────

def _track_colour(risk_score: float) -> str:
    if risk_score >= 80:
        return "#8B0000"   # dark red
    if risk_score >= 70:
        return "#FF0000"   # red
    if risk_score >= 60:
        return "#FF6600"   # orange
    return "#FFD700"       # yellow


def _risk_label(risk_score: float) -> str:
    if risk_score >= 80:
        return "CRITICAL"
    if risk_score >= 70:
        return "HIGH"
    if risk_score >= 60:
        return "ELEVATED"
    return "MODERATE"


# ── Popup builder ─────────────────────────────────────────────────────────────

def _vessel_popup(v: dict) -> str:
    label = _risk_label(v["risk_score"])
    colour = _track_colour(v["risk_score"])
    reasons = v.get("reason", "").replace(";", "<br>  •")

    gap_count  = len(v.get("gap_events", []))
    spat_count = len(v.get("spatial_events", []))
    loit_count = len(v.get("loiter_clusters", []))

    return f"""
    <div style="font-family: monospace; font-size: 13px; min-width: 260px;">
      <div style="background:{colour}; color:white; padding:6px 10px; border-radius:4px; margin-bottom:8px;">
        <b>{v.get('vessel_name', 'UNKNOWN')}</b>
        &nbsp;&nbsp;<span style="font-size:11px;">{label}</span>
      </div>
      <table style="border-collapse:collapse; width:100%;">
        <tr><td style="color:#888;">MMSI</td><td><b>{v['mmsi']}</b></td></tr>
        <tr><td style="color:#888;">Type</td><td>{v.get('vessel_type','?')}</td></tr>
        <tr><td style="color:#888;">Risk</td><td><b>{v['risk_score']}</b> / 100</td></tr>
        <tr><td style="color:#888;">Pings</td><td>{v.get('ping_count','?')}</td></tr>
        <tr><td style="color:#888;">First seen</td><td>{str(v.get('first_seen',''))[:19]}</td></tr>
        <tr><td style="color:#888;">Last seen</td><td>{str(v.get('last_seen',''))[:19]}</td></tr>
        <tr><td style="color:#888;">Gap events</td><td>{gap_count}</td></tr>
        <tr><td style="color:#888;">Spatial events</td><td>{spat_count}</td></tr>
        <tr><td style="color:#888;">Loiter clusters</td><td>{loit_count}</td></tr>
      </table>
      <div style="margin-top:8px; padding:6px; background:#f5f5f5; border-radius:4px; font-size:11px;">
        <b>Signals:</b><br>  • {reasons}
      </div>
      <div style="margin-top:6px; font-size:11px; color:#888;">
        iso={v.get('isolation_score',0):.2f}
        gap={v.get('gap_score',0):.2f}
        traj={v.get('trajectory_score',0):.2f}
        spat={v.get('spatial_score',0):.2f}
      </div>
    </div>
    """


# ── Track drawing ─────────────────────────────────────────────────────────────

def _draw_vessel(
    m: folium.Map,
    v: dict,
    df_vessel: pd.DataFrame,
    track_group: folium.FeatureGroup,
    event_group: folium.FeatureGroup,
) -> None:
    colour = _track_colour(v["risk_score"])
    mmsi   = v["mmsi"]

    # ── Track polyline ────────────────────────────────────────────────────────
    if not df_vessel.empty:
        coords = list(zip(df_vessel["lat"], df_vessel["lon"]))
        if len(coords) >= 2:
            folium.PolyLine(
                locations=coords,
                color=colour,
                weight=2.5,
                opacity=0.8,
                tooltip=f"{v.get('vessel_name','?')} ({v['risk_score']})",
            ).add_to(track_group)

        # Last position marker
        last = df_vessel.iloc[-1]
        folium.Marker(
            location=[last["lat"], last["lon"]],
            popup=folium.Popup(_vessel_popup(v), max_width=320),
            tooltip=f"{v.get('vessel_name','?')} ▸ {v['risk_score']}",
            icon=folium.Icon(color="red" if v["risk_score"] >= 70 else "orange",
                             icon="ship", prefix="fa"),
        ).add_to(track_group)

    # ── Gap events ────────────────────────────────────────────────────────────
    for ge in v.get("gap_events", []):
        lat = ge.get("start_lat") or ge.get("lat")
        lon = ge.get("start_lon") or ge.get("lon")
        if lat is None or lon is None:
            continue
        folium.CircleMarker(
            location=[lat, lon],
            radius=8,
            color="#CC0000",
            fill=True,
            fill_color="#FF4444",
            fill_opacity=0.7,
            tooltip=f"AIS gap: {ge.get('gap_hours', ge.get('duration_hours', '?')):.1f}h",
        ).add_to(event_group)

    # ── Spatial events ────────────────────────────────────────────────────────
    for se in v.get("spatial_events", []):
        lat = se.get("lat")
        lon = se.get("lon")
        if lat is None or lon is None:
            continue
        etype = se.get("event_type", "spatial")
        colour_map = {
            "port_loiter":    "#FF8C00",
            "open_ocean_stop":"#8B008B",
            "eez_hugging":    "#006400",
        }
        ec = colour_map.get(etype, "#FF8C00")
        folium.CircleMarker(
            location=[lat, lon],
            radius=7,
            color=ec,
            fill=True,
            fill_color=ec,
            fill_opacity=0.6,
            tooltip=f"{etype}: {se.get('detail', {})}",
        ).add_to(event_group)

    # ── Loiter clusters ───────────────────────────────────────────────────────
    for lc in v.get("loiter_clusters", []):
        clat = lc.get("centre_lat")
        clon = lc.get("centre_lon")
        if clat is None or clon is None:
            continue
        radius_m = max((lc.get("radius_nm", 1.0) * 1852), 500)
        folium.Circle(
            location=[clat, clon],
            radius=radius_m,
            color="#6A0DAD",
            fill=True,
            fill_color="#9B59B6",
            fill_opacity=0.15,
            tooltip=f"Loiter cluster: {lc.get('ping_count','?')} pings, {lc.get('dwell_hours','?'):.1f}h",
        ).add_to(event_group)


# ── Public API ────────────────────────────────────────────────────────────────

def build_map(
    report_path: str,
    df: pd.DataFrame,
    out_path: str = "outputs/maps/gulf.html",
) -> str:
    """
    Build an interactive Folium map from a flagged.json report.

    Args:
        report_path: Path to flagged.json (output of json_report.write_json_report).
        df:          Full cleaned AIS DataFrame (for vessel tracks).
        out_path:    Where to write the HTML map.

    Returns:
        Absolute path of the written HTML file.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(report_path, encoding="utf-8") as f:
        report = json.load(f)

    vessels = report["vessels"]
    if not vessels:
        print("no flagged vessels to map")
        return out_path

    # Centre map on mean position of flagged vessels
    all_lats = [v["track_bounds"]["lat_min"] for v in vessels if v["track_bounds"]["lat_min"]]
    all_lons = [v["track_bounds"]["lon_min"] for v in vessels if v["track_bounds"]["lon_min"]]
    centre   = [sum(all_lats) / len(all_lats), sum(all_lons) / len(all_lons)]

    m = folium.Map(
        location=centre,
        zoom_start=6,
        tiles="CartoDB dark_matter",
        prefer_canvas=True,
    )

    # ── Feature groups (toggleable in legend) ─────────────────────────────────
    track_group = folium.FeatureGroup(name="Vessel Tracks", show=True)
    event_group = folium.FeatureGroup(name="Events", show=True)

    # ── Index AIS data by MMSI for fast lookup ────────────────────────────────
    flagged_mmsis = {v["mmsi"] for v in vessels}
    df_flagged    = df[df["mmsi"].isin(flagged_mmsis)].copy()
    df_flagged    = df_flagged.sort_values(["mmsi", "timestamp"])

    print(f"drawing {len(vessels)} vessel tracks...")
    for v in vessels:
        df_vessel = df_flagged[df_flagged["mmsi"] == v["mmsi"]]
        _draw_vessel(m, v, df_vessel, track_group, event_group)

    track_group.add_to(m)
    event_group.add_to(m)

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_html = """
    <div style="
        position: fixed; bottom: 30px; left: 30px; z-index: 1000;
        background: rgba(20,20,20,0.85); color: white;
        padding: 12px 16px; border-radius: 8px;
        font-family: monospace; font-size: 12px; line-height: 1.8;
        border: 1px solid #444;">
      <b>lostping</b> — risk score<br>
      <span style="color:#FFD700;">██</span> 50–60  MODERATE<br>
      <span style="color:#FF6600;">██</span> 60–70  ELEVATED<br>
      <span style="color:#FF0000;">██</span> 70–80  HIGH<br>
      <span style="color:#8B0000;">██</span> 80+    CRITICAL<br>
      <hr style="border-color:#444; margin:6px 0;">
      <span style="color:#FF4444;">●</span> AIS gap&nbsp;&nbsp;
      <span style="color:#FF8C00;">●</span> port loiter<br>
      <span style="color:#8B008B;">●</span> ocean stop&nbsp;
      <span style="color:#006400;">●</span> EEZ hug<br>
      <span style="color:#9B59B6;">○</span> loiter cluster
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    # ── Layer control ─────────────────────────────────────────────────────────
    folium.LayerControl(collapsed=False).add_to(m)

    # ── Title ─────────────────────────────────────────────────────────────────
    title_html = f"""
    <div style="
        position: fixed; top: 16px; left: 50%; transform: translateX(-50%);
        z-index: 1000; background: rgba(20,20,20,0.85); color: white;
        padding: 8px 20px; border-radius: 6px;
        font-family: monospace; font-size: 14px; letter-spacing: 1px;">
      lostping &nbsp;|&nbsp; {report['vessels_flagged']} flagged
      &nbsp;/&nbsp; {report['vessels_analysed']} analysed
      &nbsp;|&nbsp; {report['source_file']}
    </div>
    """
    m.get_root().html.add_child(folium.Element(title_html))

    m.save(out_path)
    print(f"map saved → {out_path}")
    return os.path.abspath(out_path)