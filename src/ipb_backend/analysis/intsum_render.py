from __future__ import annotations

import html
from datetime import datetime, timezone
from typing import Any


def _dtg(dt: datetime | None = None) -> str:
    """Format as DTG: DDHHMMZ MON YY (e.g. 171830Z MAY 26)."""
    dt = dt or datetime.now(timezone.utc)
    return dt.strftime("%d%H%MZ %b %y").upper()


def _para(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return '<p class="empty">[NO REPORTING]</p>'
    # Split on double newlines into paragraphs, escape, preserve single newlines as <br>
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        paragraphs = [text]
    return "".join(
        f"<p>{html.escape(p).replace(chr(10), '<br>')}</p>" for p in paragraphs
    )


def _ccir(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return '<p class="empty">[NO REPORTING]</p>'
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    items = []
    for ln in lines:
        items.append(f"<li>{html.escape(ln)}</li>")
    return f'<ol class="ccir-list">{"".join(items)}</ol>'


def render_intsum_html(
    *,
    sections: dict[str, str],
    selection: dict[str, Any],
    provider: str,
    model: str | None = None,
    classification: str = "UNCLASSIFIED // FOR DEMONSTRATION",
) -> str:
    label = selection.get("label") or "UNNAMED AOI"
    area_sqkm = selection.get("area_sqkm", 0) or 0
    bounds = selection.get("bounds_wgs84") or []
    bounds_str = ", ".join(f"{b:.4f}" for b in bounds) if bounds else "N/A"
    dtg = _dtg()
    serial = f"INTSUM-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M')}"
    provider_label = "Claude (claude-sonnet-4-6)" if provider == "claude" else f"{provider.upper()} ANALYZER"
    if model:
        provider_label = f"Claude ({model})"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>INTSUM — {html.escape(label)} — {dtg}</title>
<style>
  @page {{ size: A4; margin: 18mm 16mm 22mm 16mm; }}
  * {{ box-sizing: border-box; }}
  html, body {{ margin: 0; padding: 0; background: #2a2d28; color: #1a1a1a; font-family: "Courier New", "Consolas", monospace; }}
  body {{ padding: 28px 16px 48px; }}
  .page {{
    max-width: 820px; margin: 0 auto; background: #f4efe2; padding: 0;
    box-shadow: 0 8px 32px rgba(0,0,0,0.6); border: 1px solid #2a2d28;
    position: relative;
  }}
  .classification {{
    background: #8b0000; color: #fff; text-align: center; padding: 8px 12px;
    font-weight: bold; letter-spacing: 0.15em; font-size: 0.9rem;
    border-bottom: 2px solid #000;
  }}
  .classification.bottom {{ border-bottom: none; border-top: 2px solid #000; }}
  .body {{ padding: 28px 36px; }}
  .doc-title {{
    text-align: center; font-size: 1.4rem; font-weight: bold;
    letter-spacing: 0.25em; margin: 8px 0 4px; color: #1a1a1a;
  }}
  .doc-subtitle {{
    text-align: center; font-size: 0.85rem; letter-spacing: 0.2em;
    color: #4a5040; margin-bottom: 18px; border-bottom: 2px solid #4a5040; padding-bottom: 14px;
  }}
  .meta-grid {{
    display: grid; grid-template-columns: 1fr 1fr; gap: 4px 24px;
    font-size: 0.82rem; margin-bottom: 20px; padding: 10px 14px;
    background: rgba(74,80,64,0.08); border-left: 4px solid #4a5040;
  }}
  .meta-grid .lbl {{ font-weight: bold; letter-spacing: 0.08em; color: #4a5040; }}
  .meta-grid .val {{ color: #1a1a1a; }}
  .section {{ margin-bottom: 22px; page-break-inside: avoid; }}
  .section h2 {{
    font-size: 0.95rem; letter-spacing: 0.15em; margin: 0 0 6px;
    padding: 4px 8px; background: #4a5040; color: #f4efe2;
    border-left: 6px solid #6b3a1f;
    text-transform: uppercase;
  }}
  .section h3 {{
    font-size: 0.82rem; letter-spacing: 0.1em; margin: 10px 0 4px;
    color: #4a5040; text-transform: uppercase; border-bottom: 1px dotted #8a8470;
    padding-bottom: 2px;
  }}
  .section p {{ margin: 6px 0; font-size: 0.86rem; line-height: 1.5; font-family: "Georgia", "Times New Roman", serif; }}
  .section .empty {{ color: #888; font-style: italic; }}
  .ccir-list {{ padding-left: 22px; margin: 4px 0; }}
  .ccir-list li {{ font-size: 0.86rem; line-height: 1.5; margin: 4px 0; font-family: "Georgia", "Times New Roman", serif; }}
  .terrain-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 8px 18px; }}
  .terrain-grid .sub {{ background: rgba(74,80,64,0.05); padding: 8px 10px; border-left: 3px solid #6b3a1f; }}
  .terrain-grid .sub h3 {{ margin-top: 0; border: none; }}
  .footer {{
    margin-top: 28px; padding-top: 12px; border-top: 2px solid #4a5040;
    font-size: 0.74rem; color: #4a5040; display: grid;
    grid-template-columns: 1fr 1fr; gap: 4px 24px;
  }}
  .stamp {{
    position: absolute; top: 80px; right: 40px;
    border: 3px double #8b0000; color: #8b0000; padding: 6px 12px;
    transform: rotate(-8deg); font-weight: bold; letter-spacing: 0.15em;
    font-size: 0.78rem; opacity: 0.85;
  }}
  .controls {{
    max-width: 820px; margin: 0 auto 14px; display: flex; gap: 8px; justify-content: flex-end;
  }}
  .controls button {{
    background: #4a5040; color: #f4efe2; border: 1px solid #2a2d28;
    padding: 8px 16px; font-family: "Courier New", monospace; font-size: 0.82rem;
    letter-spacing: 0.1em; cursor: pointer; text-transform: uppercase;
  }}
  .controls button:hover {{ background: #6b3a1f; }}
  @media print {{
    body {{ background: #fff; padding: 0; }}
    .page {{ box-shadow: none; border: none; }}
    .controls {{ display: none; }}
    .stamp {{ opacity: 1; }}
  }}
</style>
</head>
<body>
<div class="controls">
  <button onclick="window.print()">Print / Save as PDF</button>
  <button onclick="window.close()">Close</button>
</div>
<div class="page">
  <div class="classification top">{html.escape(classification)}</div>
  <div class="body">
    <div class="stamp">DRAFT</div>
    <div class="doc-title">INTELLIGENCE SUMMARY</div>
    <div class="doc-subtitle">(INTSUM)</div>

    <div class="meta-grid">
      <div><span class="lbl">DTG:</span> <span class="val">{dtg}</span></div>
      <div><span class="lbl">SERIAL:</span> <span class="val">{serial}</span></div>
      <div><span class="lbl">AREA OF INTEREST:</span> <span class="val">{html.escape(label)}</span></div>
      <div><span class="lbl">AREA:</span> <span class="val">{area_sqkm:.1f} km²</span></div>
      <div style="grid-column: span 2"><span class="lbl">BOUNDS (W,S,E,N):</span> <span class="val">{bounds_str}</span></div>
      <div><span class="lbl">ORIGINATOR:</span> <span class="val">IPB-AI ANALYSIS CELL</span></div>
      <div><span class="lbl">SOURCES:</span> <span class="val">OSINT (NLS, Digiroad, FMI, OSM, OpenCellID)</span></div>
    </div>

    <div class="section">
      <h2>1. Situation Overview</h2>
      {_para(sections.get("situation_overview", ""))}
    </div>

    <div class="section">
      <h2>2. Terrain Analysis (OAKOC)</h2>
      <div class="terrain-grid">
        <div class="sub">
          <h3>a. Observation &amp; Fields of Fire</h3>
          {_para(sections.get("terrain_observation", ""))}
        </div>
        <div class="sub">
          <h3>b. Avenues of Approach</h3>
          {_para(sections.get("terrain_approach", ""))}
        </div>
        <div class="sub">
          <h3>c. Key Terrain</h3>
          {_para(sections.get("terrain_key", ""))}
        </div>
        <div class="sub">
          <h3>d. Obstacles</h3>
          {_para(sections.get("terrain_obstacles", ""))}
        </div>
        <div class="sub" style="grid-column: span 2">
          <h3>e. Cover &amp; Concealment</h3>
          {_para(sections.get("terrain_cover", ""))}
        </div>
      </div>
    </div>

    <div class="section">
      <h2>3. Weather Impact</h2>
      {_para(sections.get("weather_impact", ""))}
    </div>

    <div class="section">
      <h2>4. Critical Infrastructure</h2>
      {_para(sections.get("infrastructure", ""))}
    </div>

    <div class="section">
      <h2>5. Civil Considerations</h2>
      {_para(sections.get("civil_considerations", ""))}
    </div>

    <div class="section">
      <h2>6. CCIR — Commander's Critical Information Requirements</h2>
      {_ccir(sections.get("ccir_answers", ""))}
    </div>

    <div class="section">
      <h2>7. Analyst Assessment</h2>
      {_para(sections.get("assessment", ""))}
    </div>

    <div class="section">
      <h2>8. Intelligence Gaps &amp; Limitations</h2>
      {_para(sections.get("limitations", ""))}
    </div>

    <div class="footer">
      <div><span class="lbl">PREPARED BY:</span> {html.escape(provider_label)}</div>
      <div><span class="lbl">DISTRIBUTION:</span> COMMAND GROUP</div>
      <div><span class="lbl">NEXT INTSUM:</span> +24h</div>
      <div><span class="lbl">REFERENCE:</span> Open-source intelligence inputs</div>
    </div>
  </div>
  <div class="classification bottom">{html.escape(classification)}</div>
</div>
</body>
</html>"""
