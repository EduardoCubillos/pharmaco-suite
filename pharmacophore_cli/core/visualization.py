"""
visualization.py — Interactive 3D pharmacophore viewer.

Generates a self-contained HTML file that can be opened in any browser.
Embeds the protein PDB as base64, renders protein/ligand/active site
with 3Dmol.js, and overlays pharmacophoric spheres color-coded by type.

Important: sphere dimensions must be in CSS (width/height on the #viewer
div with position:relative + overflow:hidden), NOT in createViewer() JS
arguments — passing dimensions in JS breaks the viewer on some browsers.
"""

import os
import base64
from typing import List, Dict, Any, Optional

# Color palette per feature type
COLOR_MAP = {
    "ACCEPTOR"      : "#FF4444",   # red
    "DONOR"         : "#4488FF",   # blue
    "HYDROPHOBIC"   : "#FFCC00",   # yellow
    "POS_IONIZABLE" : "#FF8800",   # orange
    "NEG_IONIZABLE" : "#9370DB",   # violet
}

# Sphere style per consensus weight level
_WEIGHT_STYLE = {
    3: {"radius": 1.5, "opacity": 0.95},
    2: {"radius": 1.0, "opacity": 0.70},
    1: {"radius": 0.6, "opacity": 0.40},
}
_DEFAULT_SPHERE_STYLE = {"radius": 0.9, "opacity": 0.65}


def generate_html_viewer(
    pdb_file: str,
    features_dict: Dict[str, List[Dict[str, Any]]],
    output_html: str,
    ligand_resname: str = "RIT",
    title: str = "Pharmacophore",
    active_site_residues: Optional[List[str]] = None,
    width: int = 900,
    height: int = 600,
) -> str:
    """
    Generate an interactive 3D HTML viewer using embedded 3Dmol.js.

    Parameters
    ----------
    pdb_file             : protein PDB (embedded as base64 in the HTML)
    features_dict        : {model_name: [feature_list]} — each list must be
                           pre-clustered before calling this function
    output_html          : output path
    ligand_resname       : residue name of the ligand in the PDB
    title                : viewer title
    active_site_residues : list of 'ResnameResidChain' strings to highlight
    width, height        : viewer dimensions in pixels

    Returns
    -------
    Path to the generated HTML file.
    """
    with open(pdb_file) as fh:
        pdb_text = fh.read()
    pdb_b64 = base64.b64encode(pdb_text.encode()).decode()

    spheres_js    = _spheres_to_js(features_dict)
    legend_html   = _build_legend(features_dict)
    checkboxes    = _build_checkboxes(features_dict)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{title}</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.0.6/3Dmol-min.js"></script>
<style>
  body {{
    margin: 0; padding: 0;
    background: #0d1117;
    color: #c9d1d9;
    font-family: 'Segoe UI', system-ui, sans-serif;
  }}
  h1 {{
    text-align: center;
    color: #58a6ff;
    padding: 16px 0 4px;
    font-size: 1.3em;
    margin: 0;
  }}
  .subtitle {{
    text-align: center;
    color: #8b949e;
    font-size: 0.85em;
    margin-bottom: 8px;
  }}
  #container {{
    display: flex;
    flex-direction: column;
    align-items: center;
  }}
  #viewer {{
    width: {width}px;
    height: {height}px;
    border: 1px solid #30363d;
    border-radius: 8px;
    position: relative;
    overflow: hidden;
  }}
  .controls {{
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
    justify-content: center;
    margin: 10px 0;
    padding: 8px 16px;
    background: #161b22;
    border-radius: 8px;
    border: 1px solid #30363d;
  }}
  .ctrl-group {{
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 0.9em;
  }}
  label {{ cursor: pointer; user-select: none; }}
  .legend {{
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
    justify-content: center;
    margin: 6px 0 10px;
    font-size: 0.82em;
  }}
  .leg-item {{
    display: flex;
    align-items: center;
    gap: 4px;
  }}
  .leg-dot {{
    width: 12px; height: 12px;
    border-radius: 50%;
    display: inline-block;
  }}
  .stats {{
    margin: 4px 16px 10px;
    padding: 8px 16px;
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 6px;
    font-size: 0.82em;
    color: #8b949e;
  }}
  .stats b {{ color: #c9d1d9; }}
</style>
</head>
<body>
<div id="container">
  <h1>🔬 {title}</h1>
  <div class="subtitle">
    Interactive 3D pharmacophore — left-click to rotate · scroll to zoom · right-click to translate
  </div>

  {legend_html}

  <div class="controls">
    <div class="ctrl-group">
      <b>Protein:</b>
      <label><input type="checkbox" id="chk_protein" checked onchange="toggleProtein()"> cartoon</label>
    </div>
    <div class="ctrl-group">
      <b>Ligand:</b>
      <label><input type="checkbox" id="chk_ligand" checked onchange="toggleLigand()"> sticks</label>
    </div>
    <div class="ctrl-group">
      <b>Binding site:</b>
      <label><input type="checkbox" id="chk_site" checked onchange="toggleSite()"> sticks (magenta)</label>
    </div>
    <div class="ctrl-group">
      <b>Models:</b>
      {checkboxes}
    </div>
  </div>

  <div id="viewer"></div>
  {_build_stats(features_dict)}
</div>

<script>
const pdb_b64  = "{pdb_b64}";
const pdb_text = atob(pdb_b64);

{spheres_js}

let viewer = null;

function initViewer() {{
  viewer = $3Dmol.createViewer("viewer", {{
    backgroundColor: "0x0d1117",
  }});

  viewer.addModel(pdb_text, "pdb");

  // Protein cartoon
  viewer.setStyle({{}}, {{cartoon: {{color: "#cccccc", opacity: 0.5}}}});

  // Ligand sticks
  viewer.setStyle(
    {{resn: "{ligand_resname}"}},
    {{stick: {{colorscheme: "elementWithCarbon", radius: 0.15}}}}
  );

  renderSite();
  renderSpheres();

  viewer.zoomTo({{resn: "{ligand_resname}"}});
  viewer.render();
}}

function renderSite() {{
  const siteRes = {_js_array(active_site_residues or [])};
  if (!siteRes.length) return;
  siteRes.forEach(function(r) {{
    const rnum = parseInt(r.match(/\\d+/)[0]);
    viewer.setStyle({{resi: rnum}}, {{stick: {{color: "magenta", radius: 0.12}}}});
  }});
}}

function renderSpheres() {{
  viewer.removeAllShapes();
  renderSite();

  for (const [name, spheres] of Object.entries(PHARMACOPHORE_DATA)) {{
    const chk = document.getElementById("chk_" + name);
    if (chk && !chk.checked) continue;
    spheres.forEach(function(s) {{
      viewer.addSphere({{
        center: {{x: s.x, y: s.y, z: s.z}},
        radius: s.radius,
        color: s.color,
        opacity: s.opacity,
      }});
    }});
  }}
  viewer.render();
}}

function toggleProtein() {{
  const chk = document.getElementById("chk_protein");
  viewer.setStyle({{}}, {{cartoon: {{color: "#cccccc", opacity: chk.checked ? 0.5 : 0.0}}}});
  viewer.render();
}}

function toggleLigand() {{
  const chk = document.getElementById("chk_ligand");
  viewer.setStyle(
    {{resn: "{ligand_resname}"}},
    chk.checked ? {{stick: {{colorscheme: "elementWithCarbon", radius: 0.15}}}} : {{}}
  );
  viewer.render();
}}

function toggleSite() {{
  renderSpheres();
}}

window.onload = initViewer;
</script>
</body>
</html>
"""

    with open(output_html, "w") as fh:
        fh.write(html)

    return output_html


# ── HTML/JS helpers ───────────────────────────────────────────────────────────

def _spheres_to_js(features_dict: Dict[str, List[Dict]]) -> str:
    """Build the PHARMACOPHORE_DATA JavaScript object."""
    lines = ["const PHARMACOPHORE_DATA = {"]
    for name, features in features_dict.items():
        key = _safe_js_key(name)
        lines.append(f"  {key}: [")
        for f in features:
            x, y, z = [float(c) for c in f["coords"]]
            color   = COLOR_MAP.get(f["type"], "#00FF00")
            weight  = f.get("weight")
            style   = _WEIGHT_STYLE.get(weight, _DEFAULT_SPHERE_STYLE) if weight else _DEFAULT_SPHERE_STYLE
            r, o    = style["radius"], style["opacity"]
            lines.append(
                f"    {{x:{x:.3f}, y:{y:.3f}, z:{z:.3f}, "
                f'radius:{r}, color:"{color}", opacity:{o}}},'
            )
        lines.append("  ],")
    lines.append("};")
    return "\n".join(lines)


def _build_legend(features_dict: Dict[str, List[Dict]]) -> str:
    items = [
        f'<div class="leg-item">'
        f'<span class="leg-dot" style="background:{color}"></span>'
        f'{ftype.replace("_", " ").title()}</div>'
        for ftype, color in COLOR_MAP.items()
    ]
    if any("weight" in f for feats in features_dict.values() for f in feats):
        items.append(
            '<div class="leg-item">|&nbsp;<b>Size</b>: '
            '★★★ large=triple &nbsp;★★☆ medium=double &nbsp;★☆☆ small=single</div>'
        )
    return '<div class="legend">' + "".join(items) + "</div>"


def _build_checkboxes(features_dict: Dict[str, List[Dict]]) -> str:
    return " ".join(
        f'<label><input type="checkbox" id="chk_{name.replace(" ", "_")}" checked '
        f'onchange="renderSpheres()"> {name}</label>'
        for name in features_dict
    )


def _build_stats(features_dict: Dict[str, List[Dict]]) -> str:
    from collections import Counter
    rows = []
    for name, features in features_dict.items():
        cnt   = Counter(f["type"] for f in features)
        total = len(features)
        row   = f"<b>{name}</b> ({total} total) — " + " | ".join(
            f"{t}: {n}" for t, n in sorted(cnt.items())
        )
        rows.append(f"<div>{row}</div>")
    return '<div class="stats">' + "\n".join(rows) + "</div>"


def _js_array(lst) -> str:
    return "[" + ", ".join(f'"{item}"' for item in lst) + "]"


def _safe_js_key(s: str) -> str:
    """Turn a model name into a quoted JS object key."""
    return '"' + s.replace(" ", "_").replace("-", "_") + '"'
