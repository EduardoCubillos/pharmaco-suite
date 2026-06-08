"""
visualization.py — Visualización 3D interactiva con py3Dmol.

Genera un archivo HTML auto-contenido que puede abrirse en cualquier navegador.
Incluye la proteína, el ligando, el sitio activo y las esferas farmacofóricas.

Los modelos se renderizan sobre VERSIONES CLUSTERIZADAS de los features
(nunca sobre los crudos) para evitar superposición de esferas.
"""

import os
import base64
from typing import List, Dict, Any, Optional

# ── Paleta de colores por tipo de feature ────────────────────────────────────
COLOR_MAP = {
    "ACCEPTOR"      : "#FF4444",   # rojo
    "DONOR"         : "#4488FF",   # azul
    "HYDROPHOBIC"   : "#FFCC00",   # amarillo
    "POS_IONIZABLE" : "#FF8800",   # naranja
    "NEG_IONIZABLE" : "#9370DB",   # violeta
}

# Estilo por nivel de consenso {weight: {radius, opacity}}
CONSENSUS_STYLE = {
    3: {"radius": 1.5, "opacity": 0.95},
    2: {"radius": 1.0, "opacity": 0.70},
    1: {"radius": 0.6, "opacity": 0.40},
}
DEFAULT_STYLE = {"radius": 0.9, "opacity": 0.65}


def generate_html_viewer(
    pdb_file: str,
    features_dict: Dict[str, List[Dict[str, Any]]],
    output_html: str,
    ligand_resname: str = "RIT",
    title: str = "Farmacóforo",
    active_site_residues: Optional[List[str]] = None,
    width: int = 900,
    height: int = 600,
) -> str:
    """
    Genera un HTML con visualización 3D interactiva usando py3Dmol embebido.

    Parameters
    ----------
    pdb_file            : ruta al PDB de la proteína
    features_dict       : dict {nombre_modelo → lista_de_features}
                          Ej: {'SBP': [...], 'LBP': [...], 'Consenso': [...]}
                          Cada lista debe estar PRE-CLUSTERIZADA.
    output_html         : archivo HTML de salida
    ligand_resname      : residuo del ligando en el PDB
    title               : título del visor
    active_site_residues: lista de 'ResnameResidChain' para destacar (ej. ['ASP25A'])
    width, height       : dimensiones del visor

    Returns
    -------
    Ruta del archivo HTML generado.
    """
    # Leer PDB como texto y encodear en base64 para embedir en HTML
    with open(pdb_file) as fh:
        pdb_text = fh.read()
    pdb_b64 = base64.b64encode(pdb_text.encode()).decode()

    # Construir datos de esferas por modelo
    spheres_js = _build_spheres_js(features_dict)

    # Construir leyenda HTML
    legend_html = _build_legend_html(features_dict)

    # Construir checkboxes para activar/desactivar modelos
    checkboxes_html = _build_checkboxes(features_dict)

    html = f"""<!DOCTYPE html>
<html lang="es">
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
    Farmacóforo 3D interactivo — rota con clic izquierdo, zoom con rueda, traslada con clic derecho
  </div>

  {legend_html}

  <div class="controls">
    <div class="ctrl-group">
      <b>Proteína:</b>
      <label><input type="checkbox" id="chk_protein" checked onchange="toggleProtein()"> cartoon</label>
    </div>
    <div class="ctrl-group">
      <b>Ligando:</b>
      <label><input type="checkbox" id="chk_ligand" checked onchange="toggleLigand()"> sticks</label>
    </div>
    <div class="ctrl-group">
      <b>Sitio activo:</b>
      <label><input type="checkbox" id="chk_site" checked onchange="toggleSite()"> sticks magenta</label>
    </div>
    <div class="ctrl-group">
      <b>Modelos:</b>
      {checkboxes_html}
    </div>
  </div>

  <div id="viewer"></div>
  {_build_stats_html(features_dict)}
</div>

<script>
// ── PDB embebido ─────────────────────────────────────────────
const pdb_b64 = "{pdb_b64}";
const pdb_text = atob(pdb_b64);

// ── Datos de esferas ─────────────────────────────────────────
{spheres_js}

// ── Viewer global ────────────────────────────────────────────
let viewer = null;
let modelAdded = false;

function initViewer() {{
  viewer = $3Dmol.createViewer("viewer", {{
    backgroundColor: "0x0d1117",
  }});

  viewer.addModel(pdb_text, "pdb");

  // Proteína (cartoon gris)
  viewer.setStyle({{}}, {{cartoon: {{color: "#cccccc", opacity: 0.5}}}});

  // Ligando (sticks blancos/elementales)
  viewer.setStyle(
    {{resn: "{ligand_resname}"}},
    {{stick: {{colorscheme: "elementWithCarbon", radius: 0.15}}}}
  );

  // Sitio activo (sticks magenta, si se especificaron)
  renderSite();

  // Esferas farmacofóricas
  renderSpheres();

  viewer.zoomTo({{resn: "{ligand_resname}"}});
  viewer.render();
}}

function renderSite() {{
  const siteRes = {_js_list(active_site_residues or [])};
  if (!siteRes.length) return;
  siteRes.forEach(function(r) {{
    const rnum = parseInt(r.match(/\\d+/)[0]);
    viewer.setStyle(
      {{resi: rnum}},
      {{stick: {{color: "magenta", radius: 0.12}}}}
    );
  }});
}}

function renderSpheres() {{
  // Limpiar esferas existentes
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


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_spheres_js(features_dict: Dict[str, List[Dict]]) -> str:
    """Genera el objeto JavaScript PHARMACOPHORE_DATA con todas las esferas."""
    lines = ["const PHARMACOPHORE_DATA = {"]
    for name, features in features_dict.items():
        lines.append(f"  {json_key(name)}: [")
        for f in features:
            x, y, z = [float(c) for c in f["coords"]]
            color   = COLOR_MAP.get(f["type"], "#00FF00")
            weight  = f.get("weight")
            if weight is not None:
                style = CONSENSUS_STYLE.get(weight, DEFAULT_STYLE)
            else:
                style = DEFAULT_STYLE
            r = style["radius"]
            o = style["opacity"]
            lines.append(
                f"    {{x:{x:.3f}, y:{y:.3f}, z:{z:.3f}, "
                f'radius:{r}, color:"{color}", opacity:{o}}},'
            )
        lines.append("  ],")
    lines.append("};")
    return "\n".join(lines)


def _build_legend_html(features_dict: Dict[str, List[Dict]]) -> str:
    items = []
    for ftype, color in COLOR_MAP.items():
        items.append(
            f'<div class="leg-item">'
            f'<span class="leg-dot" style="background:{color}"></span>'
            f'{ftype.replace("_"," ").title()}</div>'
        )
    # Añadir leyenda de tamaño para consenso
    if any("weight" in f for feats in features_dict.values() for f in feats):
        items.append(
            '<div class="leg-item">|&nbsp;<b>Tamaño</b>: '
            '★★★ grande=triple &nbsp;★★☆ medio=doble &nbsp;★☆☆ pequeño=único</div>'
        )
    return '<div class="legend">' + "".join(items) + "</div>"


def _build_checkboxes(features_dict: Dict[str, List[Dict]]) -> str:
    parts = []
    for name in features_dict:
        safe = name.replace(" ", "_")
        parts.append(
            f'<label><input type="checkbox" id="chk_{safe}" checked '
            f'onchange="renderSpheres()"> {name}</label>'
        )
    return " ".join(parts)


def _build_stats_html(features_dict: Dict[str, List[Dict]]) -> str:
    from collections import Counter
    parts = []
    for name, features in features_dict.items():
        cnt = Counter(f["type"] for f in features)
        total = len(features)
        row = f"<b>{name}</b> ({total} total) — " + " | ".join(
            f"{t}: {n}" for t, n in sorted(cnt.items())
        )
        parts.append(f"<div>{row}</div>")
    return '<div class="stats">' + "\n".join(parts) + "</div>"


def _js_list(lst) -> str:
    quoted = ", ".join(f'"{item}"' for item in lst)
    return f"[{quoted}]"


def json_key(s: str) -> str:
    """Convierte nombre de modelo a clave JS válida."""
    s = s.replace(" ", "_").replace("-", "_")
    return f'"{s}"'
