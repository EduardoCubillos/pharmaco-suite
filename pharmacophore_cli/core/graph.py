"""
graph.py — Grafo farmacofórico 2D.

Representa el modelo farmacofórico como un grafo donde:
  - Nodos   : features (DONOR, ACCEPTOR, HYDROPHOBIC, POS/NEG_IONIZABLE)
              Tamaño proporcional al nivel de consenso (weight 1-3)
  - Aristas : árbol de expansión mínima (MST) sobre las distancias euclidianas 3D
              Exactamente N-1 aristas — conecta todos los features con mínima
              longitud total, mostrando la estructura espacial sin ruido visual
  - Ángulos : calculados para tríos adyacentes en el MST (i-j y j-k aristas del MST)
              Representan la geometría local del farmacóforo

Salidas:
  - PNG/PDF estático  (matplotlib + networkx)
  - HTML interactivo  (vis.js embebido, sin dependencias externas)

Referencias:
  - Wolber & Langer (2005) J. Chem. Inf. Model. 45, 160–169
  - Güner (2000) Pharmacophore Perception, Development and Use in Drug Design
"""

import os
import math
import itertools
import json
import numpy as np
from typing import List, Dict, Any, Tuple, Optional

# ── Paleta de colores ──────────────────────────────────────────────────────────
NODE_COLORS = {
    "ACCEPTOR"      : "#FF4444",
    "DONOR"         : "#4488FF",
    "HYDROPHOBIC"   : "#FFCC00",
    "POS_IONIZABLE" : "#FF8800",
    "NEG_IONIZABLE" : "#9370DB",
}
NODE_BORDER = {
    "ACCEPTOR"      : "#CC0000",
    "DONOR"         : "#0055CC",
    "HYDROPHOBIC"   : "#CC9900",
    "POS_IONIZABLE" : "#CC5500",
    "NEG_IONIZABLE" : "#5500AA",
}
EDGE_COLOR   = "#555555"
MAX_EDGE_DIST = 20.0    # Å — umbral para calcular distancias (tabla sidebar)


# ═════════════════════════════════════════════════════════════════════════════
#  ALGORITMOS AUXILIARES
# ═════════════════════════════════════════════════════════════════════════════

def _compute_all_distances(nodes: List[Dict]) -> List[Dict]:
    """Calcula todas las distancias euclidianas entre pares de nodos."""
    edges = []
    eid = 0
    for i, j in itertools.combinations(range(len(nodes)), 2):
        ci = np.array(nodes[i]["coords"], dtype=float)
        cj = np.array(nodes[j]["coords"], dtype=float)
        d  = float(np.linalg.norm(ci - cj))
        edges.append({
            "id"      : eid,
            "from"    : i,
            "to"      : j,
            "distance": round(d, 2),
        })
        eid += 1
    return sorted(edges, key=lambda e: e["distance"])


def _compute_mst(nodes: List[Dict], all_edges: List[Dict]) -> List[Dict]:
    """
    Árbol de expansión mínima (MST) por algoritmo de Kruskal.
    Devuelve exactamente N-1 aristas que conectan todos los nodos
    con mínima suma de distancias.
    """
    n = len(nodes)
    if n <= 1:
        return []

    parent = list(range(n))
    rank   = [0] * n

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        px, py = find(x), find(y)
        if px == py:
            return False
        if rank[px] < rank[py]:
            px, py = py, px
        parent[py] = px
        if rank[px] == rank[py]:
            rank[px] += 1
        return True

    mst = []
    for e in all_edges:  # ya vienen ordenados por distancia
        if union(e["from"], e["to"]):
            mst.append(e)
        if len(mst) == n - 1:
            break

    return mst


def _compute_mst_angles(nodes: List[Dict], mst_edges: List[Dict]) -> List[Dict]:
    """
    Calcula ángulos para cada trío (i, j, k) donde j es vértice y tanto
    la arista i-j como j-k pertenecen al MST.

    Esto captura la geometría *local* del farmacóforo a lo largo del árbol.
    """
    # Construir lista de adyacencia del MST
    adj: Dict[int, List[int]] = {n["id"]: [] for n in nodes}
    for e in mst_edges:
        adj[e["from"]].append(e["to"])
        adj[e["to"]].append(e["from"])

    angles = []
    for j, neighbors in adj.items():
        if len(neighbors) < 2:
            continue
        cj = np.array(nodes[j]["coords"], dtype=float)
        for i, k in itertools.combinations(neighbors, 2):
            ci = np.array(nodes[i]["coords"], dtype=float)
            ck = np.array(nodes[k]["coords"], dtype=float)
            v1 = ci - cj
            v2 = ck - cj
            denom = np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9
            cos_a = float(np.clip(np.dot(v1, v2) / denom, -1.0, 1.0))
            angle = math.degrees(math.acos(cos_a))
            ti = nodes[i]["type"]
            tj = nodes[j]["type"]
            tk = nodes[k]["type"]
            angles.append({
                "i"        : i,
                "j"        : j,
                "k"        : k,
                "angle_deg": round(angle, 1),
                "types"    : f"{ti[:3]}–{tj[:3]}–{tk[:3]}",
            })

    # Ordenar: primero los más cercanos a 90° o 180° (más relevantes farmacofóricamente)
    angles.sort(key=lambda a: min(abs(a["angle_deg"] - 90), abs(a["angle_deg"] - 180)))
    return angles


def _project_2d(nodes: List[Dict]) -> Dict[int, Tuple[float, float]]:
    """
    Proyecta coordenadas 3D a 2D usando los dos primeros componentes PCA.
    Devuelve dict {node_id: (x, y)} escalado a [-300, 300] para vis.js.
    """
    coords = np.array([n["coords"] for n in nodes], dtype=float)
    coords -= coords.mean(axis=0)

    if coords.shape[0] >= 2 and coords.shape[1] >= 2:
        # PCA manual: SVD sobre la matriz centrada
        U, S, Vt = np.linalg.svd(coords, full_matrices=False)
        proj = coords @ Vt[:2].T   # (N, 2)
    else:
        proj = coords[:, :2]

    # Escalar al rango [-300, 300]
    scale = np.max(np.abs(proj)) + 1e-9
    proj  = proj / scale * 300

    return {nodes[i]["id"]: (float(proj[i, 0]), float(proj[i, 1]))
            for i in range(len(nodes))}


# ═════════════════════════════════════════════════════════════════════════════
#  CONSTRUCCIÓN DEL GRAFO
# ═════════════════════════════════════════════════════════════════════════════

def build_pharmacophore_graph(
    features: List[Dict[str, Any]],
    max_edge_dist: float = MAX_EDGE_DIST,
) -> Dict:
    """
    Construye el grafo farmacofórico como un dict serializable.

    Returns
    -------
    {
      'nodes'     : [{'id', 'label', 'type', 'weight', 'coords'}],
      'edges'     : MST edges [{'id', 'from', 'to', 'distance'}]  ← para visualización
      'edges_all' : todas las distancias ordenadas                 ← para tabla sidebar
      'angles'    : ángulos MST-adyacentes [{'i','j','k','angle_deg','types'}]
    }
    """
    nodes = []
    for i, feat in enumerate(features):
        nodes.append({
            "id"     : i,
            "label"  : feat["type"].replace("_", "\n"),
            "type"   : feat["type"],
            "weight" : feat.get("weight", 1),
            "coords" : list(feat["coords"]),
        })

    if len(nodes) < 2:
        return {"nodes": nodes, "edges": [], "edges_all": [], "angles": []}

    all_edges = _compute_all_distances(nodes)         # todos los pares, ordenados
    mst_edges = _compute_mst(nodes, all_edges)        # N-1 aristas del MST
    angles    = _compute_mst_angles(nodes, mst_edges) # ángulos locales del MST

    return {
        "nodes"    : nodes,
        "edges"    : mst_edges,
        "edges_all": all_edges,
        "angles"   : angles,
    }


# ═════════════════════════════════════════════════════════════════════════════
#  GRÁFICO ESTÁTICO — matplotlib + networkx
# ═════════════════════════════════════════════════════════════════════════════

def plot_graph_static(
    graph: Dict,
    output_path: str,
    title: str = "Grafo Farmacofórico",
    show_angles: bool = True,
    dpi: int = 150,
) -> str:
    """
    Genera un PNG + PDF del grafo usando las aristas del MST.
    Layout: proyección XY de coordenadas 3D reales (representación espacial fiel).
    """
    try:
        import networkx as nx
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError as e:
        print(f"[Graph] Dependencia faltante: {e}. Instala: pip install networkx matplotlib")
        return ""

    nodes = graph["nodes"]
    edges = graph["edges"]        # MST

    if not nodes:
        print("[Graph] No hay features para graficar.")
        return ""

    # ── Construir grafo NetworkX con MST ──────────────────────────────────────
    G = nx.Graph()
    for n in nodes:
        G.add_node(n["id"], **n)
    for e in edges:
        G.add_edge(e["from"], e["to"], distance=e["distance"])

    # ── Layout: proyección XY de coordenadas 3D ───────────────────────────────
    coords = np.array([n["coords"] for n in nodes], dtype=float)
    coords -= coords.mean(axis=0)
    pos = {n["id"]: (coords[i, 0], coords[i, 1]) for i, n in enumerate(nodes)}

    # ── Colores y tamaños ─────────────────────────────────────────────────────
    node_colors  = [NODE_COLORS.get(n["type"], "#AAAAAA") for n in nodes]
    node_borders = [NODE_BORDER.get(n["type"], "#555555") for n in nodes]
    node_sizes   = [900 + 350 * n.get("weight", 1) for n in nodes]
    node_labels  = {n["id"]: n["type"].replace("_", "\n") for n in nodes}

    # Grosor de aristas proporcional a 1/distancia
    edge_widths = [max(0.8, 5.0 - e["distance"] / 3.0) for e in edges]

    # ── Figura ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    # Aristas MST (sin etiquetas de distancia — demasiado ruido visual)
    nx.draw_networkx_edges(
        G, pos, ax=ax,
        width=edge_widths,
        edge_color=EDGE_COLOR,
        alpha=0.75,
    )

    # Etiquetas de distancia solo en las aristas más cortas (top 5)
    # Nota: se usa ax.text manual para evitar bug de NetworkX con CurvedArrowText
    short_edges = sorted(edges, key=lambda e: e["distance"])[:5]
    for e in short_edges:
        x0, y0 = pos[e["from"]]
        x1, y1 = pos[e["to"]]
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        ax.text(
            mx, my, f"{e['distance']:.1f} Å",
            fontsize=7, color="#AAAAAA", ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.15", fc="#0d1117", ec="none", alpha=0.8),
        )

    # Nodos
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color=node_colors,
        node_size=node_sizes,
        edgecolors=node_borders,
        linewidths=2,
    )

    # Etiquetas de tipo en nodos
    nx.draw_networkx_labels(
        G, pos, labels=node_labels, ax=ax,
        font_size=7, font_color="white", font_weight="bold",
    )

    # Índices de nodos (pequeños, encima)
    idx_labels = {n["id"]: f"#{n['id']+1}" for n in nodes}
    y_range = max((p[1] for p in pos.values()), default=1) - min((p[1] for p in pos.values()), default=0)
    offset  = 0.07 * (y_range if y_range > 0.1 else 1.0)
    pos_idx = {k: (v[0], v[1] + offset) for k, v in pos.items()}
    nx.draw_networkx_labels(
        G, pos_idx, labels=idx_labels, ax=ax,
        font_size=6, font_color="#AAAAAA",
    )

    # ── Ángulos MST (top 6 más relevantes) ───────────────────────────────────
    if show_angles and graph["angles"]:
        top_a = graph["angles"][:6]
        angle_text = "Ángulos (vértices MST):\n" + "\n".join(
            f"  #{a['i']+1}–#{a['j']+1}–#{a['k']+1}  [{a['types']}]  {a['angle_deg']:.1f}°"
            for a in top_a
        )
        ax.text(
            0.01, 0.01, angle_text,
            transform=ax.transAxes,
            fontsize=6.5, color="#8b949e",
            verticalalignment="bottom",
            bbox=dict(boxstyle="round", fc="#161b22", ec="#30363d", alpha=0.85),
        )

    # ── Leyenda ───────────────────────────────────────────────────────────────
    legend_handles = [
        mpatches.Patch(color=color, label=ftype.replace("_", " ").title())
        for ftype, color in NODE_COLORS.items()
    ]
    legend_handles.append(
        mpatches.Patch(color="none", label="Tamaño ∝ nivel consenso")
    )
    legend_handles.append(
        mpatches.Patch(color=EDGE_COLOR, label=f"Aristas: MST ({len(edges)} de {len(graph['edges_all'])} pares)")
    )
    ax.legend(
        handles=legend_handles,
        loc="upper right", fontsize=7,
        facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9",
    )

    ax.set_title(title, color="#58a6ff", fontsize=13, fontweight="bold", pad=12)
    ax.axis("off")
    plt.tight_layout()

    # Guardar PNG + PDF
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    pdf_path = os.path.splitext(output_path)[0] + ".pdf"
    plt.savefig(pdf_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return output_path


# ═════════════════════════════════════════════════════════════════════════════
#  HTML INTERACTIVO — vis.js
# ═════════════════════════════════════════════════════════════════════════════

def graph_to_interactive_html(
    graph: Dict,
    output_path: str,
    title: str = "Grafo Farmacofórico Interactivo",
) -> str:
    """
    Genera un HTML interactivo con vis.js Network usando el MST.

    - Nodos con posición inicial desde coordenadas 3D (proyección PCA)
    - Aristas MST: grosor ∝ 1/distancia, sin texto encima (hover para ver valor)
    - Sidebar: tabla de distancias (todas) y ángulos MST
    - Botones: centrar, física on/off, exportar PNG
    """
    nodes    = graph["nodes"]
    edges    = graph["edges"]        # MST
    all_dist = graph["edges_all"]    # todos los pares (para sidebar)
    angles   = graph["angles"]

    if not nodes:
        return ""

    # Proyección 2D para posiciones iniciales
    pos2d = _project_2d(nodes)

    # ── JSON de nodos ─────────────────────────────────────────────────────────
    vis_nodes = []
    for n in nodes:
        w      = n.get("weight", 1)
        color  = NODE_COLORS.get(n["type"], "#AAAAAA")
        border = NODE_BORDER.get(n["type"], "#555555")
        size   = 20 + 8 * w
        x, y, z = n["coords"]
        px, py   = pos2d[n["id"]]
        vis_nodes.append({
            "id"   : n["id"],
            "label": f"#{n['id']+1}\n{n['type'].replace('_',' ')}",
            "title": (f"<b>Feature #{n['id']+1}</b><br>"
                      f"Tipo: {n['type'].replace('_',' ')}<br>"
                      f"Consenso: {'★' * w}{'☆' * (3-w)}<br>"
                      f"Coords: ({x:.2f}, {y:.2f}, {z:.2f}) Å"),
            "color": {"background": color, "border": border,
                      "highlight": {"background": color, "border": "white"}},
            "size" : size,
            "font" : {"color": "white", "size": 11, "bold": True},
            "shape": "dot",
            "x"    : px,
            "y"    : py,
        })

    # ── JSON de aristas MST ───────────────────────────────────────────────────
    vis_edges = []
    for e in edges:
        d     = e["distance"]
        width = max(1.5, 6.0 - d / 3.0)
        vis_edges.append({
            "id"   : e["id"],
            "from" : e["from"],
            "to"   : e["to"],
            # Sin label en el grafo — la distancia va solo en el hover
            "label": "",
            "title": f"<b>Distancia:</b> {d:.2f} Å",
            "width": round(width, 1),
            "color": {"color": "#777777", "highlight": "#cccccc"},
            "font" : {"color": "transparent", "size": 1},
            "smooth": {"type": "dynamic"},
        })

    # ── Tabla de distancias (top 20 más cortas) ───────────────────────────────
    top_dist = all_dist[:20]
    dist_rows = "\n".join(
        f"<tr><td>#{e['from']+1}–#{e['to']+1}</td>"
        f"<td>{e['distance']:.2f} Å</td></tr>"
        for e in top_dist
    )
    if len(all_dist) > 20:
        dist_rows += f"\n<tr><td colspan='2' style='color:#8b949e;font-style:italic'>… {len(all_dist)-20} pares más</td></tr>"

    # ── Tabla de ángulos (MST, top 12) ────────────────────────────────────────
    top_angles = angles[:12]
    angle_rows = "\n".join(
        f"<tr><td>#{a['i']+1}</td><td>#{a['j']+1}</td><td>#{a['k']+1}</td>"
        f"<td style='color:#58a6ff'><b>{a['angle_deg']:.1f}°</b></td>"
        f"<td style='color:#8b949e;font-size:0.85em'>{a['types']}</td></tr>"
        for a in top_angles
    )
    if not angle_rows:
        angle_rows = "<tr><td colspan='5' style='color:#8b949e'>Sin ángulos (< 3 nodos)</td></tr>"

    nodes_json = json.dumps(vis_nodes, ensure_ascii=False)
    edges_json = json.dumps(vis_edges, ensure_ascii=False)

    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<title>{title}</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/vis/4.21.0/vis.min.js"></script>
<link href="https://cdnjs.cloudflare.com/ajax/libs/vis/4.21.0/vis.min.css" rel="stylesheet">
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: #0d1117; color: #c9d1d9; font-family: 'Segoe UI', sans-serif; height:100vh; overflow:hidden; }}
  h1 {{ text-align:center; color:#58a6ff; padding:10px 0 6px; font-size:1.15em; }}
  #subtitle {{ text-align:center; color:#8b949e; font-size:0.78em; margin-bottom:6px; }}
  #layout {{ display:flex; height:calc(100vh - 58px); }}
  #graph-container {{ flex:1; border:1px solid #30363d; border-radius:4px; margin:0 4px 4px 4px; }}
  #network {{ width:100%; height:100%; }}
  #sidebar {{
    width:260px; background:#161b22; border-left:1px solid #30363d;
    overflow-y:auto; padding:10px 12px; font-size:0.8em; flex-shrink:0;
  }}
  #sidebar h3 {{ color:#58a6ff; margin: 10px 0 6px; font-size:0.9em; border-bottom:1px solid #30363d; padding-bottom:3px; }}
  #sidebar h3:first-child {{ margin-top:0; }}
  table {{ width:100%; border-collapse:collapse; margin-bottom:4px; }}
  th {{ background:#21262d; color:#8b949e; padding:3px 5px; font-weight:normal; font-size:0.9em; }}
  td {{ padding:3px 5px; border-bottom:1px solid #21262d; }}
  tr:hover td {{ background:#1f2937; }}
  .legend-item {{ display:flex; align-items:center; gap:6px; margin:3px 0; }}
  .legend-dot {{ width:11px; height:11px; border-radius:50%; flex-shrink:0; }}
  .btn {{
    display:block; width:100%; margin:5px 0; padding:5px 8px;
    background:#21262d; color:#c9d1d9; border:1px solid #30363d;
    border-radius:4px; cursor:pointer; font-size:0.82em; text-align:center;
  }}
  .btn:hover {{ background:#30363d; }}
  #info-box {{
    background:#21262d; border:1px solid #30363d; border-radius:4px;
    padding:8px; margin-bottom:8px; font-size:0.8em; color:#8b949e;
    min-height:55px; line-height:1.5;
  }}
  .stat-badge {{
    display:inline-block; background:#21262d; border:1px solid #30363d;
    border-radius:3px; padding:1px 6px; margin:2px; font-size:0.82em;
    color:#c9d1d9;
  }}
</style>
</head>
<body>
<h1>🔬 {title}</h1>
<div id="subtitle">
  MST: {len(edges)} aristas | {len(nodes)} features | arrastra nodos · zoom con rueda
</div>
<div id="layout">
  <div id="graph-container">
    <div id="network"></div>
  </div>
  <div id="sidebar">

    <h3>Seleccionado</h3>
    <div id="info-box">Clic en un nodo o arista para ver detalles.</div>

    <h3>Leyenda</h3>
    {"".join(
        f'<div class="legend-item"><span class="legend-dot" style="background:{c}"></span>'
        f'<span>{t.replace("_"," ").title()}</span></div>'
        for t, c in NODE_COLORS.items()
    )}
    <div style="margin-top:5px;color:#8b949e;font-size:0.78em;">
      Tamaño nodo ∝ nivel de consenso<br>
      Grosor arista ∝ 1/distancia (MST)
    </div>

    <h3>Ángulos MST</h3>
    <p style="color:#8b949e;margin-bottom:5px;font-size:0.78em;">
      i — <b>vértice</b> — k &nbsp;|&nbsp; ordenados por relevancia (≈90° ó 180°)
    </p>
    <table>
      <tr><th>i</th><th>vértice</th><th>k</th><th>Ángulo</th><th>Tipos</th></tr>
      {angle_rows}
    </table>

    <h3>Distancias (top 20)</h3>
    <table>
      <tr><th>Par</th><th>Distancia</th></tr>
      {dist_rows}
    </table>

    <h3>Acciones</h3>
    <button class="btn" onclick="network.fit()">⊙ Centrar grafo</button>
    <button class="btn" onclick="togglePhysics()">⚡ Física on/off</button>
    <button class="btn" onclick="exportPNG()">⬇ Exportar PNG</button>

  </div>
</div>

<script>
const nodesData = new vis.DataSet({nodes_json});
const edgesData = new vis.DataSet({edges_json});
const container = document.getElementById('network');

const options = {{
  physics: {{
    enabled: true,
    stabilization: {{ iterations: 150, fit: true }},
    barnesHut: {{
      gravitationalConstant: -4000,
      springLength: 180,
      springConstant: 0.04,
      damping: 0.12,
    }},
  }},
  interaction: {{
    hover: true,
    tooltipDelay: 80,
    navigationButtons: false,
    keyboard: false,
    zoomView: true,
    dragView: true,
  }},
  edges: {{
    smooth: {{ type: 'dynamic' }},
    hoverWidth: 2,
  }},
  nodes: {{
    borderWidth: 2,
    shadow: {{ enabled: true, color: 'rgba(0,0,0,0.4)', size: 6 }},
  }},
}};

const network = new vis.Network(
  container,
  {{ nodes: nodesData, edges: edgesData }},
  options
);

network.on('click', function(params) {{
  const box = document.getElementById('info-box');
  if (params.nodes.length > 0) {{
    const n = nodesData.get(params.nodes[0]);
    box.innerHTML = n.title;
  }} else if (params.edges.length > 0) {{
    const e = edgesData.get(params.edges[0]);
    box.innerHTML = e.title;
  }} else {{
    box.innerHTML = 'Clic en un nodo o arista para ver detalles.';
  }}
}});

let physicsOn = true;
function togglePhysics() {{
  physicsOn = !physicsOn;
  network.setOptions({{ physics: {{ enabled: physicsOn }} }});
}}

function exportPNG() {{
  const canvas = container.querySelector('canvas');
  if (!canvas) return;
  const a = document.createElement('a');
  a.download = 'farmacoforo_grafo.png';
  a.href = canvas.toDataURL('image/png');
  a.click();
}}
</script>
</body>
</html>
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    return output_path


# ═════════════════════════════════════════════════════════════════════════════
#  FUNCIÓN PRINCIPAL
# ═════════════════════════════════════════════════════════════════════════════

def generate_pharmacophore_graph(
    features: List[Dict[str, Any]],
    outdir: str,
    prefix: str = "pharmacophore",
    title: str = "Grafo Farmacofórico",
    max_edge_dist: float = MAX_EDGE_DIST,
    show_angles: bool = True,
    verbose: bool = True,
) -> Dict[str, str]:
    """
    Genera el grafo farmacofórico en PNG, PDF y HTML interactivo.

    Returns
    -------
    {'png': path, 'pdf': path, 'html': path}
    """
    _log = print if verbose else lambda *a, **k: None
    os.makedirs(outdir, exist_ok=True)

    graph = build_pharmacophore_graph(features, max_edge_dist=max_edge_dist)
    n_nodes  = len(graph["nodes"])
    n_mst    = len(graph["edges"])
    n_all    = len(graph["edges_all"])
    n_angles = len(graph["angles"])
    _log(f"[Graph] {n_nodes} nodos | MST: {n_mst} aristas (de {n_all} pares) | {n_angles} ángulos")

    png_path  = os.path.join(outdir, f"{prefix}_graph.png")
    html_path = os.path.join(outdir, f"{prefix}_graph.html")

    plot_graph_static(graph, png_path, title=title, show_angles=show_angles)
    graph_to_interactive_html(graph, html_path, title=title)

    pdf_path = png_path.replace(".png", ".pdf")
    _log(f"[Graph] ✓ PNG : {png_path}")
    _log(f"[Graph] ✓ PDF : {pdf_path}")
    _log(f"[Graph] ✓ HTML: {html_path}")

    return {"png": png_path, "pdf": pdf_path, "html": html_path}
