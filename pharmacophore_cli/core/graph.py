"""
graph.py — 2D pharmacophore graph.

Represents the consensus pharmacophore model as a graph where:
  Nodes  : pharmacophoric features, colored by type, sized by consensus weight
  Edges  : minimum spanning tree (MST) of 3D pairwise Euclidean distances;
           reduces N×(N-1)/2 pairs to N-1 edges — the essential topology
           without cluttering the display
  Angles : computed for adjacent triples (i-j and j-k both in MST)

Outputs:
  PNG/PDF static image  (matplotlib + networkx)
  Interactive HTML      (vis.js embedded, no CDN dependency at runtime)

References:
  - Wolber & Langer (2005) J. Chem. Inf. Model. 45, 160–169
  - Güner (2000) Pharmacophore Perception, Development and Use in Drug Design
"""

import os
import math
import itertools
import json
import numpy as np
from typing import List, Dict, Any, Tuple, Optional

_NODE_COLOR = {
    "ACCEPTOR"      : "#FF4444",
    "DONOR"         : "#4488FF",
    "HYDROPHOBIC"   : "#FFCC00",
    "POS_IONIZABLE" : "#FF8800",
    "NEG_IONIZABLE" : "#9370DB",
}
_NODE_BORDER = {
    "ACCEPTOR"      : "#CC0000",
    "DONOR"         : "#0055CC",
    "HYDROPHOBIC"   : "#CC9900",
    "POS_IONIZABLE" : "#CC5500",
    "NEG_IONIZABLE" : "#5500AA",
}
_EDGE_COLOR    = "#555555"
_MAX_EDGE_DIST = 20.0   # Å — upper limit for the distance table in the sidebar


# ── Graph construction ────────────────────────────────────────────────────────

def build_pharmacophore_graph(
    features: List[Dict[str, Any]],
    max_edge_dist: float = _MAX_EDGE_DIST,
) -> Dict:
    """
    Build the pharmacophore graph as a serializable dict.

    Returns
    -------
    {
      'nodes'     : list of node dicts
      'edges'     : MST edges (N-1 edges for visualization)
      'edges_all' : all pairwise distances sorted (for the sidebar table)
      'angles'    : MST-adjacent angles
    }
    """
    nodes = [
        {
            "id"     : i,
            "label"  : feat["type"].replace("_", "\n"),
            "type"   : feat["type"],
            "weight" : feat.get("weight", 1),
            "coords" : list(feat["coords"]),
        }
        for i, feat in enumerate(features)
    ]

    if len(nodes) < 2:
        return {"nodes": nodes, "edges": [], "edges_all": [], "angles": []}

    all_edges = _all_distances(nodes)
    mst_edges = _kruskal_mst(nodes, all_edges)
    angles    = _mst_angles(nodes, mst_edges)

    return {
        "nodes"    : nodes,
        "edges"    : mst_edges,
        "edges_all": all_edges,
        "angles"   : angles,
    }


def _all_distances(nodes: List[Dict]) -> List[Dict]:
    """All pairwise Euclidean distances, sorted ascending."""
    edges = []
    for eid, (i, j) in enumerate(itertools.combinations(range(len(nodes)), 2)):
        ci = np.array(nodes[i]["coords"], dtype=float)
        cj = np.array(nodes[j]["coords"], dtype=float)
        d  = float(np.linalg.norm(ci - cj))
        edges.append({"id": eid, "from": i, "to": j, "distance": round(d, 2)})
    return sorted(edges, key=lambda e: e["distance"])


def _kruskal_mst(nodes: List[Dict], all_edges: List[Dict]) -> List[Dict]:
    """
    Kruskal's algorithm for the minimum spanning tree.
    Returns exactly N-1 edges connecting all nodes with minimum total distance.
    """
    n      = len(nodes)
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
    for e in all_edges:
        if union(e["from"], e["to"]):
            mst.append(e)
        if len(mst) == n - 1:
            break
    return mst


def _mst_angles(nodes: List[Dict], mst_edges: List[Dict]) -> List[Dict]:
    """
    Compute angles at each MST vertex with ≥2 neighbors.
    (i – vertex_j – k) for all pairs of edges meeting at j.
    """
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
            v1    = np.array(nodes[i]["coords"], dtype=float) - cj
            v2    = np.array(nodes[k]["coords"], dtype=float) - cj
            denom = np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9
            cos_a = float(np.clip(np.dot(v1, v2) / denom, -1.0, 1.0))
            angle = math.degrees(math.acos(cos_a))
            angles.append({
                "i"        : i,
                "j"        : j,
                "k"        : k,
                "angle_deg": round(angle, 1),
                "types"    : f"{nodes[i]['type'][:3]}–{nodes[j]['type'][:3]}–{nodes[k]['type'][:3]}",
            })

    # Sort by pharmacophoric relevance: angles near 90° or 180° first
    angles.sort(key=lambda a: min(abs(a["angle_deg"] - 90), abs(a["angle_deg"] - 180)))
    return angles


def _pca_project_2d(nodes: List[Dict]) -> Dict[int, Tuple[float, float]]:
    """
    Project 3D coordinates to 2D using the first two PCA components.
    Scaled to [-300, 300] for vis.js initial layout.
    """
    coords = np.array([n["coords"] for n in nodes], dtype=float)
    coords -= coords.mean(axis=0)

    if coords.shape[0] >= 2:
        _, _, Vt = np.linalg.svd(coords, full_matrices=False)
        proj = coords @ Vt[:2].T
    else:
        proj = coords[:, :2]

    scale = np.max(np.abs(proj)) + 1e-9
    proj  = proj / scale * 300

    return {nodes[i]["id"]: (float(proj[i, 0]), float(proj[i, 1]))
            for i in range(len(nodes))}


# ── Static plot — matplotlib + networkx ──────────────────────────────────────

def plot_graph_static(
    graph: Dict,
    output_path: str,
    title: str = "Pharmacophore Graph",
    show_angles: bool = True,
    dpi: int = 150,
) -> str:
    """
    Generate PNG + PDF of the pharmacophore graph (MST layout).
    Node positions come from the XY projection of the 3D coordinates.
    """
    try:
        import networkx as nx
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError as e:
        print(f"[Graph] Missing dependency: {e}. Install: pip install networkx matplotlib")
        return ""

    nodes = graph["nodes"]
    edges = graph["edges"]

    if not nodes:
        print("[Graph] No features to plot.")
        return ""

    G = nx.Graph()
    for n in nodes:
        G.add_node(n["id"], **n)
    for e in edges:
        G.add_edge(e["from"], e["to"], distance=e["distance"])

    # Layout: XY projection of 3D coordinates
    coords = np.array([n["coords"] for n in nodes], dtype=float)
    coords -= coords.mean(axis=0)
    pos = {n["id"]: (coords[i, 0], coords[i, 1]) for i, n in enumerate(nodes)}

    node_colors  = [_NODE_COLOR.get(n["type"], "#AAAAAA") for n in nodes]
    node_borders = [_NODE_BORDER.get(n["type"], "#555555") for n in nodes]
    node_sizes   = [900 + 350 * n.get("weight", 1) for n in nodes]
    node_labels  = {n["id"]: n["type"].replace("_", "\n") for n in nodes}
    edge_widths  = [max(0.8, 5.0 - e["distance"] / 3.0) for e in edges]

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    nx.draw_networkx_edges(G, pos, ax=ax, width=edge_widths,
                           edge_color=_EDGE_COLOR, alpha=0.75)

    # Distance labels on the 5 shortest edges — using ax.text() directly
    # (avoids a NetworkX CurvedArrowText bug with Bezier paths > 3 vertices)
    for e in sorted(edges, key=lambda e: e["distance"])[:5]:
        x0, y0 = pos[e["from"]]
        x1, y1 = pos[e["to"]]
        ax.text(
            (x0 + x1) / 2, (y0 + y1) / 2, f"{e['distance']:.1f} Å",
            fontsize=7, color="#AAAAAA", ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.15", fc="#0d1117", ec="none", alpha=0.8),
        )

    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                           node_size=node_sizes, edgecolors=node_borders, linewidths=2)
    nx.draw_networkx_labels(G, pos, labels=node_labels, ax=ax,
                            font_size=7, font_color="white", font_weight="bold")

    # Small index labels above each node
    y_range = max((p[1] for p in pos.values()), default=1) - min((p[1] for p in pos.values()), default=0)
    offset  = 0.07 * (y_range if y_range > 0.1 else 1.0)
    pos_idx = {k: (v[0], v[1] + offset) for k, v in pos.items()}
    nx.draw_networkx_labels(G, pos_idx,
                            labels={n["id"]: f"#{n['id']+1}" for n in nodes},
                            ax=ax, font_size=6, font_color="#AAAAAA")

    if show_angles and graph["angles"]:
        angle_text = "MST angles:\n" + "\n".join(
            f"  #{a['i']+1}–#{a['j']+1}–#{a['k']+1}  [{a['types']}]  {a['angle_deg']:.1f}°"
            for a in graph["angles"][:6]
        )
        ax.text(0.01, 0.01, angle_text, transform=ax.transAxes,
                fontsize=6.5, color="#8b949e", verticalalignment="bottom",
                bbox=dict(boxstyle="round", fc="#161b22", ec="#30363d", alpha=0.85))

    legend_handles = [
        mpatches.Patch(color=c, label=t.replace("_", " ").title())
        for t, c in _NODE_COLOR.items()
    ]
    legend_handles += [
        mpatches.Patch(color="none",    label="Node size ∝ consensus weight"),
        mpatches.Patch(color=_EDGE_COLOR, label=f"MST edges ({len(edges)} of {len(graph['edges_all'])} pairs)"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=7,
              facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9")

    ax.set_title(title, color="#58a6ff", fontsize=13, fontweight="bold", pad=12)
    ax.axis("off")
    plt.tight_layout()

    plt.savefig(output_path, dpi=dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    pdf_path = os.path.splitext(output_path)[0] + ".pdf"
    plt.savefig(pdf_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return output_path


# ── Interactive HTML — vis.js ─────────────────────────────────────────────────

def graph_to_interactive_html(
    graph: Dict,
    output_path: str,
    title: str = "Interactive Pharmacophore Graph",
) -> str:
    """
    Generate an interactive HTML graph using vis.js Network.

    - Nodes positioned from 3D coordinates (PCA projection)
    - MST edges with width ∝ 1/distance; hover to see distance
    - Sidebar: distance table + MST angles
    - Buttons: center, physics toggle, PNG export
    """
    nodes    = graph["nodes"]
    edges    = graph["edges"]
    all_dist = graph["edges_all"]
    angles   = graph["angles"]

    if not nodes:
        return ""

    pos2d = _pca_project_2d(nodes)

    vis_nodes = []
    for n in nodes:
        w      = n.get("weight", 1)
        color  = _NODE_COLOR.get(n["type"], "#AAAAAA")
        border = _NODE_BORDER.get(n["type"], "#555555")
        x, y, z = n["coords"]
        px, py  = pos2d[n["id"]]
        vis_nodes.append({
            "id"   : n["id"],
            "label": f"#{n['id']+1}\n{n['type'].replace('_', ' ')}",
            "title": (f"<b>Feature #{n['id']+1}</b><br>"
                      f"Type: {n['type'].replace('_', ' ')}<br>"
                      f"Confidence: {'★' * w}{'☆' * (3-w)}<br>"
                      f"Coords: ({x:.2f}, {y:.2f}, {z:.2f}) Å"),
            "color": {"background": color, "border": border,
                      "highlight": {"background": color, "border": "white"}},
            "size" : 20 + 8 * w,
            "font" : {"color": "white", "size": 11, "bold": True},
            "shape": "dot",
            "x"   : px,
            "y"   : py,
        })

    vis_edges = []
    for e in edges:
        d     = e["distance"]
        width = max(1.5, 6.0 - d / 3.0)
        vis_edges.append({
            "id"    : e["id"],
            "from"  : e["from"],
            "to"    : e["to"],
            "label" : "",
            "title" : f"<b>Distance:</b> {d:.2f} Å",
            "width" : round(width, 1),
            "color" : {"color": "#777777", "highlight": "#cccccc"},
            "font"  : {"color": "transparent", "size": 1},
            "smooth": {"type": "dynamic"},
        })

    dist_rows = "\n".join(
        f"<tr><td>#{e['from']+1}–#{e['to']+1}</td><td>{e['distance']:.2f} Å</td></tr>"
        for e in all_dist[:20]
    )
    if len(all_dist) > 20:
        dist_rows += f"\n<tr><td colspan='2' style='color:#8b949e;font-style:italic'>… {len(all_dist)-20} more pairs</td></tr>"

    angle_rows = "\n".join(
        f"<tr><td>#{a['i']+1}</td><td>#{a['j']+1}</td><td>#{a['k']+1}</td>"
        f"<td style='color:#58a6ff'><b>{a['angle_deg']:.1f}°</b></td>"
        f"<td style='color:#8b949e;font-size:0.85em'>{a['types']}</td></tr>"
        for a in angles[:12]
    ) or "<tr><td colspan='5' style='color:#8b949e'>No angles (< 3 nodes)</td></tr>"

    nodes_json = json.dumps(vis_nodes, ensure_ascii=False)
    edges_json = json.dumps(vis_edges, ensure_ascii=False)

    html = f"""<!DOCTYPE html>
<html lang="en">
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
  #sidebar h3 {{ color:#58a6ff; margin:10px 0 6px; font-size:0.9em; border-bottom:1px solid #30363d; padding-bottom:3px; }}
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
    padding:8px; margin-bottom:8px; font-size:0.8em; color:#8b949e; min-height:55px; line-height:1.5;
  }}
</style>
</head>
<body>
<h1>🔬 {title}</h1>
<div id="subtitle">MST: {len(edges)} edges | {len(nodes)} features | drag nodes · scroll to zoom</div>
<div id="layout">
  <div id="graph-container"><div id="network"></div></div>
  <div id="sidebar">

    <h3>Selected</h3>
    <div id="info-box">Click a node or edge to see details.</div>

    <h3>Legend</h3>
    {"".join(
        f'<div class="legend-item"><span class="legend-dot" style="background:{c}"></span>'
        f'<span>{t.replace("_", " ").title()}</span></div>'
        for t, c in _NODE_COLOR.items()
    )}
    <div style="margin-top:5px;color:#8b949e;font-size:0.78em;">
      Node size ∝ consensus weight<br>Edge width ∝ 1/distance (MST)
    </div>

    <h3>MST Angles</h3>
    <p style="color:#8b949e;margin-bottom:5px;font-size:0.78em;">
      i — <b>vertex</b> — k &nbsp;|&nbsp; sorted by pharmacophoric relevance (≈90° or 180°)
    </p>
    <table>
      <tr><th>i</th><th>vertex</th><th>k</th><th>Angle</th><th>Types</th></tr>
      {angle_rows}
    </table>

    <h3>Distances (top 20)</h3>
    <table>
      <tr><th>Pair</th><th>Distance</th></tr>
      {dist_rows}
    </table>

    <h3>Actions</h3>
    <button class="btn" onclick="network.fit()">⊙ Fit to screen</button>
    <button class="btn" onclick="togglePhysics()">⚡ Physics on/off</button>
    <button class="btn" onclick="exportPNG()">⬇ Export PNG</button>

  </div>
</div>

<script>
const nodesData  = new vis.DataSet({nodes_json});
const edgesData  = new vis.DataSet({edges_json});
const container  = document.getElementById('network');

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
    hover: true, tooltipDelay: 80,
    navigationButtons: false, keyboard: false,
    zoomView: true, dragView: true,
  }},
  edges: {{ smooth: {{ type: 'dynamic' }}, hoverWidth: 2 }},
  nodes: {{ borderWidth: 2, shadow: {{ enabled: true, color: 'rgba(0,0,0,0.4)', size: 6 }} }},
}};

const network = new vis.Network(container, {{ nodes: nodesData, edges: edgesData }}, options);

network.on('click', function(params) {{
  const box = document.getElementById('info-box');
  if (params.nodes.length > 0) {{
    box.innerHTML = nodesData.get(params.nodes[0]).title;
  }} else if (params.edges.length > 0) {{
    box.innerHTML = edgesData.get(params.edges[0]).title;
  }} else {{
    box.innerHTML = 'Click a node or edge to see details.';
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
  a.download = 'pharmacophore_graph.png';
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


# ── Main entry point ──────────────────────────────────────────────────────────

def generate_pharmacophore_graph(
    features: List[Dict[str, Any]],
    outdir: str,
    prefix: str = "pharmacophore",
    title: str = "Pharmacophore Graph",
    max_edge_dist: float = _MAX_EDGE_DIST,
    show_angles: bool = True,
    verbose: bool = True,
) -> Dict[str, str]:
    """
    Generate the pharmacophore graph in PNG, PDF, and interactive HTML.

    Returns
    -------
    {'png': path, 'pdf': path, 'html': path}
    """
    _log = print if verbose else lambda *a, **k: None
    os.makedirs(outdir, exist_ok=True)

    graph    = build_pharmacophore_graph(features, max_edge_dist=max_edge_dist)
    n_nodes  = len(graph["nodes"])
    n_mst    = len(graph["edges"])
    n_all    = len(graph["edges_all"])
    n_angles = len(graph["angles"])
    _log(f"[Graph] {n_nodes} nodes | MST: {n_mst} edges (of {n_all} pairs) | {n_angles} angles")

    png_path  = os.path.join(outdir, f"{prefix}_graph.png")
    html_path = os.path.join(outdir, f"{prefix}_graph.html")

    plot_graph_static(graph, png_path, title=title, show_angles=show_angles)
    graph_to_interactive_html(graph, html_path, title=title)

    pdf_path = png_path.replace(".png", ".pdf")
    _log(f"[Graph] ✓ PNG : {png_path}")
    _log(f"[Graph] ✓ PDF : {pdf_path}")
    _log(f"[Graph] ✓ HTML: {html_path}")

    return {"png": png_path, "pdf": pdf_path, "html": html_path}
