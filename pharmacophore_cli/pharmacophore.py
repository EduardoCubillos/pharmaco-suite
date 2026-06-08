#!/usr/bin/env python3
"""
pharmacophore.py — Herramienta CLI para modelado de farmacóforos.

Subcomandos:
  sbp       Farmacóforo Basado en Estructura (PLIP + fallback BioPython)
  lbp       Farmacóforo Basado en Ligando (RDKit, uno o múltiples SMILES)
  rbp       Farmacóforo Basado en Receptor (pH 7.4, sitio activo)
  all       Ejecutar los tres modelos y construir el consenso automáticamente
  consensus Cargar modelos PDB ya generados y construir el consenso

Ejemplos de uso:
  # Descargar 1HXW automáticamente y generar los tres modelos
  python pharmacophore.py all --pdb 1HXW --smiles "CC(C)c1nc(c...)..." --ligand RIT

  # Usar un PDB local
  python pharmacophore.py sbp --pdb-file ./mi_proteina.pdb --ligand LIG

  # Solo LBP con múltiples SMILES desde archivo
  python pharmacophore.py lbp --smiles-file ligandos.smi

  # Construir consenso desde PDBs ya generados
  python pharmacophore.py consensus --sbp sbp.pdb --lbp lbp.pdb --rbp rbp.pdb

Autor: Eduardo Cubillos-Llantén / pharmacophore_tool
"""

import argparse
import os
import sys
import textwrap

# ── Asegurar que el paquete local se importa ──────────────────────────────────
_TOOL_DIR = os.path.dirname(os.path.abspath(__file__))
if _TOOL_DIR not in sys.path:
    sys.path.insert(0, _TOOL_DIR)

from core.clustering   import cluster_features, CLUSTER_RADII
from core.sbp          import generate_sbp
from core.lbp          import generate_lbp
from core.rbp          import generate_rbp
from core.consensus     import build_consensus, load_features_from_pdb
from core.io_pdb        import write_pharmacophore_pdb
from core.visualization import generate_html_viewer
from core.graph         import generate_pharmacophore_graph

# ── Colores para terminal ──────────────────────────────────────────────────────
BOLD  = "\033[1m"
GREEN = "\033[92m"
CYAN  = "\033[96m"
RESET = "\033[0m"


# ═════════════════════════════════════════════════════════════════════════════
#  FUNCIONES DE APOYO
# ═════════════════════════════════════════════════════════════════════════════

def download_pdb(pdb_id: str, outdir: str = ".") -> str:
    """Descarga un PDB del RCSB si no está en disco."""
    import requests
    pdb_id   = pdb_id.upper().strip()
    filename = os.path.join(outdir, f"{pdb_id}.pdb")
    if os.path.exists(filename):
        print(f"[PDB] {filename} ya existe, se omite la descarga.")
        return filename
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    print(f"[PDB] Descargando {pdb_id} desde {url}...")
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    with open(filename, "w") as fh:
        fh.write(resp.text)
    print(f"[PDB] Guardado: {filename} ({len(resp.text):,} caracteres)")
    return filename


def resolve_pdb(args) -> str:
    """
    Devuelve la ruta al PDB a usar, según los argumentos:
    - --pdb-file  : usa el archivo directamente
    - --pdb       : descarga desde RCSB si no existe
    - automático  : busca *.pdb en el directorio actual
    """
    if hasattr(args, "pdb_file") and args.pdb_file:
        if not os.path.exists(args.pdb_file):
            sys.exit(f"ERROR: No se encontró {args.pdb_file}")
        return args.pdb_file

    if hasattr(args, "pdb") and args.pdb:
        return download_pdb(args.pdb, args.outdir)

    # Buscar automáticamente
    pdbs = [f for f in os.listdir(args.outdir) if f.lower().endswith(".pdb")]
    # Ignorar PDBs generados por la herramienta
    pdbs = [f for f in pdbs if not any(
        tag in f for tag in ["sbp", "lbp", "rbp", "consensus"]
    )]
    if len(pdbs) == 1:
        path = os.path.join(args.outdir, pdbs[0])
        print(f"[PDB] PDB encontrado automáticamente: {path}")
        return path
    elif len(pdbs) > 1:
        sys.exit(f"ERROR: Hay más de un PDB en {args.outdir}. "
                 "Especifica --pdb o --pdb-file.")
    else:
        sys.exit(f"ERROR: No hay PDB en {args.outdir}. "
                 "Usa --pdb ID o --pdb-file ruta.pdb.")


def parse_smiles(args) -> list:
    """
    Devuelve lista de SMILES desde:
    - --smiles  : SMILES único como argumento
    - --smiles-file : archivo con un SMILES por línea (o CSV col1)
    """
    if hasattr(args, "smiles_file") and args.smiles_file:
        smiles_list = []
        with open(args.smiles_file) as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                # Soporte para CSV: primer campo = SMILES
                smiles_list.append(line.split(",")[0].split("\t")[0])
        print(f"[LBP] {len(smiles_list)} SMILES cargados desde {args.smiles_file}")
        return smiles_list

    if hasattr(args, "smiles") and args.smiles:
        return [s.strip() for s in args.smiles.split("|") if s.strip()]

    sys.exit("ERROR: Debes proporcionar --smiles 'SMILES' o --smiles-file archivo.smi")


def print_summary(name: str, features: list, clustered: list):
    from collections import Counter
    raw_c = Counter(f["type"] for f in features)
    clu_c = Counter(f["type"] for f in clustered)
    print(f"\n{BOLD}{CYAN}── {name} ──{RESET}")
    print(f"{'Tipo':20s} {'Crudos':>8}  {'Clustered':>10}")
    print("─" * 42)
    for t in ["ACCEPTOR", "DONOR", "HYDROPHOBIC", "POS_IONIZABLE", "NEG_IONIZABLE"]:
        r = raw_c.get(t, 0)
        c = clu_c.get(t, 0)
        if r or c:
            print(f"  {t:18s} {r:>8}  {c:>10}")
    print(f"  {'TOTAL':18s} {len(features):>8}  {len(clustered):>10}")


def make_outname(outdir: str, prefix: str, ext: str) -> str:
    return os.path.join(outdir, f"pharmacophore_{prefix}.{ext}")


# ═════════════════════════════════════════════════════════════════════════════
#  SUBCOMANDOS
# ═════════════════════════════════════════════════════════════════════════════

def cmd_sbp(args):
    """Generar solo el SBP."""
    os.makedirs(args.outdir, exist_ok=True)
    pdb_file = resolve_pdb(args)

    print(f"\n{BOLD}[SBP] Generando Farmacóforo Basado en Estructura...{RESET}")
    features = generate_sbp(
        pdb_file,
        ligand_resname=args.ligand,
        plip_outdir=os.path.join(args.outdir, "plip_output"),
        verbose=not args.quiet,
    )

    clustered = cluster_features(features, radii=_parse_radii(args))
    print_summary("SBP", features, clustered)

    pdb_out  = make_outname(args.outdir, "sbp", "pdb")
    html_out = make_outname(args.outdir, "sbp", "html")

    write_pharmacophore_pdb(clustered, pdb_out, title=f"SBP — {pdb_file}")
    generate_html_viewer(
        pdb_file,
        {"SBP (clusterizado)": clustered},
        html_out,
        ligand_resname=args.ligand,
        title=f"Farmacóforo SBP — {os.path.basename(pdb_file)}",
    )

    print(f"\n{GREEN}[✓] PDB  : {pdb_out}{RESET}")
    print(f"{GREEN}[✓] HTML : {html_out}{RESET}")


def cmd_lbp(args):
    """Generar solo el LBP."""
    os.makedirs(args.outdir, exist_ok=True)
    smiles_list = parse_smiles(args)

    # ¿Alinear al cristal?
    pdb_file = None
    if hasattr(args, "align_pdb") and args.align_pdb:
        pdb_file = resolve_pdb(args)
    elif hasattr(args, "pdb_file") and args.pdb_file:
        pdb_file = args.pdb_file
    elif hasattr(args, "pdb") and args.pdb:
        try:
            pdb_file = resolve_pdb(args)
        except SystemExit:
            pdb_file = None

    print(f"\n{BOLD}[LBP] Generando Farmacóforo Basado en Ligando...{RESET}")
    features = generate_lbp(
        smiles_list,
        align_to_pdb=pdb_file,
        align_ligand_resname=args.ligand if hasattr(args, "ligand") else "RIT",
        verbose=not args.quiet,
    )

    clustered = cluster_features(features, radii=_parse_radii(args))
    print_summary("LBP", features, clustered)

    pdb_out  = make_outname(args.outdir, "lbp", "pdb")
    html_out = make_outname(args.outdir, "lbp", "html")

    write_pharmacophore_pdb(clustered, pdb_out, title="LBP — RDKit")

    if pdb_file and os.path.exists(pdb_file):
        generate_html_viewer(
            pdb_file,
            {"LBP (clusterizado)": clustered},
            html_out,
            ligand_resname=args.ligand if hasattr(args, "ligand") else "RIT",
            title="Farmacóforo LBP",
        )
    else:
        _write_lbp_only_html(clustered, html_out)

    print(f"\n{GREEN}[✓] PDB  : {pdb_out}{RESET}")
    print(f"{GREEN}[✓] HTML : {html_out}{RESET}")


def cmd_rbp(args):
    """Generar solo el RBP."""
    os.makedirs(args.outdir, exist_ok=True)
    pdb_file = resolve_pdb(args)

    print(f"\n{BOLD}[RBP] Generando Farmacóforo Basado en Receptor...{RESET}")
    features = generate_rbp(
        pdb_file,
        ligand_resname=args.ligand,
        site_cutoff=args.site_cutoff,
        ph=args.ph,
        verbose=not args.quiet,
    )

    # El RBP también se clusteriza (igual que SBP y LBP)
    clustered = cluster_features(features, radii=_parse_radii(args))
    print_summary("RBP", features, clustered)

    pdb_out  = make_outname(args.outdir, "rbp", "pdb")
    html_out = make_outname(args.outdir, "rbp", "html")

    write_pharmacophore_pdb(clustered, pdb_out, title=f"RBP — {pdb_file} (pH {args.ph})")
    generate_html_viewer(
        pdb_file,
        {"RBP (clusterizado)": clustered},
        html_out,
        ligand_resname=args.ligand,
        title=f"Farmacóforo RBP — pH {args.ph}",
    )

    print(f"\n{GREEN}[✓] PDB  : {pdb_out}{RESET}")
    print(f"{GREEN}[✓] HTML : {html_out}{RESET}")


def cmd_all(args):
    """Ejecutar SBP + LBP + RBP y construir el consenso."""
    os.makedirs(args.outdir, exist_ok=True)
    pdb_file    = resolve_pdb(args)
    smiles_list = parse_smiles(args)
    radii = _parse_radii(args)

    # ── SBP ───────────────────────────────────────────────────────────────────
    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}[1/4] SBP — Farmacóforo Basado en Estructura{RESET}")
    print(f"{BOLD}{'='*60}{RESET}")
    sbp_raw = generate_sbp(
        pdb_file,
        ligand_resname=args.ligand,
        plip_outdir=os.path.join(args.outdir, "plip_output"),
        verbose=not args.quiet,
    )
    sbp_clustered = cluster_features(sbp_raw, radii=radii)
    print_summary("SBP", sbp_raw, sbp_clustered)
    sbp_pdb = make_outname(args.outdir, "sbp", "pdb")
    write_pharmacophore_pdb(sbp_clustered, sbp_pdb, title="SBP")

    # ── LBP ───────────────────────────────────────────────────────────────────
    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}[2/4] LBP — Farmacóforo Basado en Ligando{RESET}")
    print(f"{BOLD}{'='*60}{RESET}")
    lbp_raw = generate_lbp(
        smiles_list,
        align_to_pdb=pdb_file,
        align_ligand_resname=args.ligand,
        verbose=not args.quiet,
    )
    lbp_clustered = cluster_features(lbp_raw, radii=radii)
    print_summary("LBP", lbp_raw, lbp_clustered)
    lbp_pdb = make_outname(args.outdir, "lbp", "pdb")
    write_pharmacophore_pdb(lbp_clustered, lbp_pdb, title="LBP")

    # ── RBP ───────────────────────────────────────────────────────────────────
    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}[3/4] RBP — Farmacóforo Basado en Receptor{RESET}")
    print(f"{BOLD}{'='*60}{RESET}")
    rbp_raw = generate_rbp(
        pdb_file,
        ligand_resname=args.ligand,
        site_cutoff=args.site_cutoff,
        ph=args.ph,
        verbose=not args.quiet,
    )
    rbp_clustered = cluster_features(rbp_raw, radii=radii)
    print_summary("RBP", rbp_raw, rbp_clustered)
    rbp_pdb = make_outname(args.outdir, "rbp", "pdb")
    write_pharmacophore_pdb(rbp_clustered, rbp_pdb, title="RBP")

    # ── CONSENSO ──────────────────────────────────────────────────────────────
    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}[4/4] CONSENSO — SBP ∩ LBP ∩ RBP{RESET}")
    print(f"{BOLD}{'='*60}{RESET}")
    consensus = build_consensus(
        sbp_clustered, lbp_clustered, rbp_clustered,
        radius=args.consensus_radius,
        verbose=not args.quiet,
        min_weight=args.min_consensus_weight,
    )
    cons_pdb = make_outname(args.outdir, "consensus", "pdb")
    write_pharmacophore_pdb(consensus, cons_pdb, title="Consenso Final",
                            radius=args.consensus_radius, is_consensus=True)

    # ── VISUALIZACIÓN COMPLETA ────────────────────────────────────────────────
    # IMPORTANTE: solo se muestran features CLUSTERIZADOS en todos los modelos
    features_for_html = {
        "SBP"             : sbp_clustered,
        "LBP"             : lbp_clustered,
        "RBP"             : rbp_clustered,
        "Consenso"        : consensus,
    }

    # Vista individual por modelo
    for name, feats, pdb_name in [
        ("SBP", sbp_clustered, "sbp"),
        ("LBP", lbp_clustered, "lbp"),
        ("RBP", rbp_clustered, "rbp"),
    ]:
        html_out = make_outname(args.outdir, pdb_name, "html")
        generate_html_viewer(
            pdb_file, {name: feats}, html_out,
            ligand_resname=args.ligand,
            title=f"Farmacóforo {name}",
        )

    # Vista de superposición (solo clusterizados)
    overlap_html = make_outname(args.outdir, "overlap", "html")
    generate_html_viewer(
        pdb_file,
        {
            "SBP (clust.)"  : sbp_clustered,
            "LBP (clust.)"  : lbp_clustered,
            "RBP (clust.)"  : rbp_clustered,
        },
        overlap_html,
        ligand_resname=args.ligand,
        title="Superposición SBP + LBP + RBP (clusterizados)",
    )

    # Vista consenso
    cons_html = make_outname(args.outdir, "consensus", "html")
    generate_html_viewer(
        pdb_file,
        {"Consenso (SBP∩LBP∩RBP)": consensus},
        cons_html,
        ligand_resname=args.ligand,
        title="Modelo Farmacofórico Consenso Final",
    )

    # ── GRAFO FARMACOFÓRICO ───────────────────────────────────────────────────
    if not args.quiet:
        print(f"\n[Graph] Generando grafo farmacofórico 2D...")
    generate_pharmacophore_graph(
        features=consensus,
        outdir=args.outdir,
        prefix="pharmacophore_consensus",
        title=f"Grafo Farmacofórico Consenso — {os.path.basename(pdb_file)}",
        show_angles=True,
        verbose=not args.quiet,
    )

    # ── RESUMEN FINAL ─────────────────────────────────────────────────────────
    triple = [f for f in consensus if f["weight"] == 3]
    double = [f for f in consensus if f["weight"] == 2]
    single = [f for f in consensus if f["weight"] == 1]

    print(f"\n{BOLD}{'═'*60}{RESET}")
    print(f"{BOLD}{GREEN}ARCHIVOS GENERADOS{RESET}")
    print(f"{BOLD}{'═'*60}{RESET}")
    for fname in [sbp_pdb, lbp_pdb, rbp_pdb, cons_pdb,
                  make_outname(args.outdir, "sbp", "html"),
                  make_outname(args.outdir, "lbp", "html"),
                  make_outname(args.outdir, "rbp", "html"),
                  overlap_html, cons_html]:
        if os.path.exists(fname):
            size = os.path.getsize(fname)
            print(f"  {GREEN}✓{RESET} {fname}  ({size:,} bytes)")

    print(f"\n{BOLD}CONSENSO FINAL:{RESET}")
    print(f"  Triple (★★★): {len(triple)} features")
    print(f"  Doble  (★★☆): {len(double)} features")
    print(f"  Único  (★☆☆): {len(single)} features")
    print(f"  TOTAL       : {len(consensus)} features\n")


def cmd_consensus(args):
    """Cargar PDBs de modelos ya generados y construir el consenso."""
    if not (args.sbp and args.lbp and args.rbp):
        sys.exit("ERROR: Debes proporcionar --sbp, --lbp y --rbp (rutas a PDB de cada modelo).")

    for f in [args.sbp, args.lbp, args.rbp]:
        if not os.path.exists(f):
            sys.exit(f"ERROR: No se encontró {f}")

    os.makedirs(args.outdir, exist_ok=True)

    print(f"\n{BOLD}[CONSENSO] Cargando modelos...{RESET}")
    sbp_feats = load_features_from_pdb(args.sbp, "SBP")
    lbp_feats = load_features_from_pdb(args.lbp, "LBP")
    rbp_feats = load_features_from_pdb(args.rbp, "RBP")
    print(f"  SBP: {len(sbp_feats)} features | LBP: {len(lbp_feats)} | RBP: {len(rbp_feats)}")

    # Si se pide re-clusterizar antes del consenso
    if args.recluster:
        radii = _parse_radii(args)
        sbp_feats = cluster_features(sbp_feats, radii=radii)
        lbp_feats = cluster_features(lbp_feats, radii=radii)
        rbp_feats = cluster_features(rbp_feats, radii=radii)
        print(f"  Re-clusterizado → SBP: {len(sbp_feats)} | LBP: {len(lbp_feats)} | RBP: {len(rbp_feats)}")

    consensus = build_consensus(
        sbp_feats, lbp_feats, rbp_feats,
        radius=args.consensus_radius,
        verbose=not args.quiet,
        min_weight=args.min_consensus_weight,
    )

    cons_pdb  = make_outname(args.outdir, "consensus", "pdb")
    write_pharmacophore_pdb(consensus, cons_pdb, title="Consenso (cargado desde PDBs)",
                            radius=args.consensus_radius, is_consensus=True)

    # Visualización si se proporciona PDB de la proteína
    pdb_file = None
    if hasattr(args, "pdb_file") and args.pdb_file and os.path.exists(args.pdb_file):
        pdb_file = args.pdb_file
    elif hasattr(args, "pdb") and args.pdb:
        try:
            pdb_file = download_pdb(args.pdb, args.outdir)
        except Exception:
            pass

    if pdb_file:
        cons_html = make_outname(args.outdir, "consensus", "html")
        generate_html_viewer(
            pdb_file,
            {"Consenso": consensus},
            cons_html,
            ligand_resname=args.ligand if hasattr(args, "ligand") else "LIG",
            title="Consenso desde PDBs cargados",
        )
        print(f"\n{GREEN}[✓] HTML : {cons_html}{RESET}")

    # Grafo farmacofórico
    graph_paths = generate_pharmacophore_graph(
        features=consensus,
        outdir=args.outdir,
        prefix="pharmacophore_consensus",
        title="Grafo Farmacofórico Consenso",
        verbose=not args.quiet,
    )
    print(f"{GREEN}[✓] PNG  : {graph_paths['png']}{RESET}")
    print(f"{GREEN}[✓] HTML : {graph_paths['html']}{RESET}")

    print(f"\n{GREEN}[✓] PDB  : {cons_pdb}{RESET}")


# ── Helpers de argumentos ─────────────────────────────────────────────────────

def _parse_radii(args):
    """
    Devuelve el dict de radios de clustering.
    Si el usuario pasó --cluster-radius, aplica ese valor para todos los tipos.
    Si pasó --radius-donor, --radius-hydrophobic, etc., sobreescribe por tipo.
    """
    radii = dict(CLUSTER_RADII)  # copia con valores de literatura

    # Radio global (sobreescribe todos)
    if hasattr(args, "cluster_radius") and args.cluster_radius is not None:
        for k in radii:
            radii[k] = args.cluster_radius

    # Radios por tipo
    type_map = {
        "radius_donor"    : "DONOR",
        "radius_acceptor" : "ACCEPTOR",
        "radius_hydro"    : "HYDROPHOBIC",
        "radius_pos"      : "POS_IONIZABLE",
        "radius_neg"      : "NEG_IONIZABLE",
    }
    for attr, ftype in type_map.items():
        val = getattr(args, attr, None)
        if val is not None:
            radii[ftype] = val

    return radii


def _write_lbp_only_html(features, output_html):
    """
    Genera un HTML mínimo para LBP sin proteína (sin PDB).
    Usa py3Dmol solo para esferas.
    """
    from core.visualization import COLOR_MAP, DEFAULT_STYLE, CONSENSUS_STYLE
    from collections import Counter
    cnt = Counter(f["type"] for f in features)

    spheres_js_parts = []
    for f in features:
        x, y, z = [float(c) for c in f["coords"]]
        color = COLOR_MAP.get(f["type"], "#00FF00")
        s = DEFAULT_STYLE
        spheres_js_parts.append(
            f"viewer.addSphere({{center:{{x:{x:.3f},y:{y:.3f},z:{z:.3f}}},"
            f"radius:{s['radius']},color:'{color}',opacity:{s['opacity']}}});"
        )

    stats = " | ".join(f"{t}: {n}" for t, n in sorted(cnt.items()))
    html = f"""<!DOCTYPE html><html><head><meta charset="UTF-8">
<title>LBP Farmacóforo</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.0.6/3Dmol-min.js"></script>
<style>body{{background:#0d1117;color:#c9d1d9;font-family:sans-serif;text-align:center;}}</style>
</head><body>
<h2 style="color:#58a6ff">Farmacóforo LBP — Solo ligando (sin proteína)</h2>
<p style="color:#8b949e">{stats} — Total: {len(features)}</p>
<div id="v" style="width:800px;height:500px;margin:auto;border:1px solid #30363d;border-radius:8px;"></div>
<script>
var viewer = $3Dmol.createViewer("v",{{backgroundColor:"0x0d1117",width:800,height:500}});
{"".join(spheres_js_parts)}
viewer.zoomTo(); viewer.render();
</script>
</body></html>"""
    with open(output_html, "w") as fh:
        fh.write(html)


# ═════════════════════════════════════════════════════════════════════════════
#  PARSEO DE ARGUMENTOS
# ═════════════════════════════════════════════════════════════════════════════

def add_pdb_args(parser):
    g = parser.add_mutually_exclusive_group()
    g.add_argument("--pdb",      metavar="ID",
                   help="ID de PDB a descargar del RCSB (ej. 1HXW)")
    g.add_argument("--pdb-file", dest="pdb_file", metavar="FILE",
                   help="Ruta a un archivo PDB local")


def add_cluster_args(parser):
    g = parser.add_argument_group("Clustering")
    g.add_argument("--cluster-radius", dest="cluster_radius", type=float,
                   help="Radio global de clustering (Å) para todos los tipos. "
                        "Por defecto usa radios por tipo basados en literatura: "
                        "Donor/Acceptor=1.5, Hydrophobic=2.5, Ionic=2.0")
    g.add_argument("--radius-donor",    dest="radius_donor",    type=float,
                   help="Radio clustering para features DONOR (Å, default 1.5)")
    g.add_argument("--radius-acceptor", dest="radius_acceptor", type=float,
                   help="Radio clustering para features ACCEPTOR (Å, default 1.5)")
    g.add_argument("--radius-hydro",    dest="radius_hydro",    type=float,
                   help="Radio clustering para features HYDROPHOBIC (Å, default 2.5)")
    g.add_argument("--radius-pos",      dest="radius_pos",      type=float,
                   help="Radio clustering para POS_IONIZABLE (Å, default 2.0)")
    g.add_argument("--radius-neg",      dest="radius_neg",      type=float,
                   help="Radio clustering para NEG_IONIZABLE (Å, default 2.0)")


def add_common_args(parser):
    parser.add_argument("--ligand",  default="RIT", metavar="RESNAME",
                        help="Nombre del residuo del ligando en el PDB (default: RIT)")
    parser.add_argument("--outdir",  default=".",  metavar="DIR",
                        help="Directorio de salida (default: directorio actual)")
    parser.add_argument("--quiet",   action="store_true",
                        help="Suprimir mensajes de progreso")


def build_parser():
    parser = argparse.ArgumentParser(
        prog="pharmacophore",
        description=textwrap.dedent("""\
            Herramienta para modelado de farmacóforos 3D.
            Genera modelos SBP, LBP, RBP y Consenso a partir de
            estructuras PDB y/o SMILES de ligandos.
        """),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Ejemplos:
              # Todo en uno (descarga PDB, genera los 3 modelos + consenso):
              python pharmacophore.py all --pdb 1HXW --smiles "CC(C)..." --ligand RIT

              # Solo SBP con PDB local:
              python pharmacophore.py sbp --pdb-file mi_proteina.pdb --ligand LIG

              # LBP con múltiples ligandos desde archivo:
              python pharmacophore.py lbp --smiles-file ligandos.smi

              # RBP con sitio activo personalizado:
              python pharmacophore.py rbp --pdb 2HXW --ligand LIG --site-cutoff 5.0

              # Consenso desde PDBs ya generados:
              python pharmacophore.py consensus --sbp sbp.pdb --lbp lbp.pdb --rbp rbp.pdb
        """),
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # ── sbp ──────────────────────────────────────────────────────────────────
    p_sbp = sub.add_parser("sbp", help="Generar Farmacóforo Basado en Estructura")
    add_pdb_args(p_sbp)
    add_common_args(p_sbp)
    add_cluster_args(p_sbp)

    # ── lbp ──────────────────────────────────────────────────────────────────
    p_lbp = sub.add_parser("lbp", help="Generar Farmacóforo Basado en Ligando")
    add_pdb_args(p_lbp)   # opcional, para alineamiento
    smiles_group = p_lbp.add_mutually_exclusive_group(required=True)
    smiles_group.add_argument("--smiles",      metavar="SMILES",
                              help="SMILES del ligando (separar múltiples con '|')")
    smiles_group.add_argument("--smiles-file", dest="smiles_file", metavar="FILE",
                              help="Archivo con SMILES (uno por línea, o CSV/TSV col1)")
    add_common_args(p_lbp)
    add_cluster_args(p_lbp)

    # ── rbp ──────────────────────────────────────────────────────────────────
    p_rbp = sub.add_parser("rbp", help="Generar Farmacóforo Basado en Receptor")
    add_pdb_args(p_rbp)
    p_rbp.add_argument("--site-cutoff", dest="site_cutoff", type=float,
                       default=6.0, metavar="Å",
                       help="Radio del sitio activo alrededor del ligando (default: 6.0 Å)")
    p_rbp.add_argument("--ph", type=float, default=7.4,
                       help="pH para calcular estados de protonación (default: 7.4)")
    add_common_args(p_rbp)
    add_cluster_args(p_rbp)

    # ── all ───────────────────────────────────────────────────────────────────
    p_all = sub.add_parser("all",
                           help="Ejecutar SBP + LBP + RBP + Consenso automáticamente")
    add_pdb_args(p_all)
    smiles_group2 = p_all.add_mutually_exclusive_group(required=True)
    smiles_group2.add_argument("--smiles",      metavar="SMILES",
                               help="SMILES del ligando principal")
    smiles_group2.add_argument("--smiles-file", dest="smiles_file", metavar="FILE",
                               help="Archivo con SMILES (uno por línea)")
    p_all.add_argument("--site-cutoff", dest="site_cutoff", type=float, default=6.0)
    p_all.add_argument("--ph", type=float, default=7.4)
    p_all.add_argument("--consensus-radius", dest="consensus_radius", type=float,
                       default=3.5, metavar="Å",
                       help="Radio para fusionar features entre modelos (default: 3.5 Å)")
    p_all.add_argument("--min-consensus-weight", dest="min_consensus_weight", type=int,
                       default=1, choices=[1, 2, 3], metavar="{1,2,3}",
                       help="Peso mínimo para incluir un feature en el consenso (default: 1). "
                            "1=incluir todos (★☆☆ a ★★★): conserva interacciones de un único "
                            "modelo que podrían ser evidencia cristalográfica real (SBP/PLIP). "
                            "2=solo doble o triple consenso: más estricto, menos ruido. "
                            "3=solo triple consenso: máxima confianza.")
    add_common_args(p_all)
    add_cluster_args(p_all)

    # ── consensus ─────────────────────────────────────────────────────────────
    p_cons = sub.add_parser("consensus",
                            help="Construir consenso desde PDBs de modelos ya generados")
    p_cons.add_argument("--sbp", required=True, metavar="FILE",
                        help="PDB del modelo SBP (generado previamente)")
    p_cons.add_argument("--lbp", required=True, metavar="FILE",
                        help="PDB del modelo LBP")
    p_cons.add_argument("--rbp", required=True, metavar="FILE",
                        help="PDB del modelo RBP")
    p_cons.add_argument("--consensus-radius", dest="consensus_radius", type=float,
                        default=3.5, metavar="Å")
    p_cons.add_argument("--min-consensus-weight", dest="min_consensus_weight", type=int,
                        default=1, choices=[1, 2, 3], metavar="{1,2,3}",
                        help="Peso mínimo para incluir feature en consenso (default: 1 = todos). "
                             "Usar 2 para cribado estricto (solo features de ≥2 modelos).")
    p_cons.add_argument("--recluster", action="store_true",
                        help="Re-clusterizar los features cargados antes del consenso")
    add_pdb_args(p_cons)   # opcional, para visualización
    add_common_args(p_cons)
    add_cluster_args(p_cons)

    return parser


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = build_parser()
    args   = parser.parse_args()

    dispatch = {
        "sbp"      : cmd_sbp,
        "lbp"      : cmd_lbp,
        "rbp"      : cmd_rbp,
        "all"      : cmd_all,
        "consensus": cmd_consensus,
    }

    fn = dispatch.get(args.command)
    if fn is None:
        parser.print_help()
        sys.exit(1)

    try:
        fn(args)
    except KeyboardInterrupt:
        print("\n[!] Interrumpido por el usuario.")
        sys.exit(130)
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        if os.environ.get("PHARM_DEBUG"):
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
