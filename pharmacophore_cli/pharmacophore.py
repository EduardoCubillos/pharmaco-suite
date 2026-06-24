#!/usr/bin/env python3
"""
pharmacophore.py — 3D pharmacophore modeling CLI.

Subcommands:
  sbp       Structure-Based Pharmacophore (PLIP + BioPython fallback)
  lbp       Ligand-Based Pharmacophore (RDKit, single or multi-SMILES)
  rbp       Receptor-Based Pharmacophore (binding site + pH-aware protonation)
  all       Run SBP + LBP + RBP and build the consensus automatically
  consensus Load previously generated PDB models and build their consensus

Examples:
  # Download 1HXW from RCSB and run the full pipeline:
  python pharmacophore.py all --pdb 1HXW --smiles "CC(C)c1nc(c...)..." --ligand RIT

  # Local PDB, SBP only:
  python pharmacophore.py sbp --pdb-file my_protein.pdb --ligand LIG

  # Multi-ligand LBP from a SMILES file:
  python pharmacophore.py lbp --smiles-file active_compounds.smi

  # Consensus from PDB files already on disk:
  python pharmacophore.py consensus --sbp sbp.pdb --lbp lbp.pdb --rbp rbp.pdb

Author: Eduardo Cubillos
"""

import argparse
import os
import sys
import textwrap

_TOOL_DIR = os.path.dirname(os.path.abspath(__file__))
if _TOOL_DIR not in sys.path:
    sys.path.insert(0, _TOOL_DIR)

from core.clustering    import cluster_features, CLUSTER_RADII
from core.sbp           import generate_sbp
from core.lbp           import generate_lbp
from core.rbp           import generate_rbp
from core.consensus     import build_consensus, load_features_from_pdb
from core.io_pdb        import write_pharmacophore_pdb
from core.visualization import generate_html_viewer
from core.graph         import generate_pharmacophore_graph

BOLD  = "\033[1m"
GREEN = "\033[92m"
CYAN  = "\033[96m"
RESET = "\033[0m"


# ── Utility functions ─────────────────────────────────────────────────────────

def download_pdb(pdb_id: str, outdir: str = ".") -> str:
    """Download a PDB file from RCSB. Skips if already on disk."""
    import requests
    pdb_id   = pdb_id.upper().strip()
    filename = os.path.join(outdir, f"{pdb_id}.pdb")
    if os.path.exists(filename):
        print(f"[PDB] {filename} already exists, skipping download.")
        return filename
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    print(f"[PDB] Downloading {pdb_id} from {url}...")
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    with open(filename, "w") as fh:
        fh.write(resp.text)
    print(f"[PDB] Saved: {filename} ({len(resp.text):,} characters)")
    return filename


def resolve_pdb(args) -> str:
    """
    Return the PDB path to use:
    - --pdb-file  : use the path directly
    - --pdb       : download from RCSB if not on disk
    - auto-detect : look for a single *.pdb in the output directory
    """
    if hasattr(args, "pdb_file") and args.pdb_file:
        if not os.path.exists(args.pdb_file):
            sys.exit(f"ERROR: File not found: {args.pdb_file}")
        return args.pdb_file

    if hasattr(args, "pdb") and args.pdb:
        return download_pdb(args.pdb, args.outdir)

    # Try to auto-detect a single PDB in the output dir
    candidates = [
        f for f in os.listdir(args.outdir)
        if f.lower().endswith(".pdb")
        and not any(tag in f for tag in ["sbp", "lbp", "rbp", "consensus"])
    ]
    if len(candidates) == 1:
        path = os.path.join(args.outdir, candidates[0])
        print(f"[PDB] Auto-detected: {path}")
        return path
    elif len(candidates) > 1:
        sys.exit(f"ERROR: Multiple PDB files in {args.outdir}. Specify --pdb or --pdb-file.")
    else:
        sys.exit(f"ERROR: No PDB found in {args.outdir}. Use --pdb <ID> or --pdb-file <path>.")


def parse_smiles(args) -> list:
    """
    Return the list of SMILES from:
    - --smiles      : single or pipe-separated SMILES string
    - --smiles-file : file with one SMILES per line (comments with #, CSV col1)
    """
    if hasattr(args, "smiles_file") and args.smiles_file:
        smiles_list = []
        with open(args.smiles_file) as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                smiles_list.append(line.split(",")[0].split("\t")[0])
        print(f"[LBP] {len(smiles_list)} SMILES loaded from {args.smiles_file}")
        return smiles_list

    if hasattr(args, "smiles") and args.smiles:
        return [s.strip() for s in args.smiles.split("|") if s.strip()]

    sys.exit("ERROR: Provide --smiles 'SMILES' or --smiles-file <file>")


def print_feature_summary(name: str, raw: list, clustered: list):
    from collections import Counter
    raw_cnt = Counter(f["type"] for f in raw)
    clu_cnt = Counter(f["type"] for f in clustered)
    print(f"\n{BOLD}{CYAN}── {name} ──{RESET}")
    print(f"{'Type':20s} {'Raw':>8}  {'Clustered':>10}")
    print("─" * 42)
    for t in ["ACCEPTOR", "DONOR", "HYDROPHOBIC", "POS_IONIZABLE", "NEG_IONIZABLE"]:
        r = raw_cnt.get(t, 0)
        c = clu_cnt.get(t, 0)
        if r or c:
            print(f"  {t:18s} {r:>8}  {c:>10}")
    print(f"  {'TOTAL':18s} {len(raw):>8}  {len(clustered):>10}")


def _outpath(outdir: str, prefix: str, ext: str) -> str:
    return os.path.join(outdir, f"pharmacophore_{prefix}.{ext}")


# ── Subcommands ───────────────────────────────────────────────────────────────

def cmd_sbp(args):
    os.makedirs(args.outdir, exist_ok=True)
    pdb_file = resolve_pdb(args)

    print(f"\n{BOLD}[SBP] Generating Structure-Based Pharmacophore...{RESET}")
    raw = generate_sbp(
        pdb_file,
        ligand_resname=args.ligand,
        plip_outdir=os.path.join(args.outdir, "plip_output"),
        verbose=not args.quiet,
    )
    clustered = cluster_features(raw, radii=_build_radii(args))
    print_feature_summary("SBP", raw, clustered)

    pdb_out  = _outpath(args.outdir, "sbp", "pdb")
    html_out = _outpath(args.outdir, "sbp", "html")

    write_pharmacophore_pdb(clustered, pdb_out, title=f"SBP — {pdb_file}")
    generate_html_viewer(pdb_file, {"SBP (clustered)": clustered}, html_out,
                         ligand_resname=args.ligand,
                         title=f"SBP Pharmacophore — {os.path.basename(pdb_file)}")

    print(f"\n{GREEN}[✓] PDB  : {pdb_out}{RESET}")
    print(f"{GREEN}[✓] HTML : {html_out}{RESET}")


def cmd_lbp(args):
    os.makedirs(args.outdir, exist_ok=True)
    smiles_list = parse_smiles(args)

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

    print(f"\n{BOLD}[LBP] Generating Ligand-Based Pharmacophore...{RESET}")
    raw = generate_lbp(
        smiles_list,
        align_to_pdb=pdb_file,
        align_ligand_resname=args.ligand if hasattr(args, "ligand") else "LIG",
        verbose=not args.quiet,
    )
    clustered = cluster_features(raw, radii=_build_radii(args))
    print_feature_summary("LBP", raw, clustered)

    pdb_out  = _outpath(args.outdir, "lbp", "pdb")
    html_out = _outpath(args.outdir, "lbp", "html")

    write_pharmacophore_pdb(clustered, pdb_out, title="LBP — RDKit")

    if pdb_file and os.path.exists(pdb_file):
        generate_html_viewer(pdb_file, {"LBP (clustered)": clustered}, html_out,
                             ligand_resname=args.ligand if hasattr(args, "ligand") else "LIG",
                             title="LBP Pharmacophore")
    else:
        _write_ligand_only_html(clustered, html_out)

    print(f"\n{GREEN}[✓] PDB  : {pdb_out}{RESET}")
    print(f"{GREEN}[✓] HTML : {html_out}{RESET}")


def cmd_rbp(args):
    os.makedirs(args.outdir, exist_ok=True)
    pdb_file = resolve_pdb(args)

    print(f"\n{BOLD}[RBP] Generating Receptor-Based Pharmacophore...{RESET}")
    raw = generate_rbp(
        pdb_file,
        ligand_resname=args.ligand,
        site_cutoff=args.site_cutoff,
        ph=args.ph,
        verbose=not args.quiet,
    )
    clustered = cluster_features(raw, radii=_build_radii(args))
    print_feature_summary("RBP", raw, clustered)

    pdb_out  = _outpath(args.outdir, "rbp", "pdb")
    html_out = _outpath(args.outdir, "rbp", "html")

    write_pharmacophore_pdb(clustered, pdb_out, title=f"RBP — {pdb_file} (pH {args.ph})")
    generate_html_viewer(pdb_file, {"RBP (clustered)": clustered}, html_out,
                         ligand_resname=args.ligand,
                         title=f"RBP Pharmacophore — pH {args.ph}")

    print(f"\n{GREEN}[✓] PDB  : {pdb_out}{RESET}")
    print(f"{GREEN}[✓] HTML : {html_out}{RESET}")


def cmd_all(args):
    os.makedirs(args.outdir, exist_ok=True)
    pdb_file    = resolve_pdb(args)
    smiles_list = parse_smiles(args)
    radii       = _build_radii(args)

    # ── SBP ──────────────────────────────────────────────────────────────────
    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}[1/4] SBP — Structure-Based Pharmacophore{RESET}")
    print(f"{BOLD}{'='*60}{RESET}")
    sbp_raw  = generate_sbp(pdb_file, ligand_resname=args.ligand,
                             plip_outdir=os.path.join(args.outdir, "plip_output"),
                             verbose=not args.quiet)
    sbp_clst = cluster_features(sbp_raw, radii=radii)
    print_feature_summary("SBP", sbp_raw, sbp_clst)
    sbp_pdb  = _outpath(args.outdir, "sbp", "pdb")
    write_pharmacophore_pdb(sbp_clst, sbp_pdb, title="SBP")

    # ── LBP ──────────────────────────────────────────────────────────────────
    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}[2/4] LBP — Ligand-Based Pharmacophore{RESET}")
    print(f"{BOLD}{'='*60}{RESET}")
    lbp_raw  = generate_lbp(smiles_list, align_to_pdb=pdb_file,
                             align_ligand_resname=args.ligand,
                             verbose=not args.quiet)
    lbp_clst = cluster_features(lbp_raw, radii=radii)
    print_feature_summary("LBP", lbp_raw, lbp_clst)
    lbp_pdb  = _outpath(args.outdir, "lbp", "pdb")
    write_pharmacophore_pdb(lbp_clst, lbp_pdb, title="LBP")

    # ── RBP ──────────────────────────────────────────────────────────────────
    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}[3/4] RBP — Receptor-Based Pharmacophore{RESET}")
    print(f"{BOLD}{'='*60}{RESET}")
    rbp_raw  = generate_rbp(pdb_file, ligand_resname=args.ligand,
                             site_cutoff=args.site_cutoff, ph=args.ph,
                             verbose=not args.quiet)
    rbp_clst = cluster_features(rbp_raw, radii=radii)
    print_feature_summary("RBP", rbp_raw, rbp_clst)
    rbp_pdb  = _outpath(args.outdir, "rbp", "pdb")
    write_pharmacophore_pdb(rbp_clst, rbp_pdb, title="RBP")

    # ── Consensus ─────────────────────────────────────────────────────────────
    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}[4/4] Consensus — SBP ∩ LBP ∩ RBP{RESET}")
    print(f"{BOLD}{'='*60}{RESET}")
    consensus = build_consensus(
        sbp_clst, lbp_clst, rbp_clst,
        radius=args.consensus_radius,
        verbose=not args.quiet,
        min_weight=args.min_consensus_weight,
    )
    cons_pdb = _outpath(args.outdir, "consensus", "pdb")
    write_pharmacophore_pdb(consensus, cons_pdb, title="Consensus",
                            radius=args.consensus_radius, is_consensus=True)

    # ── Viewers ───────────────────────────────────────────────────────────────
    for name, feats, tag in [("SBP", sbp_clst, "sbp"),
                              ("LBP", lbp_clst, "lbp"),
                              ("RBP", rbp_clst, "rbp")]:
        generate_html_viewer(
            pdb_file, {name: feats}, _outpath(args.outdir, tag, "html"),
            ligand_resname=args.ligand, title=f"{name} Pharmacophore",
        )

    overlap_html = _outpath(args.outdir, "overlap", "html")
    generate_html_viewer(
        pdb_file,
        {"SBP": sbp_clst, "LBP": lbp_clst, "RBP": rbp_clst},
        overlap_html,
        ligand_resname=args.ligand,
        title="SBP + LBP + RBP Overlay (clustered)",
    )

    cons_html = _outpath(args.outdir, "consensus", "html")
    generate_html_viewer(
        pdb_file, {"Consensus (SBP∩LBP∩RBP)": consensus}, cons_html,
        ligand_resname=args.ligand, title="Consensus Pharmacophore",
    )

    # ── Graph ─────────────────────────────────────────────────────────────────
    if not args.quiet:
        print(f"\n[Graph] Generating 2D pharmacophore graph...")
    generate_pharmacophore_graph(
        features=consensus,
        outdir=args.outdir,
        prefix="pharmacophore_consensus",
        title=f"Consensus Pharmacophore Graph — {os.path.basename(pdb_file)}",
        show_angles=True,
        verbose=not args.quiet,
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    triple = [f for f in consensus if f["weight"] == 3]
    double = [f for f in consensus if f["weight"] == 2]
    single = [f for f in consensus if f["weight"] == 1]

    print(f"\n{BOLD}{'═'*60}{RESET}")
    print(f"{BOLD}{GREEN}OUTPUT FILES{RESET}")
    print(f"{BOLD}{'═'*60}{RESET}")
    for fname in [sbp_pdb, lbp_pdb, rbp_pdb, cons_pdb,
                  _outpath(args.outdir, "sbp", "html"),
                  _outpath(args.outdir, "lbp", "html"),
                  _outpath(args.outdir, "rbp", "html"),
                  overlap_html, cons_html]:
        if os.path.exists(fname):
            print(f"  {GREEN}✓{RESET} {fname}  ({os.path.getsize(fname):,} bytes)")

    print(f"\n{BOLD}CONSENSUS SUMMARY:{RESET}")
    print(f"  Triple (★★★): {len(triple)} features")
    print(f"  Double (★★☆): {len(double)} features")
    print(f"  Single (★☆☆): {len(single)} features")
    print(f"  TOTAL       : {len(consensus)} features\n")


def cmd_consensus(args):
    if not (args.sbp and args.lbp and args.rbp):
        sys.exit("ERROR: Provide --sbp, --lbp, and --rbp (paths to PDB files).")

    for f in [args.sbp, args.lbp, args.rbp]:
        if not os.path.exists(f):
            sys.exit(f"ERROR: File not found: {f}")

    os.makedirs(args.outdir, exist_ok=True)

    print(f"\n{BOLD}[CONSENSUS] Loading models...{RESET}")
    sbp_feats = load_features_from_pdb(args.sbp, "SBP")
    lbp_feats = load_features_from_pdb(args.lbp, "LBP")
    rbp_feats = load_features_from_pdb(args.rbp, "RBP")
    print(f"  SBP: {len(sbp_feats)} features | LBP: {len(lbp_feats)} | RBP: {len(rbp_feats)}")

    if args.recluster:
        radii     = _build_radii(args)
        sbp_feats = cluster_features(sbp_feats, radii=radii)
        lbp_feats = cluster_features(lbp_feats, radii=radii)
        rbp_feats = cluster_features(rbp_feats, radii=radii)
        print(f"  After re-clustering → SBP: {len(sbp_feats)} | LBP: {len(lbp_feats)} | RBP: {len(rbp_feats)}")

    consensus = build_consensus(
        sbp_feats, lbp_feats, rbp_feats,
        radius=args.consensus_radius,
        verbose=not args.quiet,
        min_weight=args.min_consensus_weight,
    )

    cons_pdb = _outpath(args.outdir, "consensus", "pdb")
    write_pharmacophore_pdb(consensus, cons_pdb, title="Consensus (from loaded PDBs)",
                            radius=args.consensus_radius, is_consensus=True)

    pdb_file = None
    if hasattr(args, "pdb_file") and args.pdb_file and os.path.exists(args.pdb_file):
        pdb_file = args.pdb_file
    elif hasattr(args, "pdb") and args.pdb:
        try:
            pdb_file = download_pdb(args.pdb, args.outdir)
        except Exception:
            pass

    if pdb_file:
        cons_html = _outpath(args.outdir, "consensus", "html")
        generate_html_viewer(pdb_file, {"Consensus": consensus}, cons_html,
                             ligand_resname=args.ligand if hasattr(args, "ligand") else "LIG",
                             title="Consensus from loaded PDBs")
        print(f"\n{GREEN}[✓] HTML : {cons_html}{RESET}")

    graph_paths = generate_pharmacophore_graph(
        features=consensus, outdir=args.outdir,
        prefix="pharmacophore_consensus",
        title="Consensus Pharmacophore Graph",
        verbose=not args.quiet,
    )
    print(f"{GREEN}[✓] PNG  : {graph_paths['png']}{RESET}")
    print(f"{GREEN}[✓] HTML : {graph_paths['html']}{RESET}")
    print(f"\n{GREEN}[✓] PDB  : {cons_pdb}{RESET}")


# ── Argument helpers ──────────────────────────────────────────────────────────

def _build_radii(args):
    """
    Build the clustering radii dict from CLI arguments.
    Global --cluster-radius overrides all types; per-type flags override individually.
    """
    radii = dict(CLUSTER_RADII)

    if hasattr(args, "cluster_radius") and args.cluster_radius is not None:
        radii = {k: args.cluster_radius for k in radii}

    overrides = {
        "radius_donor"   : "DONOR",
        "radius_acceptor": "ACCEPTOR",
        "radius_hydro"   : "HYDROPHOBIC",
        "radius_pos"     : "POS_IONIZABLE",
        "radius_neg"     : "NEG_IONIZABLE",
    }
    for attr, ftype in overrides.items():
        val = getattr(args, attr, None)
        if val is not None:
            radii[ftype] = val

    return radii


def _write_ligand_only_html(features, output_html):
    """Minimal HTML viewer for LBP when no protein PDB is available."""
    from core.visualization import COLOR_MAP, _DEFAULT_SPHERE_STYLE
    from collections import Counter
    cnt = Counter(f["type"] for f in features)

    spheres = []
    for f in features:
        x, y, z = [float(c) for c in f["coords"]]
        color   = COLOR_MAP.get(f["type"], "#00FF00")
        s       = _DEFAULT_SPHERE_STYLE
        spheres.append(
            f"viewer.addSphere({{center:{{x:{x:.3f},y:{y:.3f},z:{z:.3f}}},"
            f"radius:{s['radius']},color:'{color}',opacity:{s['opacity']}}});"
        )

    stats = " | ".join(f"{t}: {n}" for t, n in sorted(cnt.items()))
    html  = f"""<!DOCTYPE html><html><head><meta charset="UTF-8">
<title>LBP Pharmacophore</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.0.6/3Dmol-min.js"></script>
<style>
  body {{ background:#0d1117; color:#c9d1d9; font-family:sans-serif; text-align:center; }}
  #v {{ width:800px; height:500px; margin:auto; border:1px solid #30363d;
       border-radius:8px; position:relative; overflow:hidden; }}
</style>
</head><body>
<h2 style="color:#58a6ff">LBP Pharmacophore — ligand only (no protein structure)</h2>
<p style="color:#8b949e">{stats} — Total: {len(features)}</p>
<div id="v"></div>
<script>
var viewer = $3Dmol.createViewer("v", {{backgroundColor:"0x0d1117"}});
{"".join(spheres)}
viewer.zoomTo(); viewer.render();
</script>
</body></html>"""
    with open(output_html, "w") as fh:
        fh.write(html)


# ── Parser ────────────────────────────────────────────────────────────────────

def _add_pdb_args(p):
    g = p.add_mutually_exclusive_group()
    g.add_argument("--pdb",      metavar="ID",   help="PDB ID to download from RCSB (e.g. 1HXW)")
    g.add_argument("--pdb-file", dest="pdb_file", metavar="FILE", help="Path to a local PDB file")


def _add_cluster_args(p):
    g = p.add_argument_group("Clustering")
    g.add_argument("--cluster-radius", dest="cluster_radius", type=float,
                   help="Global clustering radius (Å) for all feature types. "
                        "Defaults to type-specific radii from Wolber & Langer (2005): "
                        "Donor/Acceptor=1.5, Hydrophobic=2.5, Ionic=2.0")
    g.add_argument("--radius-donor",    dest="radius_donor",    type=float, help="DONOR clustering radius (Å, default 1.5)")
    g.add_argument("--radius-acceptor", dest="radius_acceptor", type=float, help="ACCEPTOR clustering radius (Å, default 1.5)")
    g.add_argument("--radius-hydro",    dest="radius_hydro",    type=float, help="HYDROPHOBIC clustering radius (Å, default 2.5)")
    g.add_argument("--radius-pos",      dest="radius_pos",      type=float, help="POS_IONIZABLE clustering radius (Å, default 2.0)")
    g.add_argument("--radius-neg",      dest="radius_neg",      type=float, help="NEG_IONIZABLE clustering radius (Å, default 2.0)")


def _add_common_args(p):
    p.add_argument("--ligand", default="RIT", metavar="RESNAME",
                   help="Ligand residue name in the PDB (default: RIT)")
    p.add_argument("--outdir", default=".",   metavar="DIR",
                   help="Output directory (default: current directory)")
    p.add_argument("--quiet",  action="store_true", help="Suppress progress output")


def build_parser():
    parser = argparse.ArgumentParser(
        prog="pharmacophore",
        description=textwrap.dedent("""\
            3D pharmacophore modeling from crystal structures and SMILES.
            Generates SBP, LBP, RBP models and a weighted consensus.
        """),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              # Full pipeline (download + 3 models + consensus):
              python pharmacophore.py all --pdb 1HXW --smiles "CC(C)..." --ligand RIT

              # SBP from local PDB:
              python pharmacophore.py sbp --pdb-file my_protein.pdb --ligand LIG

              # Multi-ligand LBP from file:
              python pharmacophore.py lbp --smiles-file compounds.smi

              # Custom binding site:
              python pharmacophore.py rbp --pdb 1HXW --ligand RIT --site-cutoff 5.0

              # Consensus from existing PDB files:
              python pharmacophore.py consensus --sbp sbp.pdb --lbp lbp.pdb --rbp rbp.pdb
        """),
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # sbp
    p_sbp = sub.add_parser("sbp", help="Structure-Based Pharmacophore")
    _add_pdb_args(p_sbp)
    _add_common_args(p_sbp)
    _add_cluster_args(p_sbp)

    # lbp
    p_lbp = sub.add_parser("lbp", help="Ligand-Based Pharmacophore")
    _add_pdb_args(p_lbp)
    smg = p_lbp.add_mutually_exclusive_group(required=True)
    smg.add_argument("--smiles",      metavar="SMILES", help="SMILES string (pipe-separate multiple: 'A|B|C')")
    smg.add_argument("--smiles-file", dest="smiles_file", metavar="FILE", help="File with SMILES (one per line or CSV col1)")
    _add_common_args(p_lbp)
    _add_cluster_args(p_lbp)

    # rbp
    p_rbp = sub.add_parser("rbp", help="Receptor-Based Pharmacophore")
    _add_pdb_args(p_rbp)
    p_rbp.add_argument("--site-cutoff", dest="site_cutoff", type=float, default=6.0, metavar="Å",
                       help="Binding site radius around ligand (default: 6.0 Å)")
    p_rbp.add_argument("--ph", type=float, default=7.4,
                       help="pH for protonation state calculation (default: 7.4)")
    _add_common_args(p_rbp)
    _add_cluster_args(p_rbp)

    # all
    p_all = sub.add_parser("all", help="Run SBP + LBP + RBP + consensus")
    _add_pdb_args(p_all)
    smg2 = p_all.add_mutually_exclusive_group(required=True)
    smg2.add_argument("--smiles",      metavar="SMILES")
    smg2.add_argument("--smiles-file", dest="smiles_file", metavar="FILE")
    p_all.add_argument("--site-cutoff", dest="site_cutoff", type=float, default=6.0)
    p_all.add_argument("--ph", type=float, default=7.4)
    p_all.add_argument("--consensus-radius", dest="consensus_radius", type=float, default=3.5, metavar="Å",
                       help="Radius for merging features across models (default: 3.5 Å)")
    p_all.add_argument("--min-consensus-weight", dest="min_consensus_weight", type=int,
                       default=1, choices=[1, 2, 3], metavar="{1,2,3}",
                       help="Minimum weight to include a feature (default: 1 = keep everything). "
                            "1=all features (★☆☆ to ★★★): includes single-model crystal evidence. "
                            "2=double/triple consensus only: stricter, less noise. "
                            "3=triple consensus only: maximum confidence.")
    _add_common_args(p_all)
    _add_cluster_args(p_all)

    # consensus
    p_cons = sub.add_parser("consensus", help="Build consensus from existing PDB model files")
    p_cons.add_argument("--sbp", required=True, metavar="FILE")
    p_cons.add_argument("--lbp", required=True, metavar="FILE")
    p_cons.add_argument("--rbp", required=True, metavar="FILE")
    p_cons.add_argument("--consensus-radius", dest="consensus_radius", type=float, default=3.5, metavar="Å")
    p_cons.add_argument("--min-consensus-weight", dest="min_consensus_weight", type=int,
                        default=1, choices=[1, 2, 3], metavar="{1,2,3}",
                        help="Minimum weight to include a feature (default: 1 = all). "
                             "Use 2 for strict screening (features supported by ≥2 models).")
    p_cons.add_argument("--recluster", action="store_true",
                        help="Re-cluster loaded features before building consensus")
    _add_pdb_args(p_cons)
    _add_common_args(p_cons)
    _add_cluster_args(p_cons)

    return parser


# ── Entry point ───────────────────────────────────────────────────────────────

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

    try:
        dispatch[args.command](args)
    except KeyboardInterrupt:
        print("\n[!] Interrupted.")
        sys.exit(130)
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        if os.environ.get("PHARM_DEBUG"):
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
