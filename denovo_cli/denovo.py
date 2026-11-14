#!/usr/bin/env python3
"""
denovo.py — De Novo Molecular Design CLI.

Subcommands:
  fetch   Download active compounds from ChEMBL or load from a local file
  run     Full pipeline: multi-ligand LBP → SELFIES GA → ADMET → SDF + PDBQT
  filter  Apply ADMET filters to an existing SDF

Examples:
  # Download HIV protease inhibitors from ChEMBL and generate 20 new molecules:
  python denovo.py run --target "CHEMBL247" --n 20 --outdir results/

  # Same but starting from your own ligand file:
  python denovo.py run --file active_compounds.smi --n 20 --outdir results/

  # Add a pharmacophore PDB for 3D scoring:
  python denovo.py run --target "CHEMBL247" --n 20 \\
         --pharmacophore pharmacophore_lbp.pdb --outdir results/

  # Just grab the ligands (no design):
  python denovo.py fetch --target "CHEMBL247" --n 50 --out ligands.smi

  # Filter ADMET on an existing SDF:
  python denovo.py filter --sdf candidates.sdf --outdir filtered/

Author: Eduardo Cubillos
"""

import argparse
import os
import sys
import textwrap

_TOOL_DIR = os.path.dirname(os.path.abspath(__file__))
if _TOOL_DIR not in sys.path:
    sys.path.insert(0, _TOOL_DIR)

BOLD   = "\033[1m"
GREEN  = "\033[92m"
CYAN   = "\033[96m"
YELLOW = "\033[93m"
RESET  = "\033[0m"


# ── Subcommands ───────────────────────────────────────────────────────────────

def cmd_fetch(args):
    """Download or load ligands and save to a .smi file."""
    from core.ligand_fetcher import fetch_ligands, save_ligands_smi

    source = args.file if args.file else args.target
    if not source:
        sys.exit("ERROR: Provide --target <name> or --file <path>")

    print(f"\n{BOLD}[FETCH] Retrieving ligands: {source}{RESET}")

    ligands = fetch_ligands(
        source=source,
        n_max=args.n,
        activity_cutoff=args.activity_cutoff,
        verbose=not args.quiet,
    )
    if not ligands:
        sys.exit("ERROR: No ligands retrieved.")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    save_ligands_smi(ligands, args.out)

    print(f"\n{GREEN}[✓] {len(ligands)} ligands saved to: {args.out}{RESET}")
    print(f"    Samples:")
    for lig in ligands[:3]:
        print(f"      {lig['chembl_id']:15s}  {lig['smiles'][:60]}")


def cmd_run(args):
    """Full de novo design pipeline."""
    from core.ligand_fetcher  import fetch_ligands, save_ligands_smi
    from core.pharma_scorer   import build_reference_profile, load_pharmacophore_pdb
    from core.generator       import generate_molecules
    from core.admet_filter    import admet_summary_table
    from core.exporter        import export_candidates

    os.makedirs(args.outdir, exist_ok=True)

    # ── Step 1: Get training ligands ──────────────────────────────────────────
    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}[1/4] Retrieving training ligands{RESET}")
    print(f"{BOLD}{'='*60}{RESET}")

    source = args.file if args.file else args.target
    if not source:
        sys.exit("ERROR: Provide --target <target name> or --file <ligand file>")

    ligands = fetch_ligands(
        source=source,
        n_max=args.n_ligands,
        activity_cutoff=args.activity_cutoff,
        verbose=not args.quiet,
    )
    if not ligands:
        sys.exit("ERROR: No ligands retrieved. Check the target name or file.")

    smiles_list   = [lig["smiles"] for lig in ligands]
    ligands_path  = os.path.join(args.outdir, "ligands_training.smi")
    save_ligands_smi(ligands, ligands_path)
    print(f"\n  {GREEN}✓{RESET} {len(smiles_list)} ligands ready.")
    print(f"  Training set saved: {ligands_path}")

    # ── Step 2: Build LBP reference profile ──────────────────────────────────
    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}[2/4] Building LBP reference profile{RESET}")
    print(f"{BOLD}{'='*60}{RESET}")

    reference_profile = build_reference_profile(smiles_list)
    print(f"\n  LBP features (present in ≥50% of training compounds):")
    for ftype, n in sorted(reference_profile.items()):
        print(f"    {ftype:20s}: {n}")

    pdb_features = None
    if args.pharmacophore and os.path.exists(args.pharmacophore):
        pdb_features = load_pharmacophore_pdb(args.pharmacophore)
        print(f"\n  Pharmacophore PDB loaded: {len(pdb_features)} features")
        print(f"  ({args.pharmacophore})")

    # ── Step 3: Generate molecules with GA ───────────────────────────────────
    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}[3/4] Generating {args.n} molecules (SELFIES GA){RESET}")
    print(f"{BOLD}{'='*60}{RESET}")

    ga_params = {
        "n_pop"              : args.pop_size,
        "n_gen"              : args.n_gen,
        "p_mut"              : args.p_mut,
        "diversity_threshold": args.diversity,
    }
    candidates = generate_molecules(
        seed_smiles=smiles_list,
        reference_profile=reference_profile,
        n_target=args.n,
        ga_params=ga_params,
        verbose=not args.quiet,
    )
    if not candidates:
        sys.exit("ERROR: No valid molecules generated. Try increasing --pop-size or --n-gen.")

    # Optional 3D pharmacophore fit
    if pdb_features:
        from core.pharma_scorer import score_against_pdb_pharmacophore
        from rdkit import Chem
        print(f"\n  Fitting candidates to the pharmacophore model "
              f"(tolerance {args.fit_tolerance} Å)...")
        n_fit = 0
        for cand in candidates:
            mol = Chem.MolFromSmiles(cand["smiles"])
            if mol is None:
                continue
            score, info = score_against_pdb_pharmacophore(
                mol, pdb_features,
                tolerance=args.fit_tolerance,
                return_details=True,
            )
            cand["pharma_score_3d"] = round(score, 3)
            cand["fit_points"]      = f"{info['n_points_matched']}/{info['n_points_total']}"
            cand["fit_rmsd"]        = info["fit_rmsd"]
            if score > 0:
                n_fit += 1
        print(f"  {n_fit}/{len(candidates)} candidates fit at least "
              f"3 pharmacophore points.")

    print(f"\n  {GREEN}✓{RESET} {len(candidates)} molecules generated.")

    # ── Step 4: Export ────────────────────────────────────────────────────────
    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}[4/4] Exporting SDF + PDBQT{RESET}")
    print(f"{BOLD}{'='*60}{RESET}\n")

    paths = export_candidates(
        candidates=candidates,
        outdir=args.outdir,
        prefix=args.prefix,
        verbose=not args.quiet,
    )

    # Summary
    print(f"\n{BOLD}{'═'*60}{RESET}")
    print(f"{BOLD}{GREEN}RESULTS — {len(candidates)} CANDIDATES{RESET}")
    print(f"{BOLD}{'═'*60}{RESET}")
    print(admet_summary_table(candidates))

    print(f"\n{BOLD}Files:{RESET}")
    for label, path in [
        ("Training ligands",  ligands_path),
        ("Candidates SDF 3D", paths["sdf"]),
        ("ADMET metrics CSV", paths["csv"]),
        ("PDBQT for Vina",    paths.get("pdbqt")),
    ]:
        if path and os.path.exists(path):
            print(f"  {GREEN}✓{RESET} {label:24s}: {path}  ({os.path.getsize(path):,} bytes)")
        elif path:
            print(f"  {YELLOW}?{RESET} {label:24s}: {path}  (not generated)")

    print(f"\n{BOLD}Next steps:{RESET}")
    print(f"  1. Prepare receptor: prepare_receptor4.py -r protein.pdb -o protein.pdbqt")
    if paths.get("pdbqt"):
        print(f"  2. Run docking:      vina --receptor protein.pdbqt --ligands {paths['pdbqt']} ...")
    else:
        print(f"  2. Prepare ligands:  mk_prepare_ligand.py -i {paths['sdf']} ...")
        print(f"  3. Run docking:      vina --receptor protein.pdbqt --ligands ligands/ ...")
    print(f"  See: {os.path.join(args.outdir, 'DOCKING_PREPARATION.md')}\n")


def cmd_filter(args):
    """Apply ADMET filters to an existing SDF file."""
    from rdkit import Chem
    from core.admet_filter import compute_admet, admet_summary_table
    from core.exporter     import export_candidates

    if not os.path.exists(args.sdf):
        sys.exit(f"ERROR: File not found: {args.sdf}")

    os.makedirs(args.outdir, exist_ok=True)
    print(f"\n{BOLD}[FILTER] Evaluating ADMET: {args.sdf}{RESET}\n")

    supplier = Chem.SDMolSupplier(args.sdf, removeHs=True)
    results  = []
    for i, mol in enumerate(supplier):
        if mol is None:
            continue
        admet = compute_admet(mol)
        admet["smiles"] = Chem.MolToSmiles(mol)
        admet["rank"]   = i + 1
        results.append(admet)

    passing = [r for r in results if r.get("passes")]
    failing = [r for r in results if not r.get("passes")]

    print(f"  Total evaluated: {len(results)}")
    print(f"  Pass filter    : {len(passing)}")
    print(f"  Fail filter    : {len(failing)}")
    print()
    print(admet_summary_table(passing))

    if passing:
        paths = export_candidates(passing, args.outdir, prefix="filtered",
                                  verbose=not args.quiet)
        print(f"\n{GREEN}[✓] Filtered results: {paths['sdf']}{RESET}")


# ── Parser ────────────────────────────────────────────────────────────────────

def build_parser():
    parser = argparse.ArgumentParser(
        prog="denovo",
        description=textwrap.dedent("""\
            De novo molecular design guided by pharmacophore profiles.
            Uses a multi-ligand LBP profile + SELFIES genetic algorithm.
        """),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              # Full pipeline from ChEMBL:
              python denovo.py run --target "CHEMBL247" --n 20 --outdir results/

              # From your own ligand file:
              python denovo.py run --file my_ligands.smi --n 20 --outdir results/

              # With 3D pharmacophore scoring:
              python denovo.py run --target "CHEMBL247" --n 20 \\
                     --pharmacophore pharmacophore_lbp.pdb --outdir results/

              # Download ligands only:
              python denovo.py fetch --target "CHEMBL247" --n 50 --out ligands.smi

              # Filter ADMET on an SDF:
              python denovo.py filter --sdf candidates.sdf --outdir filtered/
        """),
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # fetch
    p_fetch = sub.add_parser("fetch", help="Download ligands from ChEMBL or load from file")
    src = p_fetch.add_mutually_exclusive_group(required=True)
    src.add_argument("--target", metavar="NAME",
                     help="Target name in ChEMBL (e.g. 'HIV protease', 'EGFR')")
    src.add_argument("--file", metavar="FILE",
                     help="Local ligand file (.smi, .sdf, .csv)")
    p_fetch.add_argument("--n", type=int, default=50, metavar="N",
                         help="Max ligands to retrieve (default: 50)")
    p_fetch.add_argument("--activity-cutoff", dest="activity_cutoff",
                         type=float, default=10000, metavar="nM",
                         help="Max potency in nM (default: 10000 = 10 µM)")
    p_fetch.add_argument("--out", default="ligands.smi", metavar="FILE",
                         help="Output file (default: ligands.smi)")
    p_fetch.add_argument("--quiet", action="store_true")

    # run
    p_run = sub.add_parser("run", help="Full pipeline: ligands → LBP → GA → ADMET → SDF")
    src_run = p_run.add_mutually_exclusive_group(required=True)
    src_run.add_argument("--target", metavar="NAME", help="ChEMBL target name or ID")
    src_run.add_argument("--file",   metavar="FILE", help="Local ligand file")
    p_run.add_argument("--n",         type=int, default=20,    metavar="N",   help="Molecules to generate (default: 20)")
    p_run.add_argument("--n-ligands", dest="n_ligands", type=int, default=50, help="Max training ligands (default: 50)")
    p_run.add_argument("--activity-cutoff", dest="activity_cutoff", type=float, default=10000, metavar="nM")
    p_run.add_argument("--pharmacophore", metavar="PDB",
                       help="Pharmacophore PDB (from pharmacophore_cli) for additional 3D scoring")
    p_run.add_argument("--fit-tolerance", dest="fit_tolerance", type=float,
                       default=1.5, metavar="A",
                       help="Slack (Å) on every pharmacophore point-point "
                            "distance when fitting candidates (default: 1.5)")
    p_run.add_argument("--outdir", default="denovo_results", metavar="DIR")
    p_run.add_argument("--prefix", default="candidates", metavar="PREFIX")
    ga = p_run.add_argument_group("Genetic Algorithm")
    ga.add_argument("--pop-size",  dest="pop_size", type=int,   default=60,   help="Population size (default: 60)")
    ga.add_argument("--n-gen",     dest="n_gen",    type=int,   default=50,   help="Generations (default: 50)")
    ga.add_argument("--p-mut",     dest="p_mut",    type=float, default=0.65, help="Mutation probability (default: 0.65)")
    ga.add_argument("--diversity", type=float, default=0.85, help="Max Tanimoto between candidates (default: 0.85)")
    p_run.add_argument("--quiet", action="store_true")

    # filter
    p_filter = sub.add_parser("filter", help="Apply ADMET filters to an existing SDF")
    p_filter.add_argument("--sdf",    required=True, metavar="FILE")
    p_filter.add_argument("--outdir", default="filtered_results", metavar="DIR")
    p_filter.add_argument("--quiet",  action="store_true")

    return parser


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = build_parser()
    args   = parser.parse_args()

    dispatch = {
        "fetch" : cmd_fetch,
        "run"   : cmd_run,
        "filter": cmd_filter,
    }

    try:
        dispatch[args.command](args)
    except KeyboardInterrupt:
        print("\n[!] Interrupted.")
        sys.exit(130)
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        if os.environ.get("DENOVO_DEBUG"):
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
