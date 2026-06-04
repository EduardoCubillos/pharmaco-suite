#!/usr/bin/env python3
"""
denovo.py — Pipeline CLI de Diseño de Novo Molecular.

Subcomandos:
  fetch   Descargar ligandos activos desde ChEMBL o cargar desde archivo
  run     Pipeline completo: LBP multi-ligando → GA → ADMET → SDF + PDBQT
  filter  Aplicar filtro ADMET a un SDF existente

Ejemplos de uso:
  # Descargar ligandos de HIV protease desde ChEMBL
  python denovo.py fetch --target "HIV protease" --n 50 --out ligands.smi

  # Pipeline completo desde ChEMBL
  python denovo.py run --target "HIV protease" --n 20 --outdir resultados/

  # Pipeline completo desde archivo de ligandos
  python denovo.py run --ligands mis_ligandos.smi --n 20 --outdir resultados/

  # Pipeline con PDB para puntuación farmacofórica 3D
  python denovo.py run --target "HIV protease" --n 20 \\
         --pharmacophore farmacoforo_lbp.pdb --outdir resultados/

  # Solo filtrar ADMET de un SDF ya existente
  python denovo.py filter --sdf moleculas.sdf --outdir filtradas/

Autor: pharmacophore_tool / denovo_cli
"""

import argparse
import os
import sys
import textwrap

_TOOL_DIR = os.path.dirname(os.path.abspath(__file__))
if _TOOL_DIR not in sys.path:
    sys.path.insert(0, _TOOL_DIR)

BOLD  = "\033[1m"
GREEN = "\033[92m"
CYAN  = "\033[96m"
YELLOW= "\033[93m"
RESET = "\033[0m"


# ═════════════════════════════════════════════════════════════════════════════
#  SUBCOMANDOS
# ═════════════════════════════════════════════════════════════════════════════

def cmd_fetch(args):
    """Descargar o cargar ligandos y guardarlos en archivo .smi."""
    from core.ligand_fetcher import fetch_ligands, save_ligands_smi

    print(f"\n{BOLD}[FETCH] Obteniendo ligandos: {args.target or args.file}{RESET}")

    source = args.file if args.file else args.target
    if not source:
        sys.exit("ERROR: Proporciona --target 'nombre' o --file ruta.smi")

    ligands = fetch_ligands(
        source=source,
        n_max=args.n,
        activity_cutoff=args.activity_cutoff,
        verbose=not args.quiet,
    )

    if not ligands:
        sys.exit("ERROR: No se obtuvieron ligandos.")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    save_ligands_smi(ligands, args.out)

    print(f"\n{GREEN}[✓] {len(ligands)} ligandos guardados en: {args.out}{RESET}")
    print(f"    Ejemplos:")
    for lig in ligands[:3]:
        print(f"      {lig['chembl_id']:15s}  {lig['smiles'][:60]}")


def cmd_run(args):
    """Pipeline completo de diseño de novo."""
    from core.ligand_fetcher  import fetch_ligands
    from core.pharma_scorer   import build_reference_profile, load_pharmacophore_pdb
    from core.generator       import generate_molecules
    from core.admet_filter    import admet_summary_table
    from core.exporter        import export_candidates

    os.makedirs(args.outdir, exist_ok=True)

    # ── Paso 1: Obtener ligandos ──────────────────────────────────────────────
    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}[1/4] Obteniendo ligandos de entrenamiento{RESET}")
    print(f"{BOLD}{'='*60}{RESET}")

    source = args.file if args.file else args.target
    if not source:
        sys.exit("ERROR: Proporciona --target 'nombre del target' o --file ruta_ligandos")

    ligands = fetch_ligands(
        source=source,
        n_max=args.n_ligands,
        activity_cutoff=args.activity_cutoff,
        verbose=not args.quiet,
    )

    if not ligands:
        sys.exit("ERROR: No se obtuvieron ligandos. Verifica el nombre del target o el archivo.")

    smiles_list = [lig["smiles"] for lig in ligands]
    print(f"\n  {GREEN}✓{RESET} {len(smiles_list)} ligandos listos.")

    # Guardar ligandos usados
    from core.ligand_fetcher import save_ligands_smi
    ligands_path = os.path.join(args.outdir, "ligands_training.smi")
    save_ligands_smi(ligands, ligands_path)
    print(f"  Ligandos de entrenamiento guardados en: {ligands_path}")

    # ── Paso 2: Construir perfil farmacofórico LBP ────────────────────────────
    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}[2/4] Construyendo perfil farmacofórico LBP{RESET}")
    print(f"{BOLD}{'='*60}{RESET}")

    reference_profile = build_reference_profile(smiles_list)
    print(f"\n  Features LBP de referencia (presentes en ≥50% de ligandos):")
    for ftype, n in sorted(reference_profile.items()):
        print(f"    {ftype:20s}: {n}")

    # Cargar farmacóforo PDB si se proporcionó (para scoring 3D adicional)
    pdb_features = None
    if args.pharmacophore and os.path.exists(args.pharmacophore):
        from core.pharma_scorer import load_pharmacophore_pdb
        pdb_features = load_pharmacophore_pdb(args.pharmacophore)
        print(f"\n  Farmacóforo PDB cargado: {len(pdb_features)} features")
        print(f"  ({args.pharmacophore})")

    # ── Paso 3: Generar moléculas ─────────────────────────────────────────────
    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}[3/4] Generando {args.n} moléculas de novo (GA SELFIES){RESET}")
    print(f"{BOLD}{'='*60}{RESET}")

    ga_params = {
        "n_pop"  : args.pop_size,
        "n_gen"  : args.n_gen,
        "p_mut"  : args.p_mut,
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
        sys.exit("ERROR: No se generaron moléculas válidas. "
                 "Intenta aumentar --pop-size o --n-gen.")

    # Scoring 3D adicional si hay PDB
    if pdb_features:
        from core.pharma_scorer import score_against_pdb_pharmacophore
        from rdkit import Chem
        from rdkit.Chem import AllChem
        print(f"\n  Calculando score 3D contra farmacóforo PDB...")
        for cand in candidates:
            mol = Chem.MolFromSmiles(cand["smiles"])
            if mol:
                mol3d = Chem.AddHs(mol)
                if AllChem.EmbedMolecule(mol3d, AllChem.ETKDGv3()) == 0:
                    AllChem.MMFFOptimizeMolecule(mol3d)
                    mol3d = Chem.RemoveHs(mol3d)
                    cand["pharma_score_3d"] = round(
                        score_against_pdb_pharmacophore(mol3d, pdb_features), 3
                    )

    print(f"\n  {GREEN}✓{RESET} {len(candidates)} moléculas generadas.")

    # ── Paso 4: Exportar ──────────────────────────────────────────────────────
    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}[4/4] Exportando SDF + PDBQT{RESET}")
    print(f"{BOLD}{'='*60}{RESET}\n")

    paths = export_candidates(
        candidates=candidates,
        outdir=args.outdir,
        prefix=args.prefix,
        verbose=not args.quiet,
    )

    # ── Resumen final ─────────────────────────────────────────────────────────
    print(f"\n{BOLD}{'═'*60}{RESET}")
    print(f"{BOLD}{GREEN}RESUMEN — {len(candidates)} CANDIDATAS GENERADAS{RESET}")
    print(f"{BOLD}{'═'*60}{RESET}")
    print(admet_summary_table(candidates))

    print(f"\n{BOLD}Archivos:{RESET}")
    for label, path in [
        ("Ligandos entrenamiento", ligands_path),
        ("Candidatas SDF 3D",      paths["sdf"]),
        ("Métricas ADMET CSV",     paths["csv"]),
        ("PDBQT para Vina",        paths.get("pdbqt")),
    ]:
        if path and os.path.exists(path):
            size = os.path.getsize(path)
            print(f"  {GREEN}✓{RESET} {label:25s}: {path}  ({size:,} bytes)")
        elif path:
            print(f"  {YELLOW}?{RESET} {label:25s}: {path}  (no generado)")

    print(f"\n{BOLD}Próximos pasos:{RESET}")
    print(f"  1. Preparar receptor: prepare_receptor4.py -r proteina.pdb -o proteina.pdbqt")
    if paths.get("pdbqt"):
        print(f"  2. Correr docking:    vina --receptor proteina.pdbqt "
              f"--ligands {paths['pdbqt']} ...")
    else:
        print(f"  2. Preparar ligandos: mk_prepare_ligand.py -i {paths['sdf']} ...")
        print(f"  3. Correr docking:    vina --receptor proteina.pdbqt --ligands ligandos/ ...")
    print(f"  Ver: {os.path.join(args.outdir, 'PREPARACION_DOCKING.md')}\n")


def cmd_filter(args):
    """Aplicar filtro ADMET a un SDF existente."""
    from rdkit import Chem
    from core.admet_filter import compute_admet, admet_summary_table
    from core.exporter     import export_candidates

    if not os.path.exists(args.sdf):
        sys.exit(f"ERROR: No se encontró {args.sdf}")

    os.makedirs(args.outdir, exist_ok=True)

    print(f"\n{BOLD}[FILTER] Evaluando ADMET: {args.sdf}{RESET}\n")

    supplier = Chem.SDMolSupplier(args.sdf, removeHs=True)
    results  = []

    for i, mol in enumerate(supplier):
        if mol is None:
            continue
        smi   = Chem.MolToSmiles(mol)
        admet = compute_admet(mol)
        admet["smiles"] = smi
        admet["rank"]   = i + 1
        results.append(admet)

    passing = [r for r in results if r.get("passes")]
    failing = [r for r in results if not r.get("passes")]

    print(f"  Total evaluadas: {len(results)}")
    print(f"  Pasan filtro   : {len(passing)}")
    print(f"  Fallan filtro  : {len(failing)}")
    print()
    print(admet_summary_table(passing))

    if passing:
        paths = export_candidates(passing, args.outdir, prefix="filtered",
                                  verbose=not args.quiet)
        print(f"\n{GREEN}[✓] Filtradas guardadas en: {paths['sdf']}{RESET}")


# ═════════════════════════════════════════════════════════════════════════════
#  PARSEO DE ARGUMENTOS
# ═════════════════════════════════════════════════════════════════════════════

def build_parser():
    parser = argparse.ArgumentParser(
        prog="denovo",
        description=textwrap.dedent("""\
            Pipeline de Diseño de Novo Molecular.
            Genera moléculas candidatas a partir de ligandos activos conocidos,
            usando farmacóforo LBP multi-ligando + Algoritmo Genético SELFIES.
        """),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Ejemplos:
              # Pipeline completo desde ChEMBL (busca automáticamente)
              python denovo.py run --target "HIV protease" --n 20 --outdir resultados/

              # Pipeline desde archivo de ligandos propio
              python denovo.py run --file mis_ligandos.smi --n 20 --outdir resultados/

              # Con farmacóforo PDB para scoring 3D adicional
              python denovo.py run --target "HIV protease" --n 20 \\
                     --pharmacophore farmacoforo_lbp.pdb --outdir resultados/

              # Solo descargar ligandos
              python denovo.py fetch --target "HIV protease" --n 50 --out ligands.smi

              # Solo filtrar ADMET
              python denovo.py filter --sdf moleculas.sdf --outdir filtradas/
        """),
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # ── fetch ─────────────────────────────────────────────────────────────────
    p_fetch = sub.add_parser("fetch",
        help="Descargar ligandos activos desde ChEMBL o cargar desde archivo")
    src = p_fetch.add_mutually_exclusive_group(required=True)
    src.add_argument("--target", metavar="NOMBRE",
        help="Nombre del target en ChEMBL (ej. 'HIV protease', 'EGFR')")
    src.add_argument("--file", metavar="FILE",
        help="Archivo local con ligandos (.smi, .sdf, .csv)")
    p_fetch.add_argument("--n",  type=int, default=50, metavar="N",
        help="Número máximo de ligandos a obtener (default: 50)")
    p_fetch.add_argument("--activity-cutoff", dest="activity_cutoff",
        type=float, default=10000, metavar="nM",
        help="Potencia máxima en nM para incluir el compuesto (default: 10000 nM = 10 µM)")
    p_fetch.add_argument("--out", default="ligands.smi", metavar="FILE",
        help="Archivo de salida .smi (default: ligands.smi)")
    p_fetch.add_argument("--quiet", action="store_true")

    # ── run ───────────────────────────────────────────────────────────────────
    p_run = sub.add_parser("run",
        help="Pipeline completo: ligandos → LBP → GA → ADMET → SDF + PDBQT")

    src_run = p_run.add_mutually_exclusive_group(required=True)
    src_run.add_argument("--target", metavar="NOMBRE",
        help="Nombre del target en ChEMBL")
    src_run.add_argument("--file", metavar="FILE",
        help="Archivo local con ligandos de entrenamiento (.smi, .sdf, .csv)")

    p_run.add_argument("--n", type=int, default=20, metavar="N",
        help="Número de moléculas nuevas a generar (default: 20)")
    p_run.add_argument("--n-ligands", dest="n_ligands", type=int, default=50,
        metavar="N",
        help="Máximo de ligandos de entrenamiento a usar (default: 50)")
    p_run.add_argument("--activity-cutoff", dest="activity_cutoff",
        type=float, default=10000, metavar="nM",
        help="Potencia máxima en nM para ligandos de entrenamiento (default: 10000)")
    p_run.add_argument("--pharmacophore", metavar="PDB",
        help="PDB de farmacóforo (de pharmacophore_cli) para scoring 3D adicional")
    p_run.add_argument("--outdir", default="denovo_results", metavar="DIR",
        help="Directorio de salida (default: denovo_results)")
    p_run.add_argument("--prefix", default="candidates", metavar="PREFIX",
        help="Prefijo de archivos de salida (default: candidates)")

    # GA params
    ga_grp = p_run.add_argument_group("Parámetros del Algoritmo Genético")
    ga_grp.add_argument("--pop-size", dest="pop_size", type=int, default=60,
        help="Tamaño de la población (default: 60)")
    ga_grp.add_argument("--n-gen", dest="n_gen", type=int, default=50,
        help="Número de generaciones (default: 50)")
    ga_grp.add_argument("--p-mut", dest="p_mut", type=float, default=0.65,
        help="Probabilidad de mutación (default: 0.65)")
    ga_grp.add_argument("--diversity", type=float, default=0.85,
        help="Tanimoto máximo entre candidatas finales (default: 0.85)")

    p_run.add_argument("--quiet", action="store_true")

    # ── filter ────────────────────────────────────────────────────────────────
    p_filter = sub.add_parser("filter",
        help="Aplicar filtro ADMET a un SDF existente")
    p_filter.add_argument("--sdf", required=True, metavar="FILE",
        help="Archivo SDF de entrada")
    p_filter.add_argument("--outdir", default="filtered_results", metavar="DIR",
        help="Directorio de salida (default: filtered_results)")
    p_filter.add_argument("--quiet", action="store_true")

    return parser


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════════════════

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
        print("\n[!] Interrumpido por el usuario.")
        sys.exit(130)
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        if os.environ.get("DENOVO_DEBUG"):
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
