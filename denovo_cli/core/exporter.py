"""
exporter.py — Exportar candidatas en SDF 3D y PDBQT para AutoDock Vina.

Pipeline:
  1. Generar conformación 3D con RDKit (ETKDGv3 + MMFF94)
  2. Escribir SDF con metadatos (rank, score, ADMET)
  3. Intentar preparar PDBQT con Meeko (si está instalado)
     Si Meeko no está disponible, escribir instrucciones de preparación manual.

Referencia:
  - Eberhardt et al. (2021) J. Chem. Inf. Model. 61, 3891–3898 (AutoDock Vina)
  - Forli et al. (2016) Nature Protocols 11, 905–919
"""

import os
import sys
import subprocess
import shutil
from typing import List, Dict, Any


def export_candidates(
    candidates: List[Dict[str, Any]],
    outdir: str,
    prefix: str = "candidates",
    verbose: bool = True,
) -> Dict[str, str]:
    """
    Exporta las candidatas en SDF y PDBQT.

    Returns
    -------
    Dict con rutas: {'sdf': path, 'pdbqt': path_or_None, 'csv': path}
    """
    _log = print if verbose else lambda *a, **k: None
    os.makedirs(outdir, exist_ok=True)

    sdf_path  = os.path.join(outdir, f"{prefix}.sdf")
    csv_path  = os.path.join(outdir, f"{prefix}_admet.csv")
    pdbqt_path = None

    # ── SDF con conformaciones 3D ─────────────────────────────────────────────
    _log(f"[Export] Generando conformaciones 3D y escribiendo SDF...")
    n_exported = _write_sdf(candidates, sdf_path, verbose)
    _log(f"[Export] {n_exported} moléculas escritas en {sdf_path}")

    # ── CSV con métricas ──────────────────────────────────────────────────────
    _write_csv(candidates, csv_path)
    _log(f"[Export] Métricas ADMET escritas en {csv_path}")

    # ── PDBQT con Meeko (usa SDF con H explícitos) ───────────────────────────
    sdf_hs_path = sdf_path[:-4] + "_hs.sdf"
    meeko_ok = _check_meeko()
    if meeko_ok:
        pdbqt_path = os.path.join(outdir, f"{prefix}.pdbqt")
        success = _prepare_pdbqt_meeko(sdf_hs_path, pdbqt_path, verbose)
        if not success:
            pdbqt_path = None
            _log("[Export] PDBQT fallido, continuando sin él.")
    else:
        _log("[Export] Meeko no instalado. Instrucciones de preparación manual:")
        _log("  pip install meeko")
        _log(f"  mk_prepare_ligand.py -i {sdf_path} -o {os.path.join(outdir, prefix)}.pdbqt")
        _write_prep_instructions(outdir, sdf_path, prefix)

    return {"sdf": sdf_path, "pdbqt": pdbqt_path, "csv": csv_path}


# ── SDF ───────────────────────────────────────────────────────────────────────

def _write_sdf(candidates: List[Dict], path: str, verbose: bool) -> int:
    """
    Escribe dos SDF:
      - path              : sin H explícitos (visualización en PyMOL/VMD)
      - path_stem_hs.sdf  : con H explícitos (para Meeko/AutoDock Vina)
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem, SDWriter
    _log = print if verbose else lambda *a, **k: None

    stem        = path[:-4] if path.endswith(".sdf") else path
    path_hs     = stem + "_hs.sdf"   # con H — para Meeko
    writer      = SDWriter(path)      # sin H — para visualización
    writer_hs   = SDWriter(path_hs)
    exported    = 0

    for cand in candidates:
        mol = Chem.MolFromSmiles(cand["smiles"])
        if mol is None:
            continue

        # Conformación 3D con H explícitos
        mol3d = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = 42 + cand.get("rank", 0)
        status = AllChem.EmbedMolecule(mol3d, params)
        if status != 0:
            status = AllChem.EmbedMolecule(mol3d, AllChem.ETKDG())
        if status != 0:
            _log(f"  [!] No se pudo generar 3D para candidata #{cand.get('rank','?')}")
            continue
        AllChem.MMFFOptimizeMolecule(mol3d, maxIters=500)

        def _set_props(m):
            m.SetProp("_Name",          f"Candidata_{cand.get('rank', exported+1):03d}")
            m.SetProp("Rank",           str(cand.get("rank", "")))
            m.SetProp("Score",          f"{cand.get('score', 0):.4f}")
            m.SetProp("Pharma_Score",   f"{cand.get('pharma_score', 0):.4f}")
            m.SetProp("QED",            f"{cand.get('qed', 0):.4f}")
            m.SetProp("MW",             f"{cand.get('mw', 0):.2f}")
            m.SetProp("LogP",           f"{cand.get('logp', 0):.2f}")
            m.SetProp("PSA",            f"{cand.get('psa', 0):.1f}")
            m.SetProp("SA_Score",       f"{cand.get('sa_score') or 0:.2f}")
            m.SetProp("HBD",            str(cand.get("hbd", "")))
            m.SetProp("HBA",            str(cand.get("hba", "")))
            m.SetProp("Lip_Violations", str(cand.get("lip_violations", "")))
            m.SetProp("SMILES",         cand["smiles"])

        # SDF con H (para Meeko)
        _set_props(mol3d)
        writer_hs.write(mol3d)

        # SDF sin H (para visualización)
        mol3d_noH = Chem.RemoveHs(mol3d)
        _set_props(mol3d_noH)
        writer.write(mol3d_noH)

        exported += 1

    writer.close()
    writer_hs.close()
    return exported


# ── CSV ───────────────────────────────────────────────────────────────────────

def _write_csv(candidates: List[Dict], path: str):
    import csv
    fields = ["rank", "smiles", "score", "pharma_score", "qed", "sa_score",
              "mw", "logp", "psa", "hbd", "hba", "rot_bonds", "lip_violations"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(candidates)


# ── PDBQT con Meeko ───────────────────────────────────────────────────────────

def _check_meeko() -> bool:
    """Verifica si Meeko está instalado."""
    try:
        import meeko
        return True
    except ImportError:
        pass
    return bool(shutil.which("mk_prepare_ligand.py") or
                shutil.which("meeko_preparation.py"))


def _prepare_pdbqt_meeko(sdf_path: str, pdbqt_path: str, verbose: bool) -> bool:
    """Prepara PDBQT usando Meeko CLI."""
    _log = print if verbose else lambda *a, **k: None

    # Intentar el comando mk_prepare_ligand.py (Meeko >= 0.5)
    cmd = None
    if shutil.which("mk_prepare_ligand.py"):
        cmd = ["mk_prepare_ligand.py", "-i", sdf_path, "--multimol_outdir",
               os.path.dirname(pdbqt_path)]
    elif shutil.which("meeko_preparation.py"):
        cmd = ["meeko_preparation.py", "-i", sdf_path, "-o", pdbqt_path]
    else:
        # Intentar via Python API
        return _prepare_pdbqt_api(sdf_path, pdbqt_path, verbose)

    _log(f"[Export] Ejecutando: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        _log(f"[Export] PDBQT generado: {pdbqt_path}")
        return True
    else:
        _log(f"[Export] Meeko CLI error: {result.stderr[:300]}")
        return False


def _prepare_pdbqt_api(sdf_path: str, pdbqt_path: str, verbose: bool) -> bool:
    """Prepara PDBQT usando la API Python de Meeko."""
    _log = print if verbose else lambda *a, **k: None
    try:
        from meeko import MoleculePreparation
        from rdkit import Chem

        supplier = Chem.SDMolSupplier(sdf_path, removeHs=False)
        preparator = MoleculePreparation()
        pdbqt_lines = []

        for mol in supplier:
            if mol is None:
                continue
            try:
                mol_setup = preparator.prepare(mol)
                for setup in mol_setup:
                    pdbqt_lines.append(setup.write_pdbqt_string())
            except Exception as e:
                _log(f"  [!] Meeko API error: {e}")

        if pdbqt_lines:
            with open(pdbqt_path, "w") as f:
                f.write("\n".join(pdbqt_lines))
            _log(f"[Export] PDBQT generado via API: {pdbqt_path}")
            return True

    except Exception as e:
        _log(f"[Export] Meeko API no disponible: {e}")
    return False


def _write_prep_instructions(outdir: str, sdf_path: str, prefix: str):
    """Escribe un archivo README con instrucciones de preparación manual."""
    txt = f"""# Preparación de ligandos para AutoDock Vina
## Requisito: Meeko

    pip install meeko

## Preparar PDBQT desde el SDF generado

    mk_prepare_ligand.py -i {sdf_path} --multimol_outdir {outdir}/pdbqt/

## Preparar el receptor (proteína)

    # 1. Remover agua y ligando del PDB original
    # 2. Convertir a PDBQT con prepare_receptor
    prepare_receptor4.py -r proteina.pdb -o proteina.pdbqt

## Correr AutoDock Vina

    vina --receptor proteina.pdbqt \\
         --ligands {outdir}/pdbqt/ \\
         --center_x X --center_y Y --center_z Z \\
         --size_x 25 --size_y 25 --size_z 25 \\
         --exhaustiveness 16 \\
         --out resultados_docking.pdbqt

## Notas
- Ajustar las coordenadas del centro (X, Y, Z) al sitio activo de tu proteína.
- Aumentar --exhaustiveness para mayor precisión (default 8, recomendado 16-32).
"""
    readme_path = os.path.join(outdir, "PREPARACION_DOCKING.md")
    with open(readme_path, "w") as f:
        f.write(txt)
