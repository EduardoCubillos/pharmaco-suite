"""
sbp.py — Farmacóforo Basado en Estructura (Structure-Based Pharmacophore).

Estrategia:
1. Intentar PLIP (protein-ligand interaction profiler) vía CLI o API.
2. Si PLIP falla, usar fallback basado en distancias con BioPython.

Referencia: Salentin et al. (2015) Nucleic Acids Res. 43, W443–W447 (PLIP).
"""

import os
import sys
import subprocess
import shutil
import xml.etree.ElementTree as ET
import numpy as np
from typing import List, Dict, Any, Optional

# ── Constantes de fallback BioPython ──────────────────────────────────────────
HBOND_DIST_CUTOFF   = 3.5   # Å — distancia D···A para H-bond
HYDROPH_DIST_CUTOFF = 4.0   # Å — contacto hidrofóbico C···C
HBOND_ATOMS  = {"N", "O"}
HYDROPH_ELEMS = {"C", "S"}


def generate_sbp(
    pdb_file: str,
    ligand_resname: str = "RIT",
    plip_outdir: str = "plip_output",
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """
    Genera features SBP para el ligando indicado en el PDB.

    Returns
    -------
    Lista de dicts: {'type', 'coords', 'label', 'source'}
    """
    _log = print if verbose else lambda *a, **k: None

    os.makedirs(plip_outdir, exist_ok=True)
    plip_cmd = _find_plip()

    features = []
    if plip_cmd:
        _log(f"[SBP] Ejecutando PLIP: {plip_cmd}")
        features = _run_plip(plip_cmd, pdb_file, plip_outdir, verbose)

    if not features:
        _log("[SBP] Usando fallback BioPython (distancias atómicas)...")
        features = _fallback_biopython(pdb_file, ligand_resname, verbose)

    for f in features:
        f["source"] = "SBP"

    _log(f"[SBP] {len(features)} features generados.")
    return features


# ── Localización de PLIP ──────────────────────────────────────────────────────

def _find_plip() -> Optional[str]:
    """Devuelve la ruta al ejecutable o script de PLIP, o None."""
    # 1. Ejecutable en PATH
    cmd = shutil.which("plip")
    if cmd:
        return cmd
    # 2. Script de GitHub clonado
    for path in ["/content/plip/plip/plipcmd.py",
                 os.path.expanduser("~/plip/plip/plipcmd.py")]:
        if os.path.exists(path):
            return path
    return None


# ── Ejecución de PLIP ─────────────────────────────────────────────────────────

def _run_plip(
    plip_cmd: str,
    pdb_file: str,
    outdir: str,
    verbose: bool,
) -> List[Dict[str, Any]]:
    """Llama a PLIP y parsea el XML resultante."""
    if plip_cmd.endswith(".py"):
        cmd = [sys.executable, plip_cmd, "-f", pdb_file, "-x", "-o", outdir]
    else:
        cmd = [plip_cmd, "-f", pdb_file, "-x", "-o", outdir]

    env = os.environ.copy()
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)

    if result.returncode != 0:
        if verbose:
            print(f"[SBP] PLIP error: {result.stderr[:400]}")
        return []

    return _parse_plip_xml(outdir, verbose)


def _parse_plip_xml(xml_dir: str, verbose: bool) -> List[Dict[str, Any]]:
    """Extrae features del XML de PLIP."""
    features = []
    xml_files = [f for f in os.listdir(xml_dir) if f.endswith(".xml")]
    if not xml_files:
        return features

    for xf in xml_files:
        if verbose:
            print(f"[SBP] Parseando {xf}...")
        tree = ET.parse(os.path.join(xml_dir, xf))
        root = tree.getroot()

        # Interacciones hidrofóbicas
        for el in root.iter("hydrophobic_interaction"):
            try:
                x = float(el.findtext("ligc_x") or 0)
                y = float(el.findtext("ligc_y") or 0)
                z = float(el.findtext("ligc_z") or 0)
                if (x, y, z) != (0.0, 0.0, 0.0):
                    features.append({
                        "type"  : "HYDROPHOBIC",
                        "coords": (x, y, z),
                        "label" : f"Hidrofóbico (PLIP) lig_atom={el.findtext('lig_atom_idx','?')}",
                    })
            except (TypeError, ValueError):
                continue

        # Puentes de hidrógeno
        for el in root.iter("hydrogen_bond"):
            try:
                lig_is_donor = (el.findtext("protisdon") or "").lower() == "false"
                x = float(el.findtext("lig_x") or 0)
                y = float(el.findtext("lig_y") or 0)
                z = float(el.findtext("lig_z") or 0)
                dist  = el.findtext("dist_h-a") or el.findtext("dist_d-a") or "?"
                angle = el.findtext("angle") or "?"
                ftype = "DONOR" if lig_is_donor else "ACCEPTOR"
                if (x, y, z) != (0.0, 0.0, 0.0):
                    features.append({
                        "type"  : ftype,
                        "coords": (x, y, z),
                        "label" : f"{ftype} (PLIP) d={dist}Å θ={angle}°",
                    })
            except (TypeError, ValueError):
                continue

        # Interacciones π (aromáticas → HYDROPHOBIC)
        for el in root.iter("pi_stacking"):
            try:
                x = float(el.findtext("lig_x") or 0)
                y = float(el.findtext("lig_y") or 0)
                z = float(el.findtext("lig_z") or 0)
                if (x, y, z) != (0.0, 0.0, 0.0):
                    features.append({
                        "type"  : "HYDROPHOBIC",
                        "coords": (x, y, z),
                        "label" : "Aromático/π-stacking (PLIP)",
                    })
            except (TypeError, ValueError):
                continue

        # Interacciones catión-π → POS_IONIZABLE
        for el in root.iter("pi_cation_interaction"):
            try:
                x = float(el.findtext("lig_x") or 0)
                y = float(el.findtext("lig_y") or 0)
                z = float(el.findtext("lig_z") or 0)
                if (x, y, z) != (0.0, 0.0, 0.0):
                    features.append({
                        "type"  : "POS_IONIZABLE",
                        "coords": (x, y, z),
                        "label" : "Catión-π (PLIP)",
                    })
            except (TypeError, ValueError):
                continue

        # Puentes salinos → iónico
        for el in root.iter("salt_bridge"):
            try:
                x = float(el.findtext("lig_x") or 0)
                y = float(el.findtext("lig_y") or 0)
                z = float(el.findtext("lig_z") or 0)
                charge = (el.findtext("lig_charge") or "pos").lower()
                ftype = "POS_IONIZABLE" if "pos" in charge else "NEG_IONIZABLE"
                if (x, y, z) != (0.0, 0.0, 0.0):
                    features.append({
                        "type"  : ftype,
                        "coords": (x, y, z),
                        "label" : f"Puente salino {ftype} (PLIP)",
                    })
            except (TypeError, ValueError):
                continue

    return features


# ── Fallback BioPython ────────────────────────────────────────────────────────

def _fallback_biopython(
    pdb_file: str,
    ligand_resname: str,
    verbose: bool,
) -> List[Dict[str, Any]]:
    """
    Extracción de features por distancias atómicas con BioPython.
    Menos preciso que PLIP (sin ángulos) pero robusto y sin deps adicionales.
    """
    from Bio.PDB import PDBParser
    import warnings
    warnings.filterwarnings("ignore")

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("mol", pdb_file)
    model = structure[0]

    lig_atoms  = []
    prot_atoms = []

    for chain in model:
        for res in chain:
            rname = res.get_resname().strip()
            if rname == ligand_resname.strip().upper():
                lig_atoms.extend(res.get_atoms())
            elif res.id[0] == " ":
                prot_atoms.extend(res.get_atoms())

    if not lig_atoms:
        print(f"[SBP] ADVERTENCIA: no se encontró el ligando '{ligand_resname}' en {pdb_file}.")
        print(f"[SBP] Residuos hetero disponibles:")
        for chain in model:
            for res in chain:
                if res.id[0] not in (" ", "W"):
                    print(f"       Cadena {chain.id} | {res.get_resname().strip()}")
        return []

    if verbose:
        print(f"[SBP] Ligando '{ligand_resname}': {len(lig_atoms)} átomos | Proteína: {len(prot_atoms)} átomos")

    features = []
    added_coords: set = set()

    for la in lig_atoms:
        la_elem  = (la.element or la.get_name()[0]).strip()
        la_coord = tuple(np.round(la.get_vector().get_array(), 3))

        for pa in prot_atoms:
            pa_elem = (pa.element or pa.get_name()[0]).strip()
            dist    = la - pa  # BioPython __sub__ devuelve distancia

            # H-bond
            if (dist <= HBOND_DIST_CUTOFF
                    and la_elem in HBOND_ATOMS
                    and pa_elem in HBOND_ATOMS
                    and la_coord not in added_coords):
                ftype = "DONOR" if la_elem == "N" else "ACCEPTOR"
                features.append({
                    "type"  : ftype,
                    "coords": la_coord,
                    "label" : f"{ftype} (fallback, d={dist:.2f}Å)",
                })
                added_coords.add(la_coord)

            # Hidrofóbico
            if (dist <= HYDROPH_DIST_CUTOFF
                    and la_elem in HYDROPH_ELEMS
                    and pa_elem in HYDROPH_ELEMS
                    and la_coord not in added_coords):
                features.append({
                    "type"  : "HYDROPHOBIC",
                    "coords": la_coord,
                    "label" : f"Hidrofóbico (fallback, d={dist:.2f}Å)",
                })
                added_coords.add(la_coord)

    return features
