"""
rbp.py — Farmacóforo Basado en Receptor (Receptor-Based Pharmacophore).

Genera features a partir de los residuos del sitio activo del receptor,
teniendo en cuenta el estado de protonación a pH fisiológico (7.4).

Los features del RBP representan lo que OFRECE el receptor:
  - DONOR en el receptor → el receptor dona H → el ligando debe aceptar (ACCEPTOR)
  - ACCEPTOR en el receptor → el ligando debe donar (DONOR)

Referencia:
  - Sliwoski et al. (2014) Pharmacol Rev 66, 334–395
  - Koes & Camacho (2012) J. Chem. Inf. Model. 52, 2098–2106
"""

import numpy as np
from typing import List, Dict, Any, Tuple

# ── pKa estándar a 25°C (Lehninger, 5ª ed.) ──────────────────────────────────
STANDARD_PKA: Dict[str, float] = {
    "ASP": 3.9,
    "GLU": 4.1,
    "HIS": 6.0,
    "CYS": 8.3,
    "TYR": 10.1,
    "LYS": 10.5,
    "ARG": 12.5,
}

# Protonaciones especiales que anulan el cálculo por pKa
SPECIAL_PROTONATION: Dict[Tuple[str, str], str] = {
    # Asp catalítico de la proteasa del VIH-1 (monoprotonado en el dímero)
    ("ASP", "25A"): "PROTONATED",    # dona H → DONOR
    ("ASP", "25B"): "DEPROTONATED",  # acepta H → ACCEPTOR
}

SITE_CUTOFF_DEFAULT = 6.0   # Å desde cualquier átomo del ligando
PH_PHYSIOLOGICAL   = 7.4


def generate_rbp(
    pdb_file: str,
    ligand_resname: str = "RIT",
    site_cutoff: float = SITE_CUTOFF_DEFAULT,
    ph: float = PH_PHYSIOLOGICAL,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """
    Genera features RBP.

    Parameters
    ----------
    pdb_file        : ruta al PDB
    ligand_resname  : nombre del residuo ligando
    site_cutoff     : radio (Å) para definir el sitio activo
    ph              : pH para calcular estados de protonación
    verbose         : imprimir info

    Returns
    -------
    Lista de features con 'type', 'coords', 'label', 'source', 'residue'
    """
    from Bio.PDB import PDBParser
    import warnings
    warnings.filterwarnings("ignore")

    _log = print if verbose else lambda *a, **k: None

    parser    = PDBParser(QUIET=True)
    structure = parser.get_structure("mol", pdb_file)
    model     = structure[0]

    # Coordenadas del ligando
    lig_coords = _get_ligand_coords(model, ligand_resname)
    if not lig_coords:
        print(f"[RBP] ERROR: ligando '{ligand_resname}' no encontrado en {pdb_file}.")
        return []

    _log(f"[RBP] Ligando: {len(lig_coords)} átomos. Cutoff sitio activo: {site_cutoff} Å")

    # Residuos del sitio activo
    active_site = _get_active_site_residues(model, lig_coords, site_cutoff)
    _log(f"[RBP] Sitio activo: {len(active_site)} residuos.")

    # Generar features
    features = []
    for res in active_site:
        res_features = _residue_to_features(res, ph)
        features.extend(res_features)

    for f in features:
        f["source"] = "RBP"

    _log(f"[RBP] {len(features)} features RBP generados.")
    return features


# ── Helpers internos ──────────────────────────────────────────────────────────

def _get_ligand_coords(model, ligand_resname: str) -> List[np.ndarray]:
    coords = []
    name = ligand_resname.strip().upper()
    for chain in model:
        for res in chain:
            if res.get_resname().strip() == name:
                for atom in res.get_atoms():
                    coords.append(atom.get_vector().get_array())
    return coords


def _get_active_site_residues(model, lig_coords, cutoff: float):
    """Residuos proteicos con al menos un átomo a < cutoff Å del ligando."""
    lig_arr = np.array(lig_coords)
    active  = []
    for chain in model:
        for res in chain:
            if res.id[0] != " ":  # ignorar HETATM y agua
                continue
            for atom in res.get_atoms():
                ac = atom.get_vector().get_array()
                if np.min(np.linalg.norm(lig_arr - ac, axis=1)) < cutoff:
                    active.append(res)
                    break
    return active


def _residue_to_features(res, ph: float) -> List[Dict[str, Any]]:
    """
    Convierte un residuo en features farmacofóricos según su protonación a pH `ph`.
    """
    rname  = res.get_resname().strip()
    resid  = str(res.id[1]) + res.id[2].strip()   # ej. "25A"
    chain  = res.get_parent().id

    # Coordenadas del Cα (o del primer átomo si no existe)
    ca_coord = None
    for atom in res.get_atoms():
        if atom.get_name().strip() == "CA":
            ca_coord = tuple(atom.get_vector().get_array())
            break
    if ca_coord is None:
        atoms = list(res.get_atoms())
        if not atoms:
            return []
        ca_coord = tuple(atoms[0].get_vector().get_array())

    protonation = _get_protonation_state(rname, resid, ph)
    features = []

    # ── Residuos capaces de H-bond ────────────────────────────────────────────
    if rname in ("SER", "THR"):
        # Hidroxilo: dona Y acepta H
        oh_coord = _get_atom_coord(res, ["OG", "OG1", "OG2"])
        c = oh_coord or ca_coord
        features += [
            {"type": "DONOR",    "coords": c, "label": f"{rname}{resid}{chain} OH-dona"},
            {"type": "ACCEPTOR", "coords": c, "label": f"{rname}{resid}{chain} OH-acepta"},
        ]

    elif rname == "ASN":
        nd2 = _get_atom_coord(res, ["ND2"])
        od1 = _get_atom_coord(res, ["OD1"])
        if nd2:
            features.append({"type": "DONOR",    "coords": nd2, "label": f"ASN{resid}{chain} ND2"})
        if od1:
            features.append({"type": "ACCEPTOR", "coords": od1, "label": f"ASN{resid}{chain} OD1"})

    elif rname == "GLN":
        ne2 = _get_atom_coord(res, ["NE2"])
        oe1 = _get_atom_coord(res, ["OE1"])
        if ne2:
            features.append({"type": "DONOR",    "coords": ne2, "label": f"GLN{resid}{chain} NE2"})
        if oe1:
            features.append({"type": "ACCEPTOR", "coords": oe1, "label": f"GLN{resid}{chain} OE1"})

    elif rname == "TYR":
        oh_coord = _get_atom_coord(res, ["OH"])
        c = oh_coord or ca_coord
        if protonation in (None, "PROTONATED"):
            features += [
                {"type": "DONOR",    "coords": c, "label": f"TYR{resid}{chain} OH"},
                {"type": "ACCEPTOR", "coords": c, "label": f"TYR{resid}{chain} O⁻/OH"},
            ]
        else:
            features.append({"type": "ACCEPTOR", "coords": c, "label": f"TYR{resid}{chain} O⁻"})

    elif rname in ("ASP", "GLU"):
        o_names = ["OD1", "OD2"] if rname == "ASP" else ["OE1", "OE2"]
        o_coords = [_get_atom_coord(res, [n]) for n in o_names]
        o_coords = [c for c in o_coords if c]
        cg = o_coords[0] if o_coords else ca_coord
        if protonation == "PROTONATED":
            # Asp catalítico protonado → dona H
            features.append({"type": "DONOR",        "coords": cg, "label": f"{rname}{resid}{chain} COOH (protonado)"})
        elif protonation == "DEPROTONATED":
            features.append({"type": "NEG_IONIZABLE", "coords": cg, "label": f"{rname}{resid}{chain} COO⁻"})
            features.append({"type": "ACCEPTOR",      "coords": cg, "label": f"{rname}{resid}{chain} COO⁻ (acepta)"})
        else:
            # pH 7.4 >> pKa → casi siempre desprotonado
            features.append({"type": "NEG_IONIZABLE", "coords": cg, "label": f"{rname}{resid}{chain} COO⁻"})
            features.append({"type": "ACCEPTOR",      "coords": cg, "label": f"{rname}{resid}{chain} COO⁻ (acepta)"})

    elif rname in ("LYS",):
        nz = _get_atom_coord(res, ["NZ"]) or ca_coord
        features += [
            {"type": "POS_IONIZABLE", "coords": nz, "label": f"LYS{resid}{chain} NH3⁺"},
            {"type": "DONOR",         "coords": nz, "label": f"LYS{resid}{chain} NH3⁺ dona"},
        ]

    elif rname == "ARG":
        cz  = _get_atom_coord(res, ["CZ"]) or ca_coord
        features += [
            {"type": "POS_IONIZABLE", "coords": cz, "label": f"ARG{resid}{chain} guanidinio"},
            {"type": "DONOR",         "coords": cz, "label": f"ARG{resid}{chain} guanidinio dona"},
        ]

    elif rname == "HIS":
        nd1 = _get_atom_coord(res, ["ND1"])
        ne2 = _get_atom_coord(res, ["NE2"])
        if protonation == "PROTONATED":
            # Histidina protonada (HIP): ambos N donan
            if nd1:
                features.append({"type": "POS_IONIZABLE", "coords": nd1, "label": f"HIS{resid}{chain} ND1 (HIP)"})
            if ne2:
                features.append({"type": "DONOR",         "coords": ne2, "label": f"HIS{resid}{chain} NE2 (HIP)"})
        else:
            # HID/HIE neutro
            c = nd1 or ne2 or ca_coord
            features += [
                {"type": "DONOR",    "coords": c, "label": f"HIS{resid}{chain} N-H"},
                {"type": "ACCEPTOR", "coords": c, "label": f"HIS{resid}{chain} N:"},
            ]

    elif rname == "CYS":
        sg = _get_atom_coord(res, ["SG"]) or ca_coord
        if protonation == "PROTONATED":
            features += [
                {"type": "DONOR",    "coords": sg, "label": f"CYS{resid}{chain} SH"},
                {"type": "ACCEPTOR", "coords": sg, "label": f"CYS{resid}{chain} S:"},
            ]
        else:
            features += [
                {"type": "NEG_IONIZABLE", "coords": sg, "label": f"CYS{resid}{chain} S⁻"},
                {"type": "ACCEPTOR",      "coords": sg, "label": f"CYS{resid}{chain} S⁻ (acepta)"},
            ]

    # ── Residuos hidrofóbicos ─────────────────────────────────────────────────
    elif rname in ("ALA", "VAL", "LEU", "ILE", "PRO", "PHE", "TRP",
                   "MET", "GLY"):
        features.append({
            "type": "HYDROPHOBIC",
            "coords": ca_coord,
            "label": f"{rname}{resid}{chain} hidrofóbico",
        })

    # ── Backbone H-bond (cualquier residuo) ──────────────────────────────────
    n_coord = _get_atom_coord(res, ["N"])
    o_coord = _get_atom_coord(res, ["O"])
    if n_coord and rname != "PRO":  # PRO no tiene H en N
        features.append({"type": "DONOR",    "coords": n_coord, "label": f"{rname}{resid}{chain} N-H backbone"})
    if o_coord:
        features.append({"type": "ACCEPTOR", "coords": o_coord, "label": f"{rname}{resid}{chain} C=O backbone"})

    for f in features:
        f["residue"] = f"{rname}{resid}{chain}"

    return features


def _get_protonation_state(rname: str, resid: str, ph: float) -> str:
    """
    Devuelve 'PROTONATED', 'DEPROTONATED' o None (usar pKa).
    """
    key = (rname, resid)
    if key in SPECIAL_PROTONATION:
        return SPECIAL_PROTONATION[key]
    pka = STANDARD_PKA.get(rname)
    if pka is None:
        return None
    # Regla de Henderson-Hasselbalch simplificada:
    # pH < pKa → protonado; pH > pKa → desprotonado
    if ph < pka:
        return "PROTONATED"
    return "DEPROTONATED"


def _get_atom_coord(res, names: List[str]):
    """Devuelve coordenadas del primer átomo encontrado, o None."""
    for name in names:
        try:
            atom = res[name]
            return tuple(atom.get_vector().get_array())
        except KeyError:
            continue
    return None
