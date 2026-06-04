"""
pharma_scorer.py — Puntuación farmacofórica contra modelos SBP/LBP/consenso.

Dos modos:
  1. Feature-based : cuenta qué fracción de features del modelo están
                     químicamente presentes en la molécula candidata.
  2. PDB-based     : carga un PDB de farmacóforo generado por pharmacophore_cli
                     y evalúa distancias 3D entre la molécula y los features.
"""

import os
import sys
import numpy as np
from typing import List, Dict, Any, Optional

# Mapeo familias RDKit → tipos canónicos
FTYPE_MAP = {
    "Acceptor"          : "ACCEPTOR",
    "Donor"             : "DONOR",
    "Hydrophobe"        : "HYDROPHOBIC",
    "LumpedHydrophobe"  : "HYDROPHOBIC",
    "Aromatic"          : "HYDROPHOBIC",
    "PosIonizable"      : "POS_IONIZABLE",
    "NegIonizable"      : "NEG_IONIZABLE",
}

_factory = None

def _get_factory():
    global _factory
    if _factory is None:
        from rdkit.Chem import ChemicalFeatures, RDConfig
        fdef = os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
        _factory = ChemicalFeatures.BuildFeatureFactory(fdef)
    return _factory


def get_feature_counts(mol) -> Dict[str, int]:
    """Cuenta features farmacofóricos de una molécula con RDKit."""
    if mol is None:
        return {}
    counts = {}
    try:
        for feat in _get_factory().GetFeaturesForMol(mol):
            ftype = FTYPE_MAP.get(feat.GetFamily())
            if ftype:
                counts[ftype] = counts.get(ftype, 0) + 1
    except Exception:
        pass
    return counts


def build_reference_profile(smiles_list: List[str]) -> Dict[str, int]:
    """
    Construye el perfil de features de referencia como la UNIÓN
    de los features presentes en al menos la mitad de los ligandos.
    Usado para LBP multi-ligando.
    """
    from rdkit import Chem
    from collections import Counter

    all_counts = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            all_counts.append(get_feature_counts(mol))

    if not all_counts:
        return {}

    n = len(all_counts)
    min_freq = max(1, int(np.ceil(0.5 * n)))   # presente en ≥50% de ligandos

    # Contar en cuántas moléculas aparece cada (tipo, count)
    type_counts = Counter()
    for counts in all_counts:
        for ftype, n_feat in counts.items():
            type_counts[ftype] += 1

    profile = {ftype: 1 for ftype, freq in type_counts.items() if freq >= min_freq}
    return profile


def pharmacophore_score(mol, reference: Dict[str, int]) -> float:
    """
    Fracción de features del perfil de referencia presentes en mol.
    0.0 = ningún feature | 1.0 = todos los features presentes.
    """
    if mol is None or not reference:
        return 0.0
    counts = get_feature_counts(mol)
    total  = sum(reference.values())
    if total == 0:
        return 0.0
    match  = sum(min(counts.get(ft, 0), n) for ft, n in reference.items())
    return match / total


def load_pharmacophore_pdb(pdb_file: str) -> List[Dict[str, Any]]:
    """
    Carga features desde un PDB generado por pharmacophore_cli.
    Formato: ATOM con nombre A/D/H/P/N, coordenadas x/y/z, occupancy=weight/3.
    """
    TYPE_FROM_CHAR = {
        "A": "ACCEPTOR", "D": "DONOR", "H": "HYDROPHOBIC",
        "P": "POS_IONIZABLE", "N": "NEG_IONIZABLE",
    }
    features = []
    with open(pdb_file) as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            atom_name = line[12:16].strip()
            char = atom_name[0] if atom_name else "?"
            ftype = TYPE_FROM_CHAR.get(char)
            if ftype is None:
                continue
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                occ = float(line[54:60]) if len(line) > 60 else 1.0
            except ValueError:
                continue
            features.append({
                "type"   : ftype,
                "coords" : (x, y, z),
                "weight" : max(1, round(occ * 3)),
            })
    return features


def score_against_pdb_pharmacophore(
    mol,
    features: List[Dict],
    match_radius: float = 2.0,
) -> float:
    """
    Puntúa una molécula contra un farmacóforo PDB midiendo distancias 3D.
    Requiere que la molécula tenga una conformación 3D generada.

    Para cada feature del farmacóforo, busca el átomo de mol del tipo correcto
    más cercano. Si está dentro de match_radius Å, se considera un match.

    Returns
    -------
    Fracción de features del PDB que hacen match [0.0 – 1.0].
    Pondera por weight del feature (features triple-consenso cuentan más).
    """
    if mol is None or not features:
        return 0.0

    try:
        conf = mol.GetConformer()
    except Exception:
        return 0.0

    mol_features = get_feature_counts(mol)
    if not mol_features:
        return 0.0

    matched_weight = 0.0
    total_weight   = 0.0

    for feat in features:
        w     = feat.get("weight", 1)
        ftype = feat["type"]
        fc    = np.array(feat["coords"], dtype=float)
        total_weight += w

        # Buscar átomo de ese tipo en la molécula más cercano al feature
        best_dist = float("inf")
        mol_feat_positions = _get_type_positions(mol, ftype, conf)
        for pos in mol_feat_positions:
            d = np.linalg.norm(fc - np.array(pos))
            if d < best_dist:
                best_dist = d

        if best_dist <= match_radius:
            matched_weight += w

    return matched_weight / total_weight if total_weight > 0 else 0.0


def _get_type_positions(mol, ftype: str, conf):
    """Devuelve posiciones 3D de los features del tipo dado en la molécula."""
    positions = []
    try:
        for feat in _get_factory().GetFeaturesForMol(mol):
            if FTYPE_MAP.get(feat.GetFamily()) == ftype:
                atom_ids = list(feat.GetAtomIds())
                coords = [conf.GetAtomPosition(i) for i in atom_ids]
                cx = np.mean([c.x for c in coords])
                cy = np.mean([c.y for c in coords])
                cz = np.mean([c.z for c in coords])
                positions.append((cx, cy, cz))
    except Exception:
        pass
    return positions
