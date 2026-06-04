"""
lbp.py — Farmacóforo Basado en Ligando (Ligand-Based Pharmacophore).

Genera features a partir de uno o múltiples SMILES usando RDKit.
Cuando se proporciona una lista de ligandos, el modelo LBP es el
conjunto de features compartidos por al menos `min_frequency` de ellos
(modelo multi-ligando tipo PHASE/LigandScout).

Referencia: RDKit BaseFeatures.fdef (Riniker & Landrum, 2015).
"""

import numpy as np
from typing import List, Dict, Any, Optional
from collections import Counter, defaultdict


# Mapeo de familias RDKit → tipos canónicos
FTYPE_MAP = {
    "Acceptor"          : "ACCEPTOR",
    "Donor"             : "DONOR",
    "Hydrophobe"        : "HYDROPHOBIC",
    "LumpedHydrophobe"  : "HYDROPHOBIC",
    "PosIonizable"      : "POS_IONIZABLE",
    "NegIonizable"      : "NEG_IONIZABLE",
    "Aromatic"          : "HYDROPHOBIC",
}


def generate_lbp(
    smiles_list: List[str],
    align_to_pdb: Optional[str] = None,
    align_ligand_resname: str = "RIT",
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """
    Genera features LBP para uno o varios SMILES.

    Si se proporciona un PDB con el ligando cristalográfico, el primer
    SMILES se alinea a él para poner el LBP en el espacio de coordenadas
    del receptor.

    Parameters
    ----------
    smiles_list          : lista de SMILES; el primero es el ligando principal
    align_to_pdb         : ruta al PDB para alinear (opcional)
    align_ligand_resname : nombre del residuo del ligando en el PDB
    verbose              : imprimir información de progreso

    Returns
    -------
    Lista de features con 'type', 'coords', 'label', 'source'
    """
    _log = print if verbose else lambda *a, **k: None

    from rdkit import Chem
    from rdkit.Chem import AllChem, ChemicalFeatures, RDConfig
    import os

    fdef_path = os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
    factory   = ChemicalFeatures.BuildFeatureFactory(fdef_path)

    all_features_per_mol = []
    mols_3d = []

    for i, smi in enumerate(smiles_list):
        smi = smi.strip()
        if not smi:
            continue
        mol = _smiles_to_3d(smi, verbose, index=i)
        if mol is None:
            _log(f"[LBP] SMILES #{i+1} descartado (no se pudo generar 3D).")
            continue
        mols_3d.append(mol)
        feats = _extract_features(mol, factory)
        all_features_per_mol.append(feats)
        _log(f"[LBP] Molécula #{i+1}: {len(feats)} features crudos.")

    if not all_features_per_mol:
        return []

    # Si hay un solo ligando, tomar todos sus features
    if len(all_features_per_mol) == 1:
        features = all_features_per_mol[0]
    else:
        # Multi-ligando: tomar features presentes en ≥50% de los ligandos
        features = _merge_multi_ligand(all_features_per_mol, verbose)

    # Alinear al espacio del receptor si se proporciona PDB
    if align_to_pdb and mols_3d:
        features = _align_to_crystal(
            features, mols_3d[0], align_to_pdb, align_ligand_resname, verbose
        )

    for f in features:
        f["source"] = "LBP"

    _log(f"[LBP] {len(features)} features LBP generados.")
    return features


# ── Helpers internos ──────────────────────────────────────────────────────────

def _smiles_to_3d(smiles: str, verbose: bool, index: int = 0):
    """Convierte un SMILES en una molécula 3D con MMFF94."""
    from rdkit import Chem
    from rdkit.Chem import AllChem
    import sys

    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        print(f"[LBP] Error: no se pudo parsear SMILES #{index+1}: {smiles[:60]}",
              file=sys.stderr)
        return None
    try:
        Chem.SanitizeMol(mol)
    except Exception as e:
        print(f"[LBP] Advertencia sanitización #{index+1}: {e}", file=sys.stderr)

    mol3d = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42 + index
    status = AllChem.EmbedMolecule(mol3d, params)
    if status != 0:
        # Reintento con ETKDG clásico
        status = AllChem.EmbedMolecule(mol3d, AllChem.ETKDG())
    if status != 0:
        print(f"[LBP] Error: no se pudo generar conformación 3D para SMILES #{index+1}.",
              file=sys.stderr)
        return None

    AllChem.MMFFOptimizeMolecule(mol3d, maxIters=2000)
    return Chem.RemoveHs(mol3d)


def _extract_features(mol, factory) -> List[Dict[str, Any]]:
    """Extrae features farmacofóricos de una molécula 3D con RDKit."""
    conf     = mol.GetConformer()
    features = []
    for feat in factory.GetFeaturesForMol(mol):
        fname = feat.GetFamily()
        ftype = FTYPE_MAP.get(fname)
        if ftype is None:
            continue
        atom_ids = list(feat.GetAtomIds())
        coords   = [conf.GetAtomPosition(idx) for idx in atom_ids]
        cx = float(np.mean([c.x for c in coords]))
        cy = float(np.mean([c.y for c in coords]))
        cz = float(np.mean([c.z for c in coords]))
        features.append({
            "type"  : ftype,
            "family": fname,
            "coords": (cx, cy, cz),
            "atoms" : atom_ids,
            "label" : f"{fname} → {ftype} (átomos: {atom_ids})",
        })
    return features


def _merge_multi_ligand(
    all_features: List[List[Dict]],
    verbose: bool,
    overlap_radius: float = 2.0,
    min_freq: float = 0.5,
) -> List[Dict[str, Any]]:
    """
    Para múltiples ligandos: conserva features que aparecen en al menos
    `min_freq` fracción de los ligandos. Los features similares (mismo
    tipo, distancia < overlap_radius) se fusionan en su centroide.

    Estrategia: votar features por tipo entre moléculas.
    """
    _log = print if verbose else lambda *a, **k: None
    n_mols = len(all_features)
    min_count = max(1, int(np.ceil(min_freq * n_mols)))

    # Aplanar con etiqueta de molécula
    tagged = []
    for mol_idx, feats in enumerate(all_features):
        for f in feats:
            tagged.append({**f, "_mol": mol_idx})

    # Agrupar por tipo
    by_type = defaultdict(list)
    for f in tagged:
        by_type[f["type"]].append(f)

    consensus = []
    for ftype, group in by_type.items():
        # Clustering greedy dentro del tipo
        clusters = []
        centroids = []
        for f in group:
            fc = np.array(f["coords"], dtype=float)
            placed = False
            for k, ctr in enumerate(centroids):
                if np.linalg.norm(fc - ctr) < overlap_radius:
                    clusters[k].append(f)
                    n = len(clusters[k])
                    centroids[k] = ctr + (fc - ctr) / n
                    placed = True
                    break
            if not placed:
                clusters.append([f])
                centroids.append(fc.copy())

        for cluster, ctr in zip(clusters, centroids):
            mols_present = set(f["_mol"] for f in cluster)
            if len(mols_present) >= min_count:
                consensus.append({
                    "type"     : ftype,
                    "coords"   : tuple(ctr),
                    "label"    : f"{ftype} multi-ligando ({len(mols_present)}/{n_mols} mols)",
                    "n_mols"   : len(mols_present),
                })

    _log(f"[LBP] Multi-ligando: {len(consensus)} features consenso "
         f"(mín {min_count}/{n_mols} ligandos).")
    return consensus


def _align_to_crystal(
    features: List[Dict],
    mol3d,
    pdb_file: str,
    ligand_resname: str,
    verbose: bool,
) -> List[Dict[str, Any]]:
    """
    Alinea las coordenadas LBP al espacio del receptor usando el RMSD
    de alineamiento 3D entre el ligando SMILES y el ligando cristalográfico.
    """
    _log = print if verbose else lambda *a, **k: None

    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem, rdMolAlign
        from Bio.PDB import PDBParser
        import io, warnings
        warnings.filterwarnings("ignore")

        # Extraer coordenadas del ligando cristalográfico
        parser = PDBParser(QUIET=True)
        struct = parser.get_structure("ref", pdb_file)
        crystal_atoms = []
        for chain in struct[0]:
            for res in chain:
                if res.get_resname().strip() == ligand_resname.strip().upper():
                    crystal_atoms = list(res.get_atoms())
                    break

        if not crystal_atoms:
            _log(f"[LBP] Advertencia: ligando '{ligand_resname}' no encontrado en PDB, sin alineamiento.")
            return features

        # Construir molécula de referencia desde coordenadas del cristal
        ref_coords = np.array([a.get_vector().get_array() for a in crystal_atoms])
        n_heavy_ref = len(ref_coords)
        n_heavy_mol = mol3d.GetNumAtoms()

        if n_heavy_mol != n_heavy_ref:
            _log(f"[LBP] Advertencia: átomos pesados distintos ({n_heavy_mol} vs {n_heavy_ref}). "
                 "Alineamiento por RMSD omitido, se usan coords relativas.")
            return features

        # RMSD-align mol3d al cristal
        conf = mol3d.GetConformer()
        mol_coords = np.array([list(conf.GetAtomPosition(i)) for i in range(n_heavy_mol)])

        # Centrar y alinear (Kabsch)
        R, t = _kabsch(mol_coords, ref_coords)
        mol_coords_aligned = (mol_coords @ R.T) + t
        rmsd = float(np.sqrt(np.mean(np.sum((mol_coords_aligned - ref_coords)**2, axis=1))))
        _log(f"[LBP] RMSD de alineamiento al cristal: {rmsd:.3f} Å")

        # Trasladar features LBP con la misma transformación
        aligned = []
        for f in features:
            fc = np.array(f["coords"], dtype=float)
            fc_al = (fc @ R.T) + t
            aligned.append({**f, "coords": tuple(fc_al)})
        return aligned

    except Exception as e:
        _log(f"[LBP] Alineamiento fallido ({e}), se devuelven features sin alinear.")
        return features


def _kabsch(P: np.ndarray, Q: np.ndarray):
    """
    Algoritmo de Kabsch: encuentra R y t que minimiza RMSD entre P y Q.
    P_aligned = P @ R.T + t
    """
    p_mean = P.mean(axis=0)
    q_mean = Q.mean(axis=0)
    P_c = P - p_mean
    Q_c = Q - q_mean

    H  = P_c.T @ Q_c
    U, _, Vt = np.linalg.svd(H)
    d  = np.linalg.det(Vt.T @ U.T)
    D  = np.diag([1, 1, d])
    R  = Vt.T @ D @ U.T
    t  = q_mean - p_mean @ R.T
    return R, t
