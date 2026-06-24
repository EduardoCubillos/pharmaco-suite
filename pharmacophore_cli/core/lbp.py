"""
lbp.py — Ligand-Based Pharmacophore (LBP).

Extracts pharmacophoric features from one or more SMILES using RDKit.
When multiple ligands are given, the model retains only features present
in at least `min_frequency` of them — similar to PHASE/LigandScout
multi-ligand consensus mode.

If a co-crystal PDB is provided, the LBP is aligned to receptor space
via RMSD alignment of the SMILES-generated 3D conformer to the crystal
ligand (Kabsch algorithm).

Reference: RDKit BaseFeatures.fdef — Riniker & Landrum (2015).
"""

import numpy as np
import os
import sys
from typing import List, Dict, Any, Optional
from collections import defaultdict


# Map RDKit feature families to canonical types used across the pipeline
_RDKIT_FAMILY_MAP = {
    "Acceptor"         : "ACCEPTOR",
    "Donor"            : "DONOR",
    "Hydrophobe"       : "HYDROPHOBIC",
    "LumpedHydrophobe" : "HYDROPHOBIC",
    "PosIonizable"     : "POS_IONIZABLE",
    "NegIonizable"     : "NEG_IONIZABLE",
    "Aromatic"         : "HYDROPHOBIC",
}


def generate_lbp(
    smiles_list: List[str],
    align_to_pdb: Optional[str] = None,
    align_ligand_resname: str = "RIT",
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """
    Generate LBP features for one or more SMILES.

    Parameters
    ----------
    smiles_list          : SMILES to process; first one is treated as the primary ligand
    align_to_pdb         : PDB file to align LBP coordinates into receptor space
    align_ligand_resname : residue name of the reference ligand in the PDB
    verbose              : print progress

    Returns
    -------
    List of feature dicts with 'type', 'coords', 'label', 'source'
    """
    _log = print if verbose else lambda *a, **k: None

    from rdkit import Chem
    from rdkit.Chem import AllChem, ChemicalFeatures, RDConfig

    fdef_path = os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
    factory   = ChemicalFeatures.BuildFeatureFactory(fdef_path)

    mol_feature_sets = []
    mols_3d = []

    for i, smi in enumerate(smiles_list):
        smi = smi.strip()
        if not smi:
            continue
        mol = _smiles_to_3d(smi, verbose, index=i)
        if mol is None:
            _log(f"[LBP] SMILES #{i+1} skipped (could not generate 3D conformer).")
            continue
        mols_3d.append(mol)
        feats = _extract_rdkit_features(mol, factory)
        mol_feature_sets.append(feats)
        _log(f"[LBP] Molecule #{i+1}: {len(feats)} raw features.")

    if not mol_feature_sets:
        return []

    if len(mol_feature_sets) == 1:
        features = mol_feature_sets[0]
    else:
        # Keep features present in ≥50% of ligands
        features = _multi_ligand_consensus(mol_feature_sets, verbose)

    # Align into receptor coordinate space if a crystal structure is available
    if align_to_pdb and mols_3d:
        features = _align_to_crystal_ligand(
            features, mols_3d[0], align_to_pdb, align_ligand_resname, verbose
        )

    for f in features:
        f["source"] = "LBP"

    _log(f"[LBP] {len(features)} LBP features generated.")
    return features


# ── Internals ─────────────────────────────────────────────────────────────────

def _smiles_to_3d(smiles: str, verbose: bool, index: int = 0):
    """Embed a SMILES into a 3D conformer using ETKDGv3 + MMFF94."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        print(f"[LBP] Error: could not parse SMILES #{index+1}: {smiles[:60]}",
              file=sys.stderr)
        return None
    try:
        Chem.SanitizeMol(mol)
    except Exception as e:
        print(f"[LBP] Sanitization warning #{index+1}: {e}", file=sys.stderr)

    mol3d  = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42 + index
    status = AllChem.EmbedMolecule(mol3d, params)
    if status != 0:
        # Retry with classic ETKDG — sometimes works when v3 doesn't
        status = AllChem.EmbedMolecule(mol3d, AllChem.ETKDG())
    if status != 0:
        print(f"[LBP] Error: 3D embedding failed for SMILES #{index+1}.",
              file=sys.stderr)
        return None

    AllChem.MMFFOptimizeMolecule(mol3d, maxIters=2000)
    return Chem.RemoveHs(mol3d)


def _extract_rdkit_features(mol, factory) -> List[Dict[str, Any]]:
    """Get pharmacophoric features from a 3D RDKit molecule."""
    conf     = mol.GetConformer()
    features = []
    for feat in factory.GetFeaturesForMol(mol):
        family = feat.GetFamily()
        ftype  = _RDKIT_FAMILY_MAP.get(family)
        if ftype is None:
            continue
        atom_ids = list(feat.GetAtomIds())
        positions = [conf.GetAtomPosition(idx) for idx in atom_ids]
        cx = float(np.mean([p.x for p in positions]))
        cy = float(np.mean([p.y for p in positions]))
        cz = float(np.mean([p.z for p in positions]))
        features.append({
            "type"  : ftype,
            "family": family,
            "coords": (cx, cy, cz),
            "atoms" : atom_ids,
            "label" : f"{family} → {ftype} (atoms: {atom_ids})",
        })
    return features


def _multi_ligand_consensus(
    mol_feature_sets: List[List[Dict]],
    verbose: bool,
    overlap_radius: float = 2.0,
    min_freq: float = 0.5,
) -> List[Dict[str, Any]]:
    """
    Multi-ligand LBP: keep features present in at least `min_freq` fraction
    of the input molecules. Features of the same type within `overlap_radius`
    of each other are merged into their centroid.
    """
    _log = print if verbose else lambda *a, **k: None
    n_mols    = len(mol_feature_sets)
    min_count = max(1, int(np.ceil(min_freq * n_mols)))

    # Tag each feature with its molecule index
    tagged = []
    for mol_idx, feats in enumerate(mol_feature_sets):
        for f in feats:
            tagged.append({**f, "_mol": mol_idx})

    by_type: Dict[str, List] = defaultdict(list)
    for f in tagged:
        by_type[f["type"]].append(f)

    result = []
    for ftype, group in by_type.items():
        # Greedy spatial clustering per type
        clusters  = []
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
            n_mols_present = len({f["_mol"] for f in cluster})
            if n_mols_present >= min_count:
                result.append({
                    "type"  : ftype,
                    "coords": tuple(ctr),
                    "label" : f"{ftype} multi-ligand ({n_mols_present}/{n_mols} mols)",
                    "n_mols": n_mols_present,
                })

    _log(f"[LBP] Multi-ligand: {len(result)} consensus features "
         f"(min {min_count}/{n_mols} ligands).")
    return result


def _align_to_crystal_ligand(
    features: List[Dict],
    mol3d,
    pdb_file: str,
    ligand_resname: str,
    verbose: bool,
) -> List[Dict[str, Any]]:
    """
    Align LBP feature coordinates to receptor space by superimposing
    the RDKit 3D conformer onto the crystal ligand (Kabsch RMSD alignment).
    """
    _log = print if verbose else lambda *a, **k: None

    try:
        from Bio.PDB import PDBParser
        import warnings
        warnings.filterwarnings("ignore")

        parser = PDBParser(QUIET=True)
        struct = parser.get_structure("ref", pdb_file)
        crystal_atoms = []
        for chain in struct[0]:
            for res in chain:
                if res.get_resname().strip() == ligand_resname.strip().upper():
                    crystal_atoms = list(res.get_atoms())
                    break

        if not crystal_atoms:
            _log(f"[LBP] Warning: ligand '{ligand_resname}' not found in PDB — skipping alignment.")
            return features

        ref_coords   = np.array([a.get_vector().get_array() for a in crystal_atoms])
        n_heavy_ref  = len(ref_coords)
        n_heavy_mol  = mol3d.GetNumAtoms()

        if n_heavy_mol != n_heavy_ref:
            _log(f"[LBP] Warning: atom count mismatch ({n_heavy_mol} vs {n_heavy_ref}). "
                 "Skipping alignment — using relative LBP coordinates.")
            return features

        conf = mol3d.GetConformer()
        mol_coords = np.array([list(conf.GetAtomPosition(i)) for i in range(n_heavy_mol)])

        R, t = _kabsch(mol_coords, ref_coords)
        mol_coords_aligned = (mol_coords @ R.T) + t
        rmsd = float(np.sqrt(np.mean(np.sum((mol_coords_aligned - ref_coords)**2, axis=1))))
        _log(f"[LBP] Alignment RMSD to crystal: {rmsd:.3f} Å")

        # Apply same transformation to all feature coordinates
        aligned = []
        for f in features:
            fc = np.array(f["coords"], dtype=float)
            aligned.append({**f, "coords": tuple((fc @ R.T) + t)})
        return aligned

    except Exception as e:
        _log(f"[LBP] Alignment failed ({e}), returning unaligned features.")
        return features


def _kabsch(P: np.ndarray, Q: np.ndarray):
    """
    Kabsch algorithm: find rotation R and translation t minimizing RMSD(P, Q).
    Usage: P_aligned = P @ R.T + t
    """
    p_mean = P.mean(axis=0)
    q_mean = Q.mean(axis=0)
    Pc     = P - p_mean
    Qc     = Q - q_mean

    H  = Pc.T @ Qc
    U, _, Vt = np.linalg.svd(H)
    d  = np.linalg.det(Vt.T @ U.T)
    D  = np.diag([1, 1, d])
    R  = Vt.T @ D @ U.T
    t  = q_mean - p_mean @ R.T
    return R, t
