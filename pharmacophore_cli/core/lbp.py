"""
lbp.py — Ligand-Based Pharmacophore (LBP).

Extracts pharmacophoric features from one or more SMILES using RDKit.
When multiple ligands are given, the model retains only features present
in at least `min_frequency` of them — similar to PHASE/LigandScout
multi-ligand consensus mode.

If a co-crystal PDB is provided, the LBP is aligned to receptor space by
superimposing the SMILES-generated 3D conformer onto the crystal ligand.
Atom correspondence comes from a symmetry-aware substructure match
between the two molecules, never from atom order, and the resulting fit
is accepted only if its RMSD falls below `max_align_rmsd`. Features carry
a `frame` field ('receptor' or 'ligand') so that downstream consensus
never fuses coordinates expressed in different reference frames.

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


MAX_ALIGN_RMSD_DEFAULT = 1.5   # Å — above this the superposition is rejected


def generate_lbp(
    smiles_list: List[str],
    align_to_pdb: Optional[str] = None,
    align_ligand_resname: str = "RIT",
    verbose: bool = True,
    max_align_rmsd: float = MAX_ALIGN_RMSD_DEFAULT,
) -> List[Dict[str, Any]]:
    """
    Generate LBP features for one or more SMILES.

    Parameters
    ----------
    smiles_list          : SMILES to process; first one is treated as the primary ligand
    align_to_pdb         : PDB file to align LBP coordinates into receptor space
    align_ligand_resname : residue name of the reference ligand in the PDB
    verbose              : print progress
    max_align_rmsd       : reject the superposition above this RMSD (Å)

    Returns
    -------
    List of feature dicts with 'type', 'coords', 'label', 'source' and
    'frame' ('receptor' if successfully aligned to the crystal ligand,
    'ligand' if the coordinates remain in the conformer's own frame).
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
    frame = "ligand"
    if align_to_pdb and mols_3d:
        features, frame = _align_to_crystal_ligand(
            features, mols_3d[0], align_to_pdb, align_ligand_resname,
            verbose, max_align_rmsd,
        )

    for f in features:
        f["source"] = "LBP"
        f["frame"]  = frame

    _log(f"[LBP] {len(features)} LBP features generated (frame: {frame}).")
    if align_to_pdb and frame == "ligand":
        _log("[LBP] WARNING: coordinates are NOT in receptor space. They describe "
             "the ligand's internal geometry only and must not be fused with "
             "SBP/RBP features or compared to the crystal site.")
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
    max_rmsd: float = MAX_ALIGN_RMSD_DEFAULT,
):
    """
    Align LBP feature coordinates to receptor space by superimposing the
    RDKit 3D conformer onto the crystal ligand.

    Atom correspondence is established by substructure matching between the
    conformer and the crystal copy of the same molecule, enumerating the
    graph automorphisms so that symmetry-equivalent atoms (the two oxygens
    of a carboxylate, the ortho positions of a phenyl) are not forced into
    an arbitrary pairing. Of all valid matches the one with the lowest
    Kabsch RMSD is kept, and the fit is accepted only below `max_rmsd`.

    Returns
    -------
    (features, frame) where frame is 'receptor' if the transformation was
    applied and 'ligand' if the original coordinates were kept.
    """
    _log = print if verbose else lambda *a, **k: None

    try:
        ref = _load_crystal_ligand(pdb_file, ligand_resname)
        if ref is None:
            _log(f"[LBP] Warning: ligand '{ligand_resname}' not found in {pdb_file} "
                 "— no alignment performed.")
            return features, "ligand"

        probe = _strip_hydrogens(mol3d)
        amap  = _best_atom_map(probe, ref)
        if amap is None:
            _log(f"[LBP] Warning: the conformer and the crystal copy of "
                 f"'{ligand_resname}' do not share a common substructure "
                 f"({probe.GetNumAtoms()} vs {ref.GetNumAtoms()} heavy atoms). "
                 "No alignment performed.")
            return features, "ligand"

        probe_idx, ref_idx, rmsd, R, t = amap
        n_used = len(probe_idx)
        _log(f"[LBP] Superposition on {n_used} matched heavy atoms: RMSD {rmsd:.3f} Å")

        if rmsd > max_rmsd:
            _log(f"[LBP] Warning: RMSD {rmsd:.3f} Å exceeds the {max_rmsd} Å "
                 "threshold — the conformer does not reproduce the crystal pose. "
                 "No alignment performed.")
            return features, "ligand"

        aligned = []
        for f in features:
            fc = np.array(f["coords"], dtype=float)
            aligned.append({**f, "coords": tuple((fc @ R.T) + t),
                            "align_rmsd": round(rmsd, 3)})
        return aligned, "receptor"

    except Exception as e:
        _log(f"[LBP] Alignment failed ({e}) — no alignment performed.")
        return features, "ligand"


def _load_crystal_ligand(pdb_file: str, ligand_resname: str):
    """
    Read the first copy of `ligand_resname` from a PDB file into an RDKit
    molecule with 3D coordinates. Only one residue instance is taken, so a
    structure with several copies of the ligand cannot produce a chimeric
    reference. Alternate locations other than A are discarded.
    """
    from rdkit import Chem

    want     = ligand_resname.strip().upper()
    selected = None
    lines    = []

    with open(pdb_file) as fh:
        for line in fh:
            if not line.startswith(("HETATM", "ATOM  ")):
                continue
            if line[17:20].strip().upper() != want:
                continue
            altloc = line[16]
            if altloc not in (" ", "A"):
                continue
            key = (line[21], line[22:27])          # chain + resseq + icode
            if selected is None:
                selected = key
            elif key != selected:
                continue
            lines.append(line)

    if not lines:
        return None

    block = "".join(lines) + "END\n"
    ref = Chem.MolFromPDBBlock(block, sanitize=False, removeHs=True,
                               proximityBonding=True)
    if ref is None:
        return None
    _sanitize_loosely(ref)
    return ref


def _strip_hydrogens(mol):
    """Heavy-atom copy of a molecule, keeping its conformer."""
    from rdkit import Chem
    return Chem.RemoveHs(Chem.Mol(mol))


def _sanitize_loosely(mol) -> None:
    """
    Sanitize enough for substructure matching (ring perception, valences)
    without failing on the aromaticity and kekulization problems that
    PDB-derived connectivity routinely produces.
    """
    from rdkit import Chem
    try:
        Chem.SanitizeMol(mol)
        return
    except Exception:
        pass
    flags = (Chem.SanitizeFlags.SANITIZE_ALL
             ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE
             ^ Chem.SanitizeFlags.SANITIZE_SETAROMATICITY
             ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES)
    Chem.SanitizeMol(mol, sanitizeOps=flags, catchErrors=True)


def _best_atom_map(probe, ref, max_matches: int = 1000):
    """
    Find the probe→ref atom correspondence that minimizes Kabsch RMSD.

    Bond orders are ignored, because a PDB ligand carries no bond-order
    information: only the heavy-atom skeleton with element identity and
    connectivity is matched. All symmetry-equivalent mappings returned by
    RDKit are scored and the best one wins.

    Returns (probe_indices, ref_indices, rmsd, R, t) or None.
    """
    from rdkit import Chem

    qp = Chem.AdjustQueryParameters.NoAdjustments()
    qp.makeBondsGeneric = True
    qp.aromatizeIfPossible = False
    query = Chem.AdjustQueryProperties(Chem.Mol(probe), qp)

    matches = ref.GetSubstructMatches(
        query, uniquify=False, useChirality=False, maxMatches=max_matches
    )
    if not matches:
        return None

    probe_conf = probe.GetConformer()
    ref_conf   = ref.GetConformer()
    probe_all  = np.array([list(probe_conf.GetAtomPosition(i))
                           for i in range(probe.GetNumAtoms())])
    ref_all    = np.array([list(ref_conf.GetAtomPosition(i))
                           for i in range(ref.GetNumAtoms())])

    probe_idx = list(range(probe.GetNumAtoms()))
    best = None

    for match in matches:
        ref_idx = list(match)
        P = probe_all[probe_idx]
        Q = ref_all[ref_idx]
        R, t = _kabsch(P, Q)
        rmsd = float(np.sqrt(np.mean(np.sum(((P @ R.T) + t - Q) ** 2, axis=1))))
        if best is None or rmsd < best[2]:
            best = (probe_idx, ref_idx, rmsd, R, t)

    return best


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
