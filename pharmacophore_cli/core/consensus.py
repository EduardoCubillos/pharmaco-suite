"""
consensus.py — Weighted consensus pharmacophore (SBP ∩ LBP ∩ RBP).

Combines three independent models using receptor↔ligand chemical complementarity:
  DONOR (ligand)       ↔  ACCEPTOR (receptor)
  ACCEPTOR (ligand)    ↔  DONOR (receptor)
  HYDROPHOBIC          ↔  HYDROPHOBIC
  POS_IONIZABLE        ↔  NEG_IONIZABLE
  NEG_IONIZABLE        ↔  POS_IONIZABLE

Each feature gets a weight (1–3) based on how many models agree on it.
Features with weight ≥ 2 are the most reliable for virtual screening.

References:
  - Baroni et al. (2007) J. Chem. Inf. Model. 47, 279–294 (PHASE)
  - Sanders et al. (2012) J. Chem. Inf. Model. 52, 1261–1272
"""

import numpy as np
from typing import List, Dict, Any
from collections import defaultdict

# Chemical complementarity: what a ligand feature needs from the receptor
COMPLEMENT_OF: Dict[str, str] = {
    "DONOR"         : "ACCEPTOR",
    "ACCEPTOR"      : "DONOR",
    "HYDROPHOBIC"   : "HYDROPHOBIC",
    "POS_IONIZABLE" : "NEG_IONIZABLE",
    "NEG_IONIZABLE" : "POS_IONIZABLE",
}

# Wolber & Langer / Baroni standard for inter-model feature fusion
CONSENSUS_RADIUS_DEFAULT = 3.5   # Å


def build_consensus(
    sbp: List[Dict[str, Any]],
    lbp: List[Dict[str, Any]],
    rbp: List[Dict[str, Any]],
    radius: float = CONSENSUS_RADIUS_DEFAULT,
    verbose: bool = True,
    min_weight: int = 1,
) -> List[Dict[str, Any]]:
    """
    Build a weighted consensus from SBP, LBP, and RBP models.

    Algorithm:
    1. Pool SBP and LBP features (both describe the ligand). Merge features
       of the same type within `radius` Å into a single centroid. Weight = 2
       if both SBP and LBP contributed, otherwise 1.
    2. For each ligand-perspective feature, search the RBP for a chemically
       complementary feature within `radius` Å. If found, weight += 1.
    3. Filter by min_weight (default 1 = keep everything).

    Why keep single-model features (weight=1)?
    -------------------------------------------
    A feature unique to SBP may represent a real crystal interaction (detected
    by PLIP) that LBP missed because RDKit did not encode that group well from
    SMILES, or because the interaction point sits away from the ligand atom.
    Discarding it silently would discard experimental evidence. Weight ★☆☆
    signals the lower confidence; the user decides the threshold with
    --min-consensus-weight.

    Returns
    -------
    List of consensus features, each with:
      'type'           : feature type (ligand perspective)
      'coords'         : centroid position
      'weight'         : 1–3 (★☆☆ / ★★☆ / ★★★)
      'models'         : set of contributing model names
      'bfactor'        : 33.3 / 66.7 / 100.0
      'rbp_complement' : description of the matching RBP feature, or "—"
    """
    _log = print if verbose else lambda *a, **k: None

    sbp_tagged = [{**f, "source": "SBP"} for f in sbp]
    lbp_tagged = [{**f, "source": "LBP"} for f in lbp]
    rbp_tagged = [{**f, "source": "RBP"} for f in rbp]

    ligand_pool = sbp_tagged + lbp_tagged
    result = []
    used = set()

    for i, fi in enumerate(ligand_pool):
        if i in used:
            continue

        cluster  = [fi]
        sources  = {fi["source"]}
        used.add(i)
        ci    = np.array(fi["coords"], dtype=float)
        ftype = fi["type"]

        # Merge features of the same type within radius
        for j, fj in enumerate(ligand_pool):
            if j == i or j in used or fj["type"] != ftype:
                continue
            if np.linalg.norm(ci - np.array(fj["coords"], dtype=float)) < radius:
                cluster.append(fj)
                sources.add(fj["source"])
                used.add(j)

        centroid = np.mean([np.array(f["coords"], dtype=float) for f in cluster], axis=0)

        # Look for complementary feature in RBP
        comp_type = COMPLEMENT_OF.get(ftype, ftype)
        best_rbp  = None
        best_dist = float("inf")

        for fr in rbp_tagged:
            if fr["type"] != comp_type:
                continue
            d = np.linalg.norm(centroid - np.array(fr["coords"], dtype=float))
            if d < radius and d < best_dist:
                best_dist = d
                best_rbp  = fr

        rbp_label = "—"
        if best_rbp:
            sources.add("RBP")
            rbp_label = f"{comp_type} (d={best_dist:.2f}Å, {best_rbp.get('residue', '?')})"

        weight  = len(sources)
        bfactor = round(weight / 3 * 100, 1)

        result.append({
            "type"           : ftype,
            "coords"         : tuple(centroid),
            "models"         : sources,
            "weight"         : weight,
            "bfactor"        : bfactor,
            "rbp_complement" : rbp_label,
            "n_atoms"        : len(cluster),
            "label"          : f"{ftype} [{'+'.join(sorted(sources))}] w={weight}",
        })

    result.sort(key=lambda x: (-x["weight"], x["type"]))

    triple = [f for f in result if f["weight"] == 3]
    double = [f for f in result if f["weight"] == 2]
    single = [f for f in result if f["weight"] == 1]

    filtered = [f for f in result if f["weight"] >= min_weight]
    n_dropped = len(result) - len(filtered)

    _log(f"\n{'='*60}")
    _log("CONSENSUS PHARMACOPHORE MODEL")
    _log(f"  Clustering radius: {radius} Å")
    _log(f"{'='*60}")
    _log(f"  ★★★ Triple consensus (SBP+LBP+RBP): {len(triple):3d} features  ← most reliable")
    _log(f"  ★★☆ Double consensus               : {len(double):3d} features")
    _log(f"  ★☆☆ Single model                   : {len(single):3d} features  ← possible crystal evidence")
    _log(f"  TOTAL                              : {len(filtered):3d} features"
         + (f"  ({n_dropped} dropped by --min-consensus-weight {min_weight})" if n_dropped else ""))

    if single and min_weight == 1:
        _log(f"\n  Note: the {len(single)} ★☆☆ features come from a single model.")
        _log(f"  They may represent real crystal interactions not captured by LBP.")
        _log(f"  For strict screening use: --min-consensus-weight 2")

    if triple:
        _log("\nTriple-consensus features:")
        _log(f"  {'#':>3}  {'Ligand type':16s}  {'RBP complement':35s}  {'Models'}")
        _log(f"  {'─'*70}")
        for i, feat in enumerate(triple):
            _log(f"  [{i+1:02d}] {feat['type']:16s}  {feat['rbp_complement']:35s}  "
                 f"{'+'.join(sorted(feat['models']))}")

    return filtered


def load_features_from_pdb(pdb_file: str, source_label: str = "LOADED") -> List[Dict[str, Any]]:
    """
    Load pharmacophoric features from a PDB file written by this tool.

    Expected format: ATOM records where atom name starts with the type character
    (A/D/H/P/N), occupancy encodes weight/3, B-factor encodes confidence.
    """
    _CHAR_TO_TYPE = {
        "A": "ACCEPTOR",
        "D": "DONOR",
        "H": "HYDROPHOBIC",
        "P": "POS_IONIZABLE",
        "N": "NEG_IONIZABLE",
    }
    features = []
    with open(pdb_file) as fh:
        for line in fh:
            if not line.startswith("ATOM"):
                continue
            atom_name = line[12:16].strip()
            char  = atom_name[0] if atom_name else "?"
            ftype = _CHAR_TO_TYPE.get(char)
            if ftype is None:
                continue
            try:
                x   = float(line[30:38])
                y   = float(line[38:46])
                z   = float(line[46:54])
                occ = float(line[54:60]) if len(line) > 60 else 1.0
                bf  = float(line[60:66]) if len(line) > 66 else 0.0
            except ValueError:
                continue
            features.append({
                "type"    : ftype,
                "coords"  : (x, y, z),
                "source"  : source_label,
                "bfactor" : bf,
                "weight"  : max(1, round(occ * 3)),
                "label"   : f"{ftype} ({source_label})",
            })
    return features
