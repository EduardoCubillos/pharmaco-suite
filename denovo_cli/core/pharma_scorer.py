"""
pharma_scorer.py — Pharmacophore scoring against SBP/LBP/consensus models.

Two modes:
  1. Feature-based : what fraction of the model's features are chemically
                     present in the candidate. Ignores geometry — it reports
                     which groups exist, not where they sit.
  2. 3D fit        : loads a pharmacophore PDB written by pharmacophore_cli
                     and asks whether the molecule can adopt a conformation
                     that places its groups on the model's points, by
                     aligning a conformer ensemble onto them. Invariant to
                     rotation and translation of either input, since only
                     internal geometry enters the calculation.

References:
  - Dixon et al. (2006) J. Comput.-Aided Mol. Des. 20, 647–671 — PHASE
  - Wolber, Dornhofer & Langer (2006) J. Comput.-Aided Mol. Des. 20, 773–788
    — overlay of small organic molecules onto 3D pharmacophores
"""

import os
import sys
import numpy as np
from typing import List, Dict, Any, Optional

# RDKit feature families mapped to the canonical types used across the suite
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
    """Count a molecule's pharmacophoric features with RDKit."""
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
    Build the reference feature profile as the union of the features
    present in at least half of the ligands. Used for multi-ligand LBP.
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

    # Count in how many molecules each feature type appears
    type_counts = Counter()
    for counts in all_counts:
        for ftype, n_feat in counts.items():
            type_counts[ftype] += 1

    profile = {ftype: 1 for ftype, freq in type_counts.items() if freq >= min_freq}
    return profile


def pharmacophore_score(mol, reference: Dict[str, int]) -> float:
    """
    Fraction of the reference profile's features present in mol.
    0.0 = none present | 1.0 = all present.
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
    Load features from a PDB written by pharmacophore_cli.
    Format: ATOM records named A/D/H/P/N, x/y/z coordinates, occupancy = weight/3.
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
    tolerance: float = 1.5,
    max_points: int = 10,
    n_conformers: int = 20,
    min_points: int = 3,
    random_seed: int = 42,
    max_trials: int = 20000,
    return_details: bool = False,
):
    """
    Pharmacophore fit score: how much of the model can this molecule hit at
    once, and how well?

    The pharmacophore lives in the receptor's coordinate frame while a freshly
    generated candidate lives in whatever frame its embedding produced, so
    comparing the two coordinate sets as they stand is meaningless — the
    answer would depend on where the crystal happened to sit in its unit cell.
    The candidate is therefore *aligned* onto the model before anything is
    measured, exactly as in pharmacophore-based screening:

    1. A conformer ensemble is generated (`n_conformers`, ETKDGv3 + MMFF),
       because a single conformer says nothing about what geometries the
       molecule can reach.
    2. For each conformer, candidate alignments are proposed from triples of
       pharmacophore points and triples of same-type molecule features whose
       internal distances agree within `tolerance`. Each triple defines a
       rigid superposition (Kabsch).
    3. Under each superposition every model point is matched to the nearest
       unused molecule feature of its own type within `tolerance`, and the
       weighted fraction of points matched is the coverage.

    The best coverage over the whole ensemble is the score, so the result
    depends only on internal geometry and is invariant to any rotation or
    translation of either input. Partial matches arise naturally: a molecule
    that hits five of eight points scores accordingly instead of failing.

    Parameters
    ----------
    mol            : RDKit molecule; input 3D coordinates are ignored
    features       : model points, each {'type', 'coords', 'weight'}
    tolerance      : radius (Å) of the tolerance sphere around each point
    max_points     : keep only the `max_points` highest-weight model points
    n_conformers   : size of the conformer ensemble
    min_points     : a fit below this many matched points scores 0
    random_seed    : conformer generation seed, for reproducibility
    max_trials     : cap on superpositions evaluated, as a runtime guard
    return_details : also return the diagnostic dict

    Returns
    -------
    float in [0, 1], or (float, dict) when `return_details` is set. The dict
    carries 'n_points_matched', 'n_points_total', 'fit_rmsd' (RMSD over the
    matched points of the winning alignment) and 'reason'.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    def _out(score, **kw):
        details = {"n_points_matched": 0, "n_points_total": 0,
                   "fit_rmsd": None, "reason": "ok", **kw}
        return (score, details) if return_details else score

    if mol is None or not features:
        return _out(0.0, reason="no_input")

    # Highest-weight points first: the consensus model's own confidence
    # ordering decides what gets kept when the model is larger than max_points
    points   = sorted(features, key=lambda f: -f.get("weight", 1))[:max_points]
    n_total  = len(points)
    w_total  = sum(f.get("weight", 1) for f in points)
    if w_total <= 0 or n_total < min_points:
        return _out(0.0, n_points_total=n_total, reason="model_too_small")

    p_xyz   = np.array([[float(c) for c in p["coords"]] for p in points])
    p_type  = [p["type"] for p in points]
    p_w     = np.array([float(p.get("weight", 1)) for p in points])
    p_dist  = np.linalg.norm(p_xyz[:, None, :] - p_xyz[None, :, :], axis=-1)

    confs = _conformer_ensemble(mol, n_conformers, random_seed)
    if not confs:
        return _out(0.0, n_points_total=n_total, reason="no_conformer")

    # Triangles of model points, ordered by spread: a wide triangle pins the
    # superposition down, a thin one leaves the rotation about its long axis
    # poorly determined. A model whose points are essentially collinear — a
    # small planar ligand's features often are — yields no usable triangle at
    # all, and is then fitted from pairs with the rotation about the pair's
    # axis sampled, which costs nothing when the model has no off-axis
    # structure for that rotation to disturb.
    triples = _model_triples(p_dist)
    proposals = [("triple", t) for t in triples]
    if not triples:
        pairs = _model_pairs(p_dist)
        if not pairs:
            return _out(0.0, n_points_total=n_total, reason="model_degenerate")
        proposals = [("pair", p) for p in pairs]

    # Upper bound on reachable weight: a model point can only ever be matched
    # by a molecule feature of its own type, so a molecule carrying two
    # acceptors can satisfy at most the two highest-weight acceptor points
    # however it is oriented. Reaching this bound means the search is done.
    w_reachable = _reachable_weight(points, confs[0])

    best_w, best_n, best_rmsd = 0.0, 0, None
    trials = 0

    for mol_feats in confs:
        if best_w >= w_reachable:
            break          # nothing left to gain
        mol_feats = _dedupe_features(mol_feats)
        by_type: Dict[str, List[int]] = {}
        for idx, (ftype, _) in enumerate(mol_feats):
            by_type.setdefault(ftype, []).append(idx)
        m_xyz = np.array([pos for _, pos in mol_feats])
        if len(m_xyz) < min_points:
            continue
        m_dist = np.linalg.norm(m_xyz[:, None, :] - m_xyz[None, :, :], axis=-1)

        for kind, anchor in proposals:
            if best_w >= w_reachable:
                break
            cands = [by_type.get(p_type[idx], []) for idx in anchor]
            if not all(cands):
                continue

            for combo in _combinations(cands, anchor, m_dist, p_dist, tolerance):
                trials += 1
                if trials > max_trials:
                    return _finish(_out, best_w, w_total, best_n,
                                   n_total, best_rmsd, min_points,
                                   truncated=True)

                for R, t in _superpositions(m_xyz, p_xyz, combo, anchor, kind):
                    moved = (m_xyz @ R.T) + t
                    w, n, rmsd = _coverage(moved, mol_feats, p_xyz,
                                           p_type, p_w, tolerance)
                    if n >= min_points and (w > best_w or
                                            (w == best_w and rmsd is not None
                                             and (best_rmsd is None or rmsd < best_rmsd))):
                        best_w, best_n, best_rmsd = w, n, rmsd

    return _finish(_out, best_w, w_total, best_n, n_total, best_rmsd, min_points)


# ── 3D fit internals ──────────────────────────────────────────────────────────

def _finish(_out, best_w, w_total, best_n, n_total, best_rmsd, min_points,
            truncated: bool = False):
    if best_n < min_points:
        return _out(0.0, n_points_total=n_total,
                    reason="truncated" if truncated else "no_alignment")
    return _out(best_w / w_total,
                n_points_matched=best_n,
                n_points_total=n_total,
                fit_rmsd=round(best_rmsd, 3) if best_rmsd is not None else None,
                reason="truncated" if truncated else "ok")


def _reachable_weight(points: List[Dict], mol_feats: List) -> float:
    """
    Largest model weight any orientation could possibly match, given only how
    many features of each type the molecule has. Used to stop the alignment
    search once it has nothing left to find.
    """
    avail: Dict[str, int] = {}
    for ftype, _ in _dedupe_features(mol_feats):
        avail[ftype] = avail.get(ftype, 0) + 1

    by_type: Dict[str, List[float]] = {}
    for p in points:
        by_type.setdefault(p["type"], []).append(float(p.get("weight", 1)))

    total = 0.0
    for ftype, weights in by_type.items():
        n = min(len(weights), avail.get(ftype, 0))
        total += sum(sorted(weights, reverse=True)[:n])
    return total


def _dedupe_features(mol_feats: List, min_sep: float = 0.5) -> List:
    """
    Collapse same-type features sitting essentially on top of each other.
    RDKit's definitions overlap — an aromatic ring shows up as both
    LumpedHydrophobe and Aromatic, both mapping to HYDROPHOBIC — and the
    duplicates only multiply the alignment search without adding geometry.
    """
    kept: List = []
    for ftype, pos in mol_feats:
        if any(t == ftype and float(np.linalg.norm(pos - p)) < min_sep
               for t, p in kept):
            continue
        kept.append((ftype, pos))
    return kept


def _conformer_ensemble(mol, n_conformers: int, seed: int) -> List[List]:
    """
    Generate a conformer ensemble and return, for each conformer, its
    pharmacophoric features as a list of (canonical_type, position).
    Hydrogens are added for the embedding and the geometry optimization,
    then the features are read off the heavy-atom skeleton.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    try:
        m = Chem.AddHs(Chem.Mol(mol))
    except Exception:
        return []

    params = AllChem.ETKDGv3()
    params.randomSeed  = seed
    params.pruneRmsThresh = 0.5
    try:
        cids = list(AllChem.EmbedMultipleConfs(m, numConfs=n_conformers,
                                               params=params))
    except Exception:
        cids = []
    if not cids:
        return []

    try:
        AllChem.MMFFOptimizeMoleculeConfs(m, maxIters=500)
    except Exception:
        pass

    try:
        factory_feats = _get_factory().GetFeaturesForMol(m)
    except Exception:
        return []

    # Feature atom ids are conformer independent, so they are read once
    spec = []
    for feat in factory_feats:
        ftype = FTYPE_MAP.get(feat.GetFamily())
        if ftype:
            spec.append((ftype, list(feat.GetAtomIds())))
    if not spec:
        return []

    out = []
    for cid in cids:
        conf = m.GetConformer(cid)
        pos  = np.array([list(conf.GetAtomPosition(i))
                         for i in range(m.GetNumAtoms())])
        out.append([(ftype, pos[ids].mean(axis=0)) for ftype, ids in spec])
    return out


def _model_triples(p_dist: np.ndarray, min_side: float = 2.0,
                   min_area: float = 1.0) -> List[tuple]:
    """
    Triples of model points that define a usable superposition: sides long
    enough not to be degenerate, and spanning enough area not to be
    collinear. Wider triangles first, since those constrain the rotation
    best and reach the right answer with fewer trials.
    """
    n = p_dist.shape[0]
    triples = []
    for i in range(n):
        for j in range(i + 1, n):
            if p_dist[i, j] < min_side:
                continue
            for k in range(j + 1, n):
                a, b, c = p_dist[i, j], p_dist[i, k], p_dist[j, k]
                if min(b, c) < min_side:
                    continue
                s = (a + b + c) / 2
                area2 = s * (s - a) * (s - b) * (s - c)
                if area2 <= 0:
                    continue
                area = float(np.sqrt(area2))
                if area < min_area:
                    continue
                triples.append((area, (i, j, k)))
    triples.sort(key=lambda x: -x[0])
    return [t for _, t in triples]


def _model_pairs(p_dist: np.ndarray, min_side: float = 2.0) -> List[tuple]:
    """
    Pairs of model points, widest first. Used only when the model has no
    usable triangle, which means its points are essentially collinear.
    """
    n = p_dist.shape[0]
    pairs = [(float(p_dist[i, j]), (i, j))
             for i in range(n) for j in range(i + 1, n)
             if p_dist[i, j] >= min_side]
    pairs.sort(key=lambda x: -x[0])
    return [p for _, p in pairs]


def _combinations(cands: List[List[int]], anchor: tuple, m_dist: np.ndarray,
                  p_dist: np.ndarray, tolerance: float):
    """
    Assignments of distinct molecule features to the anchor's model points,
    keeping only those whose internal distances agree with the model's to
    within 2 * tolerance (each endpoint may be off by up to one tolerance
    radius). Yields tuples of molecule feature indices.
    """
    k = len(anchor)

    def walk(chosen: List[int], depth: int):
        if depth == k:
            yield tuple(chosen)
            return
        for m in cands[depth]:
            if m in chosen:
                continue
            ok = True
            for prev, m_prev in enumerate(chosen):
                target = p_dist[anchor[prev], anchor[depth]]
                if abs(m_dist[m_prev, m] - target) > 2 * tolerance:
                    ok = False
                    break
            if ok:
                chosen.append(m)
                yield from walk(chosen, depth + 1)
                chosen.pop()

    yield from walk([], 0)


def _superpositions(m_xyz: np.ndarray, p_xyz: np.ndarray, combo: tuple,
                    anchor: tuple, kind: str, n_spins: int = 12):
    """
    Rigid transformations placing the chosen molecule features on the anchor's
    model points. A triple determines one; a pair leaves the rotation about
    the pair's axis free, so that rotation is sampled.
    """
    P = m_xyz[list(combo)]
    Q = p_xyz[list(anchor)]
    R, t = _kabsch(P, Q)
    if kind == "triple":
        yield R, t
        return

    axis = Q[1] - Q[0]
    norm = float(np.linalg.norm(axis))
    if norm < 1e-8:
        yield R, t
        return
    axis = axis / norm
    pivot = Q[0]
    for s in range(n_spins):
        spin = _rotation_about(axis, 2.0 * np.pi * s / n_spins)
        # x -> ((x @ R.T + t) - pivot) @ spin.T + pivot, rewritten in the
        # (R, t) convention used throughout: moved = x @ R.T + t
        yield spin @ R, (t - pivot) @ spin.T + pivot


def _rotation_about(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rotation matrix about a unit axis (Rodrigues)."""
    x, y, z = axis
    K = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


def _coverage(moved: np.ndarray, mol_feats: List, p_xyz: np.ndarray,
              p_type: List[str], p_w: np.ndarray, tolerance: float):
    """
    Under a given superposition, match each model point to the nearest unused
    molecule feature of its own type within `tolerance`. Points are served in
    order of how unambiguous their best match is, so a feature is not consumed
    by a point that had an alternative while another point had none.

    Returns (matched_weight, n_matched, rmsd_over_matched).
    """
    n_p = len(p_type)
    cand = []
    for pi in range(n_p):
        opts = [(float(np.linalg.norm(moved[mi] - p_xyz[pi])), mi)
                for mi, (ftype, _) in enumerate(mol_feats)
                if ftype == p_type[pi]]
        opts = sorted(d_m for d_m in opts if d_m[0] <= tolerance)
        cand.append(opts)

    order = sorted(range(n_p), key=lambda pi: (len(cand[pi]) or 10 ** 6,
                                               cand[pi][0][0] if cand[pi] else 10 ** 6))
    used  = set()
    w     = 0.0
    n     = 0
    sq    = 0.0
    for pi in order:
        for d, mi in cand[pi]:
            if mi in used:
                continue
            used.add(mi)
            w  += p_w[pi]
            n  += 1
            sq += d * d
            break

    rmsd = float(np.sqrt(sq / n)) if n else None
    return w, n, rmsd


def _kabsch(P: np.ndarray, Q: np.ndarray):
    """Rotation and translation minimizing RMSD(P, Q): P @ R.T + t -> Q."""
    p_mean, q_mean = P.mean(axis=0), Q.mean(axis=0)
    Pc, Qc = P - p_mean, Q - q_mean
    U, _, Vt = np.linalg.svd(Pc.T @ Qc)
    d = np.linalg.det(Vt.T @ U.T)
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    t = q_mean - p_mean @ R.T
    return R, t
