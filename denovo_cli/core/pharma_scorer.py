"""
pharma_scorer.py — Puntuación farmacofórica contra modelos SBP/LBP/consenso.

Dos modos:
  1. Feature-based : cuenta qué fracción de features del modelo están
                     químicamente presentes en la molécula candidata. Ignora
                     la geometría: dice qué grupos hay, no dónde quedan.
  2. 3D fit        : carga un PDB de farmacóforo generado por pharmacophore_cli
                     y resuelve si la molécula puede adoptar una conformación
                     que coloque sus grupos sobre los puntos del modelo
                     (constrained embedding + superposición). Invariante a
                     rotaciones y traslaciones de cualquiera de las dos
                     entradas, porque solo usa geometría interna.

Referencias:
  - Dixon et al. (2006) J. Comput.-Aided Mol. Des. 20, 647–671 (PHASE)
  - Wolber et al. (2006) J. Comput.-Aided Mol. Des. 20, 773–788 (LigandScout)
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

    # Triangles of model points, non-degenerate and ordered by spread: a wide
    # triangle pins the superposition down, a nearly collinear one does not
    triples = _model_triples(p_dist, tolerance)
    if not triples:
        return _out(0.0, n_points_total=n_total, reason="model_degenerate")

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

        for (i, j, k) in triples:
            if best_w >= w_reachable:
                break
            cand_i = by_type.get(p_type[i], [])
            cand_j = by_type.get(p_type[j], [])
            cand_k = by_type.get(p_type[k], [])
            if not (cand_i and cand_j and cand_k):
                continue

            for a in cand_i:
                for b in cand_j:
                    if b == a or abs(m_dist[a, b] - p_dist[i, j]) > 2 * tolerance:
                        continue
                    for c in cand_k:
                        if c in (a, b):
                            continue
                        if abs(m_dist[a, c] - p_dist[i, k]) > 2 * tolerance:
                            continue
                        if abs(m_dist[b, c] - p_dist[j, k]) > 2 * tolerance:
                            continue

                        trials += 1
                        if trials > max_trials:
                            return _finish(_out, best_w, w_total, best_n,
                                           n_total, best_rmsd, min_points,
                                           truncated=True)

                        R, t = _kabsch(m_xyz[[a, b, c]], p_xyz[[i, j, k]])
                        moved = (m_xyz @ R.T) + t
                        w, n, rmsd = _coverage(moved, mol_feats, p_xyz,
                                               p_type, p_w, tolerance)
                        if n >= min_points and (w > best_w or
                                                (w == best_w and rmsd is not None
                                                 and (best_rmsd is None or rmsd < best_rmsd))):
                            best_w, best_n, best_rmsd = w, n, rmsd

    return _finish(_out, best_w, w_total, best_n, n_total, best_rmsd, min_points)


# ── Internals del fit 3D ──────────────────────────────────────────────────────

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


def _model_triples(p_dist: np.ndarray, tolerance: float,
                   min_side: float = 2.0) -> List[tuple]:
    """
    Triples of model points that define a usable superposition: sides long
    enough not to be degenerate, and not collinear. Wider triangles first,
    since those constrain the rotation best and reach the right answer with
    fewer trials.
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
                if area < tolerance ** 2:        # too close to collinear
                    continue
                triples.append((area, (i, j, k)))
    triples.sort(key=lambda x: -x[0])
    return [t for _, t in triples]


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
