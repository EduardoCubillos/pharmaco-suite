"""
clustering.py — Pharmacophoric feature clustering.

Groups nearby features of the same type into a single centroid.

The radii below are type-specific rather than uniform, because a hydrogen
bond is directional and tolerates little positional slack while a
hydrophobic contact is diffuse and tolerates several ångström. A single
radius for both over-merges H-bonds and under-merges hydrophobic contacts.
The specific values are defaults of this tool, not numbers taken from any
published method, and they are meant to be overridden per run from the CLI.

Background on how pharmacophore tools define feature tolerances:
  - Wolber & Langer (2005) J. Chem. Inf. Model. 45, 160–169 — LigandScout
  - Dixon et al. (2006) J. Comput.-Aided Mol. Des. 20, 647–671 — PHASE
"""

import numpy as np
from collections import defaultdict
from typing import List, Dict, Any

# Default clustering radii per feature type (Å) — this tool's own defaults
CLUSTER_RADII: Dict[str, float] = {
    "DONOR"         : 1.5,   # directional — tight
    "ACCEPTOR"      : 1.5,   # directional — tight
    "HYDROPHOBIC"   : 2.5,   # non-directional — loose
    "POS_IONIZABLE" : 2.0,
    "NEG_IONIZABLE" : 2.0,
    "AROMATIC"      : 2.0,
}

_FALLBACK_RADIUS = 2.0  # for any type not in the table above


def cluster_features(
    features: List[Dict[str, Any]],
    radii: Dict[str, float] = None,
    fallback_radius: float = _FALLBACK_RADIUS,
) -> List[Dict[str, Any]]:
    """
    Merge pharmacophoric features of the same type that fall within the
    type-specific clustering radius. Returns the list of centroids.

    Algorithm: greedy clustering per feature type. Each incoming feature
    joins the *nearest* existing cluster whose centroid lies within the
    radius; if none qualifies, it starts a new cluster. Fast, reproducible
    and good enough for pharmacophore densities, but the result still
    depends on input order, so features are sorted canonically first.

    Parameters
    ----------
    features        : list of dicts with at least 'type' and 'coords' (3-tuple)
    radii           : override dict {feature_type: radius_Å}; defaults to CLUSTER_RADII
    fallback_radius : radius for types not in `radii`

    Returns
    -------
    List of clustered features with added fields:
      'n_merged'  : how many raw features were merged into this centroid
      'clustered' : True
    """
    if radii is None:
        radii = CLUSTER_RADII

    grouped: Dict[str, List] = defaultdict(list)
    for f in features:
        grouped[f["type"]].append(f)

    out = []
    for ftype, group in grouped.items():
        r = radii.get(ftype, fallback_radius)
        for cluster in _assign_to_clusters(group, r):
            coords_arr = np.array([np.array(f["coords"], dtype=float) for f in cluster])
            centroid = tuple(coords_arr.mean(axis=0))
            sources = {f.get("source", "?") for f in cluster}
            source = sources.pop() if len(sources) == 1 else "mixed"
            frames = {f.get("frame", "receptor") for f in cluster}
            out.append({
                "type"      : ftype,
                "coords"    : centroid,
                "source"    : source,
                "frame"     : frames.pop() if len(frames) == 1 else "mixed",
                "n_merged"  : len(cluster),
                "clustered" : True,
                "label"     : f"{ftype} [{len(cluster)} pts merged, r={r}Å]",
            })

    return out


def _assign_to_clusters(
    features: List[Dict[str, Any]], radius: float
) -> List[List[Dict[str, Any]]]:
    """
    Greedy clustering: iterate features and assign each to the nearest
    existing cluster within `radius`. Start a new cluster if none qualifies.
    Centroids are updated incrementally as features are added.

    Picking the nearest candidate rather than the first one encountered
    keeps the assignment independent of how the clusters happen to be
    ordered in memory. Features are visited in a canonical coordinate
    order so that two runs on the same input always return the same
    clusters, whatever order the upstream model emitted them in.
    """
    clusters: List[List[Dict]] = []
    centroids: List[np.ndarray] = []

    ordered = sorted(features, key=lambda f: tuple(round(float(c), 4)
                                                   for c in f["coords"]))

    for f in ordered:
        fc = np.array(f["coords"], dtype=float)
        best_k, best_d = None, radius
        for k, ctr in enumerate(centroids):
            d = float(np.linalg.norm(fc - ctr))
            if d < best_d:
                best_k, best_d = k, d
        if best_k is None:
            clusters.append([f])
            centroids.append(fc.copy())
        else:
            clusters[best_k].append(f)
            n = len(clusters[best_k])
            centroids[best_k] = centroids[best_k] + (fc - centroids[best_k]) / n

    return clusters
