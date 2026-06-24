"""
clustering.py — Pharmacophoric feature clustering.

Groups nearby features of the same type into a single centroid.
Uses type-specific radii from literature, since hydrogen bonds are
directional (tight radius ~1.5 Å) while hydrophobics are not (~2.5 Å).
Collapsing everything to a single radius over-merges H-bonds and
under-merges hydrophobic contacts.

References:
  - Wolber & Langer (2005) J. Chem. Inf. Model. 45, 160–169
  - Günther et al. (2008) J. Chem. Inf. Model. 48, 2022–2039
  - Dixon et al. (2006) J. Comput.-Aided Mol. Des. 20, 647–671
"""

import numpy as np
from collections import defaultdict
from typing import List, Dict, Any

# Default clustering radii per feature type (Å)
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

    Algorithm: greedy single-linkage per feature type. Each incoming
    feature is assigned to the first existing cluster whose centroid is
    within the radius; if none qualifies, it starts a new cluster.
    Fast, reproducible, and good enough for pharmacophore densities.

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
            out.append({
                "type"      : ftype,
                "coords"    : centroid,
                "source"    : source,
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
    """
    clusters: List[List[Dict]] = []
    centroids: List[np.ndarray] = []

    for f in features:
        fc = np.array(f["coords"], dtype=float)
        placed = False
        for k, ctr in enumerate(centroids):
            if np.linalg.norm(fc - ctr) < radius:
                clusters[k].append(f)
                n = len(clusters[k])
                centroids[k] = ctr + (fc - ctr) / n   # running mean
                placed = True
                break
        if not placed:
            clusters.append([f])
            centroids.append(fc.copy())

    return clusters
