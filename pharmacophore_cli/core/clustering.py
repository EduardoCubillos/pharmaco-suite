"""
clustering.py — Agrupación de features farmacofóricos.

Radios de clustering por tipo de feature basados en literatura:
  - Wolber & Langer (2005) J. Chem. Inf. Model. 45, 160–169
  - Günther et al. (2008) J. Chem. Inf. Model. 48, 2022–2039
  - Dixon et al. (2006) J. Comput.-Aided Mol. Des. 20, 647–671

Principio: los puentes de hidrógeno son direccionales (radio pequeño ~1.5 Å),
mientras que los contactos hidrofóbicos son no-direccionales (radio mayor ~2.5 Å).
Usar el mismo radio para todos los tipos sobremerge H-bonds y submerge hidrofóbicos.
"""

import numpy as np
from collections import defaultdict
from typing import List, Dict, Any

# ── Radios de clustering por tipo (Å) ─────────────────────────────────────────
# Referencia principal: Wolber & Langer (2005); Koes & Camacho (2012)
CLUSTER_RADII: Dict[str, float] = {
    "DONOR"         : 1.5,   # H-bond: direccional, radio reducido
    "ACCEPTOR"      : 1.5,   # H-bond: direccional, radio reducido
    "HYDROPHOBIC"   : 2.5,   # no-direccional, permite más holgura
    "POS_IONIZABLE" : 2.0,   # interacción iónica: semidireccional
    "NEG_IONIZABLE" : 2.0,
    "AROMATIC"      : 2.0,
}

DEFAULT_RADIUS = 2.0  # fallback para tipos no catalogados


def cluster_features(
    features: List[Dict[str, Any]],
    radii: Dict[str, float] = None,
    default_radius: float = DEFAULT_RADIUS,
) -> List[Dict[str, Any]]:
    """
    Agrupa features farmacofóricos del mismo tipo que estén dentro del radio
    correspondiente. Devuelve una lista con los centroides.

    Algoritmo: greedy single-linkage por tipo (rápido y reproducible).
    Cada feature se une al primer cluster cuyo centro esté dentro del radio;
    si no hay ninguno, abre un nuevo cluster.

    Parameters
    ----------
    features : lista de dicts con 'type' y 'coords' (tuple de 3 floats)
    radii    : dict tipo→radio en Å; si None usa CLUSTER_RADII globales
    default_radius : radio para tipos no en `radii`

    Returns
    -------
    Lista de features clusterizados con campos adicionales:
      'n_merged'  : cuántos features originales se fusionaron
      'clustered' : True
    """
    if radii is None:
        radii = CLUSTER_RADII

    # Agrupar por tipo
    by_type: Dict[str, List] = defaultdict(list)
    for f in features:
        by_type[f["type"]].append(f)

    result = []
    for ftype, group in by_type.items():
        radius = radii.get(ftype, default_radius)
        clusters = _greedy_cluster(group, radius)
        for cluster in clusters:
            coords_arr = np.array([np.array(f["coords"], dtype=float) for f in cluster])
            centroid = tuple(coords_arr.mean(axis=0))
            # Heredar fuente si todos son iguales
            sources = set(f.get("source", "?") for f in cluster)
            source = sources.pop() if len(sources) == 1 else "mixed"
            result.append({
                "type"      : ftype,
                "coords"    : centroid,
                "source"    : source,
                "n_merged"  : len(cluster),
                "clustered" : True,
                "label"     : f"{ftype} [cluster {len(cluster)} pts, r={radius}Å]",
            })

    return result


def _greedy_cluster(
    features: List[Dict[str, Any]], radius: float
) -> List[List[Dict[str, Any]]]:
    """
    Greedy clustering: itera los features en orden y asigna cada uno
    al primer cluster existente cuyo centroide está dentro del radio.
    Si ninguno aplica, crea un cluster nuevo.
    """
    clusters: List[List[Dict]] = []
    centroids: List[np.ndarray] = []

    for f in features:
        fcoord = np.array(f["coords"], dtype=float)
        assigned = False
        for k, ctr in enumerate(centroids):
            if np.linalg.norm(fcoord - ctr) < radius:
                clusters[k].append(f)
                # Actualizar centroide incrementalmente
                n = len(clusters[k])
                centroids[k] = ctr + (fcoord - ctr) / n
                assigned = True
                break
        if not assigned:
            clusters.append([f])
            centroids.append(fcoord.copy())

    return clusters
