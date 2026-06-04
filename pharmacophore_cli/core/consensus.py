"""
consensus.py — Modelo Farmacofórico Consenso (SBP ∩ LBP ∩ RBP).

Combina los tres modelos usando complementariedad química receptor↔ligando:
  DONOR (ligando)       ↔  ACCEPTOR (receptor)
  ACCEPTOR (ligando)    ↔  DONOR (receptor)
  HYDROPHOBIC           ↔  HYDROPHOBIC
  POS_IONIZABLE         ↔  NEG_IONIZABLE
  NEG_IONIZABLE         ↔  POS_IONIZABLE

Peso del feature (1–3) según cuántos modelos lo respaldan.
Features con peso ≥ 2 (doble o triple consenso) son los más robustos
para cribado virtual.

Referencia:
  - Baroni et al. (2007) J. Chem. Inf. Model. 47, 279–294 (PHASE)
  - Sanders et al. (2012) J. Chem. Inf. Model. 52, 1261–1272
"""

import numpy as np
from typing import List, Dict, Any
from collections import defaultdict

# Complementariedad química receptor ↔ ligando
COMPLEMENT_OF: Dict[str, str] = {
    "DONOR"         : "ACCEPTOR",
    "ACCEPTOR"      : "DONOR",
    "HYDROPHOBIC"   : "HYDROPHOBIC",
    "POS_IONIZABLE" : "NEG_IONIZABLE",
    "NEG_IONIZABLE" : "POS_IONIZABLE",
}

# Radio por defecto para fusionar features entre modelos
CONSENSUS_RADIUS_DEFAULT = 3.5   # Å — estándar en literatura (Baroni 2007)


def build_consensus(
    sbp: List[Dict[str, Any]],
    lbp: List[Dict[str, Any]],
    rbp: List[Dict[str, Any]],
    radius: float = CONSENSUS_RADIUS_DEFAULT,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """
    Construye el modelo consenso combinando SBP + LBP + RBP.

    Algoritmo:
    1. Fusionar SBP y LBP (perspectiva ligando) por tipo + distancia.
    2. Para cada feature ligando, buscar en RBP el tipo complementario.
    3. Asignar peso 1/2/3 según modelos contribuyentes.

    Returns
    -------
    Lista de features consenso con:
      'type'           : tipo del feature (perspectiva del ligando)
      'coords'         : centroide
      'weight'         : 1, 2 o 3
      'models'         : set de modelos {'SBP', 'LBP', 'RBP'}
      'bfactor'        : 33.3 / 66.7 / 100.0 (para PDB)
      'rbp_complement' : descripción del feature RBP que lo respalda
    """
    _log = print if verbose else lambda *a, **k: None

    # Pre-etiquetar
    sbp_t = [{**f, "source": "SBP"} for f in sbp]
    lbp_t = [{**f, "source": "LBP"} for f in lbp]
    rbp_t = [{**f, "source": "RBP"} for f in rbp]

    ligand_feats = sbp_t + lbp_t
    consensus    = []
    processed    = set()

    for i, fi in enumerate(ligand_feats):
        if i in processed:
            continue

        cluster    = [fi]
        sources    = {fi["source"]}
        processed.add(i)
        ci         = np.array(fi["coords"], dtype=float)
        ftype      = fi["type"]

        # Buscar features del mismo tipo (ligando) cercanos
        for j, fj in enumerate(ligand_feats):
            if j == i or j in processed:
                continue
            if fj["type"] != ftype:
                continue
            if np.linalg.norm(ci - np.array(fj["coords"], dtype=float)) < radius:
                cluster.append(fj)
                sources.add(fj["source"])
                processed.add(j)

        centroid = np.mean([np.array(f["coords"], dtype=float) for f in cluster], axis=0)

        # Buscar complemento en RBP
        comp_type    = COMPLEMENT_OF.get(ftype, ftype)
        rbp_match    = None
        rbp_dist_min = float("inf")

        for fr in rbp_t:
            if fr["type"] != comp_type:
                continue
            d = np.linalg.norm(centroid - np.array(fr["coords"], dtype=float))
            if d < radius and d < rbp_dist_min:
                rbp_dist_min = d
                rbp_match    = fr

        rbp_label = "—"
        if rbp_match:
            sources.add("RBP")
            rbp_label = f"{comp_type} (d={rbp_dist_min:.2f}Å, {rbp_match.get('residue','?')})"

        weight  = len(sources)
        bfactor = round(weight / 3 * 100, 1)

        consensus.append({
            "type"           : ftype,
            "coords"         : tuple(centroid),
            "models"         : sources,
            "weight"         : weight,
            "bfactor"        : bfactor,
            "rbp_complement" : rbp_label,
            "n_atoms"        : len(cluster),
            "label"          : f"{ftype} [{'+'.join(sorted(sources))}] w={weight}",
        })

    consensus.sort(key=lambda x: (-x["weight"], x["type"]))

    triple = [f for f in consensus if f["weight"] == 3]
    double = [f for f in consensus if f["weight"] == 2]
    single = [f for f in consensus if f["weight"] == 1]

    _log(f"\n{'='*60}")
    _log("MODELO FARMACOFORO CONSENSO")
    _log(f"  Radio de clustering: {radius} Å")
    _log(f"{'='*60}")
    _log(f"  Triple-consenso (SBP+LBP+RBP): {len(triple):3d} features")
    _log(f"  Doble-consenso               : {len(double):3d} features")
    _log(f"  Único modelo                 : {len(single):3d} features")
    _log(f"  TOTAL                        : {len(consensus):3d} features")

    if triple:
        _log("\nFeatures triple-consenso:")
        _log(f"  {'#':>3}  {'Tipo ligando':16s}  {'Complemento RBP':35s}  {'Modelos'}")
        _log(f"  {'─'*70}")
        for i, feat in enumerate(triple):
            _log(f"  [{i+1:02d}] {feat['type']:16s}  {feat['rbp_complement']:35s}  "
                 f"{'+'.join(sorted(feat['models']))}")

    return consensus


def load_features_from_pdb(pdb_file: str, source_label: str = "LOADED") -> List[Dict[str, Any]]:
    """
    Carga features farmacofóricos desde un PDB generado por esta herramienta.

    El formato esperado es el de write_pharmacophore_pdb:
      ATOM record con nombre de átomo = primera letra del tipo (A/D/H/P/N)
      Columna occupancy = peso/3 (para consenso) o 1.0 (para SBP/LBP/RBP)
    """
    TYPE_FROM_CHAR = {
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
            char = atom_name[0] if atom_name else "?"
            ftype = TYPE_FROM_CHAR.get(char)
            if ftype is None:
                continue
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
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
