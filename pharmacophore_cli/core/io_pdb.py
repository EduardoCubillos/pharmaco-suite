"""
io_pdb.py — Escritura y lectura de modelos farmacofóricos en formato PDB.

Convención de átomos:
  A = ACCEPTOR
  D = DONOR
  H = HYDROPHOBIC
  P = POS_IONIZABLE
  N = NEG_IONIZABLE

Columna B-factor: nivel de consenso (33.3 / 66.7 / 100.0)
Columna Occupancy: fracción de modelos (0.33 / 0.67 / 1.00)
"""

from typing import List, Dict, Any

TYPE_CHAR   = {"ACCEPTOR": "A", "DONOR": "D", "HYDROPHOBIC": "H",
               "POS_IONIZABLE": "P", "NEG_IONIZABLE": "N"}
TYPE_RESNUM = {"ACCEPTOR": 1,   "DONOR": 2,   "HYDROPHOBIC": 3,
               "POS_IONIZABLE": 4, "NEG_IONIZABLE": 5}


def write_pharmacophore_pdb(
    features: List[Dict[str, Any]],
    filename: str,
    title: str = "Farmacóforo",
    radius: float = None,
    is_consensus: bool = False,
) -> str:
    """
    Escribe los features en un archivo PDB.

    Para features simples (SBP/LBP/RBP): occupancy=1.00, bfactor=0.0
    Para features consenso: occupancy=weight/3, bfactor=33.3*weight

    Returns
    -------
    Ruta del archivo escrito.
    """
    lines = [
        f"REMARK  {title}",
        f"REMARK  Generado por pharmacophore_tool",
        f"REMARK  Formato: A=Acceptor D=Donor H=Hydrophobic P=PosIon N=NegIon",
    ]
    if radius is not None:
        lines.append(f"REMARK  Radio de clustering: {radius} A")
    if is_consensus:
        lines += [
            "REMARK  Occupancy: 0.33=un modelo  0.67=dos modelos  1.00=triple consenso",
            "REMARK  B-factor : 33.3=un modelo  66.7=dos modelos  100.0=triple consenso",
        ]

    for idx, feat in enumerate(features, 1):
        x, y, z = feat["coords"]
        char    = TYPE_CHAR.get(feat["type"], "X")
        resnum  = TYPE_RESNUM.get(feat["type"], 9)

        if is_consensus:
            weight    = feat.get("weight", 1)
            occupancy = round(weight / 3, 2)
            bfactor   = feat.get("bfactor", round(weight / 3 * 100, 1))
        else:
            occupancy = 1.00
            bfactor   = 0.0

        line = (
            f"ATOM  {idx:5d}  {char:<3s} PH4 A{resnum:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}"
            f"{occupancy:6.2f}{bfactor:6.1f}          {char:>2s}"
        )
        lines.append(line)

        # REMARK con metadatos por feature
        if is_consensus:
            models_s = "+".join(sorted(feat.get("models", {feat.get("source", "?")})))
            rbp_comp = feat.get("rbp_complement", "—")
            lines.append(
                f"REMARK  [{idx:3d}] {feat['type']:14s} "
                f"RBP_complemento={rbp_comp:35s} modelos={models_s}"
            )
        else:
            lines.append(f"REMARK  [{idx:3d}] {feat['type']:14s} {feat.get('label','')}")

    lines.append("END")

    with open(filename, "w") as fh:
        fh.write("\n".join(lines) + "\n")

    return filename
