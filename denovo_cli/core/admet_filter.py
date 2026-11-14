"""
admet_filter.py — Filtro ADMET computacional con descriptores RDKit.

Calcula y filtra por:
  - Lipinski Rule of 5 (absorción oral)
  - QED — Quantitative Estimate of Druglikeness
  - SA Score — Synthetic Accessibility
  - PSA — Polar Surface Area
  - LogP
  - Alertas estructurales (catálogos PAINS/BRENK/NIH de RDKit)

Referencias:
  - Lipinski et al. (2001) Adv. Drug Deliv. Rev. 46, 3–26
  - Bickerton et al. (2012) Nature Chemistry 4, 90–98
  - Ertl & Schuffenhauer (2009) J. Cheminform. 1, 8
  - Baell & Holloway (2010) J. Med. Chem. 53, 2719–2740 (PAINS)
  - Brenk et al. (2008) ChemMedChem 3, 435–444
"""

import os
import sys
import numpy as np
from typing import List, Dict, Any, Optional

# ── Defaults de filtro ────────────────────────────────────────────────────────
DEFAULT_FILTERS = {
    "mw_max"     : 600,    # Da  — un poco más holgado que Lipinski estricto
    "logp_max"   : 5.5,
    "hbd_max"    : 5,
    "hba_max"    : 10,
    "psa_max"    : 140,    # Å²
    "rotbonds_max": 10,
    "qed_min"    : 0.30,
    "sa_max"     : 5.0,    # 1=fácil, 10=difícil; ≤5 es accesible
    "lip_violations_max": 1,   # tolerancia de 1 violación Ro5
    # Catálogos que descartan la molécula. PAINS son los agregadores y
    # promiscuos de Baell & Holloway; BRENK y NIH son más agresivos y se
    # reportan por separado sin descartar, para no perder quimiotipos
    # perfectamente válidos.
    "reject_catalogs": ("PAINS",),
    "report_catalogs": ("BRENK", "NIH"),
}


def compute_admet(mol, filters: Dict = None) -> Dict[str, Any]:
    """
    Calcula todos los descriptores ADMET para una molécula.
    Devuelve dict con métricas y un flag 'passes' global.

    `filters` sobreescribe DEFAULT_FILTERS. Si se omite, se usan los
    defaults — pero quien pase filtros propios los verá aplicados tanto a
    los umbrales numéricos como a los catálogos de alertas, que antes
    quedaban fijos en los defaults aunque el llamador pidiera otra cosa.
    """
    from rdkit.Chem import Descriptors, QED as QEDmod, rdMolDescriptors, RDConfig
    from rdkit import Chem

    if mol is None:
        return {"passes": False, "error": "mol_none"}

    active = {**DEFAULT_FILTERS, **(filters or {})}

    mw      = Descriptors.MolWt(mol)
    logp    = Descriptors.MolLogP(mol)
    hbd     = rdMolDescriptors.CalcNumHBD(mol)
    hba     = rdMolDescriptors.CalcNumHBA(mol)
    psa     = rdMolDescriptors.CalcTPSA(mol)
    rotb    = rdMolDescriptors.CalcNumRotatableBonds(mol)
    arom    = rdMolDescriptors.CalcNumAromaticRings(mol)
    heavy   = mol.GetNumHeavyAtoms()

    # QED
    try:
        qed = QEDmod.qed(mol)
    except Exception:
        qed = 0.0

    # SA Score
    sa = _compute_sa(mol)

    # Lipinski violations
    lip_v = sum([mw > 500, logp > 5, hbd > 5, hba > 10])

    # Alertas estructurales: catálogos curados de RDKit
    alerts      = _check_structural_alerts(mol, active.get("reject_catalogs", ()))
    alerts_info = _check_structural_alerts(mol, active.get("report_catalogs", ()))

    result = {
        "mw"          : round(mw, 2),
        "logp"        : round(logp, 2),
        "hbd"         : hbd,
        "hba"         : hba,
        "psa"         : round(psa, 1),
        "rot_bonds"   : rotb,
        "arom_rings"  : arom,
        "heavy_atoms" : heavy,
        "qed"         : round(qed, 3),
        "sa_score"    : round(sa, 2) if sa else None,
        "lip_violations": lip_v,
        "alerts"      : alerts,
        "alerts_info" : alerts_info,
        "passes"      : False,   # se asignará abajo
    }

    result["passes"] = _passes_filters(result, active)
    return result


def filter_molecules(
    smiles_list: List[str],
    filters: Dict = None,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """
    Filtra una lista de SMILES por criterios ADMET.

    Returns
    -------
    Lista de dicts: {'smiles', 'mol', ...métricas ADMET..., 'passes'}
    Ordenada por QED descendente.
    """
    from rdkit import Chem
    _log = print if verbose else lambda *a, **k: None

    active_filters = {**DEFAULT_FILTERS, **(filters or {})}
    results = []
    n_invalid = 0

    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            n_invalid += 1
            continue
        canon = Chem.MolToSmiles(mol)
        admet = compute_admet(mol, active_filters)
        admet["smiles"] = canon
        results.append(admet)

    passing  = [r for r in results if r.get("passes")]
    failing  = [r for r in results if not r.get("passes")]
    passing.sort(key=lambda x: -x["qed"])

    _log(f"[ADMET] Evaluadas: {len(results)} | Pasan: {len(passing)} | "
         f"Fallan: {len(failing)} | Inválidas: {n_invalid}")

    return passing + failing


def admet_summary_table(results: List[Dict]) -> str:
    """Devuelve una tabla ASCII con las métricas ADMET de cada molécula."""
    header = f"{'#':>4}  {'SMILES':40s}  {'MW':>6}  {'LogP':>5}  {'PSA':>5}  "
    header += f"{'QED':>5}  {'SA':>4}  {'LipV':>4}  {'OK':>4}"
    lines = [header, "─" * len(header)]
    for i, r in enumerate(results[:25], 1):
        smi_short = r["smiles"][:38] + ".." if len(r["smiles"]) > 40 else r["smiles"]
        sa_str    = f"{r['sa_score']:.1f}" if r.get("sa_score") else "  ?"
        ok_str    = "✓" if r.get("passes") else "✗"
        lines.append(
            f"{i:>4}  {smi_short:40s}  {r['mw']:>6.1f}  {r['logp']:>5.2f}  "
            f"{r['psa']:>5.1f}  {r['qed']:>5.3f}  {sa_str:>4}  "
            f"{r['lip_violations']:>4}  {ok_str:>4}"
        )
    return "\n".join(lines)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _passes_filters(metrics: Dict, filters: Dict) -> bool:
    if metrics.get("error"):
        return False
    if metrics["mw"]          > filters.get("mw_max", 600):
        return False
    if metrics["logp"]        > filters.get("logp_max", 5.5):
        return False
    if metrics["hbd"]         > filters.get("hbd_max", 5):
        return False
    if metrics["hba"]         > filters.get("hba_max", 10):
        return False
    if metrics["psa"]         > filters.get("psa_max", 140):
        return False
    if metrics["rot_bonds"]   > filters.get("rotbonds_max", 10):
        return False
    if metrics["qed"]         < filters.get("qed_min", 0.30):
        return False
    if metrics.get("sa_score") and metrics["sa_score"] > filters.get("sa_max", 5.0):
        return False
    if metrics["lip_violations"] > filters.get("lip_violations_max", 1):
        return False
    if metrics.get("alerts"):
        return False
    return True


def _compute_sa(mol) -> Optional[float]:
    """SA Score desde RDKit contrib."""
    from rdkit.Chem import RDConfig
    sa_path = os.path.join(RDConfig.RDContribDir, "SA_Score")
    if sa_path not in sys.path:
        sys.path.append(sa_path)
    try:
        import sascorer
        return sascorer.calculateScore(mol)
    except Exception:
        return None


# ── Alertas estructurales ─────────────────────────────────────────────────────
#
# Los catálogos vienen de RDKit (FilterCatalog), no escritos a mano. Los seis
# SMARTS que había antes se llamaban PAINS sin serlo: una quinona para-
# sustituida, que es el quimiotipo PAINS de manual, no activaba ninguno de los
# seis, mientras que un pirrol libre — presente en quimiotipos perfectamente
# viables — descartaba la molécula de inmediato. El patrón alfa-amino además
# estaba escrito estereoespecífico ([C@@H]) sin efecto alguno, porque
# HasSubstructMatch ignora la quiralidad salvo que se pida explícitamente.

_catalog_cache: Dict[tuple, Any] = {}


def _get_catalog(names: tuple):
    """FilterCatalog para los catálogos pedidos; se construye una sola vez."""
    from rdkit.Chem import FilterCatalog
    from rdkit.Chem.FilterCatalog import FilterCatalogParams

    key = tuple(sorted(names))
    if key in _catalog_cache:
        return _catalog_cache[key]

    params = FilterCatalogParams()
    for name in key:
        entry = getattr(FilterCatalogParams.FilterCatalogs, name.upper(), None)
        if entry is None:
            print(f"[ADMET] Catálogo desconocido, ignorado: {name}", file=sys.stderr)
            continue
        params.AddCatalog(entry)
    catalog = FilterCatalog.FilterCatalog(params)
    _catalog_cache[key] = catalog
    return catalog


def _check_structural_alerts(mol, catalogs: tuple = ("PAINS",)) -> List[str]:
    """
    Devuelve los nombres de las alertas encontradas (vacío si ninguna).
    Los nombres son los de RDKit, así que se pueden rastrear hasta la
    publicación original del filtro.
    """
    if mol is None or not catalogs:
        return []
    try:
        catalog = _get_catalog(tuple(catalogs))
    except Exception as e:
        print(f"[ADMET] FilterCatalog no disponible ({e}); sin alertas.",
              file=sys.stderr)
        return []

    try:
        return sorted({m.GetDescription() for m in catalog.GetMatches(mol)})
    except Exception:
        return []
