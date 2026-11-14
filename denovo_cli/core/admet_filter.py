"""
admet_filter.py — Computational ADMET filtering from RDKit descriptors.

Computes and filters on:
  - Lipinski rule of five (oral absorption)
  - QED — Quantitative Estimate of Druglikeness
  - SA Score — Synthetic Accessibility
  - PSA — Polar Surface Area
  - LogP
  - Structural alerts (RDKit's PAINS/BRENK/NIH catalogs)

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
    "mw_max"     : 600,    # Da  — slightly looser than strict Lipinski
    "logp_max"   : 5.5,
    "hbd_max"    : 5,
    "hba_max"    : 10,
    "psa_max"    : 140,    # Å²
    "rotbonds_max": 10,
    "qed_min"    : 0.30,
    "sa_max"     : 5.0,    # 1=easy, 10=hard; <=5 counts as accessible
    "lip_violations_max": 1,   # one Ro5 violation tolerated
    # Catalogs that reject the molecule. PAINS are Baell & Holloway's
    # aggregators and promiscuous binders; BRENK and NIH are aggressive
    # enough to discard perfectly viable chemotypes, so they are reported
    # separately instead of rejecting.
    "reject_catalogs": ("PAINS",),
    "report_catalogs": ("BRENK", "NIH"),
}


def compute_admet(mol, filters: Dict = None) -> Dict[str, Any]:
    """
    Compute every ADMET descriptor for one molecule.
    Returns a dict of metrics plus a global 'passes' flag.

    `filters` overrides DEFAULT_FILTERS. Omit it and the defaults apply —
    but a caller who does pass its own filters now sees them applied to
    both the numeric thresholds and the alert catalogs, which used to stay
    pinned to the defaults whatever the caller asked for.
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

    # Structural alerts: RDKit's curated catalogs
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
        "passes"      : False,   # assigned below
    }

    result["passes"] = _passes_filters(result, active)
    return result


def filter_molecules(
    smiles_list: List[str],
    filters: Dict = None,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """
    Filter a list of SMILES on ADMET criteria.

    Returns
    -------
    List of dicts: {'smiles', 'mol', ...ADMET metrics..., 'passes'}
    sorted by descending QED.
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

    _log(f"[ADMET] Evaluated: {len(results)} | Pass: {len(passing)} | "
         f"Fail: {len(failing)} | Invalid: {n_invalid}")

    return passing + failing


def admet_summary_table(results: List[Dict]) -> str:
    """ASCII table of the ADMET metrics of each molecule."""
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
    """SA Score from RDKit contrib."""
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
# The catalogs come from RDKit (FilterCatalog), not from hand-written SMARTS.
# The six patterns that used to live here were labelled PAINS without being
# PAINS: a para-substituted quinone, the textbook PAINS chemotype, matched
# none of the six, while a bare pyrrole — present in perfectly viable
# chemotypes — rejected the molecule outright. The alpha-amino pattern was
# additionally written stereospecific ([C@@H]) to no effect, since
# HasSubstructMatch ignores chirality unless explicitly asked.

_catalog_cache: Dict[tuple, Any] = {}


def _get_catalog(names: tuple):
    """FilterCatalog for the requested catalogs; built once and cached."""
    from rdkit.Chem import FilterCatalog
    from rdkit.Chem.FilterCatalog import FilterCatalogParams

    key = tuple(sorted(names))
    if key in _catalog_cache:
        return _catalog_cache[key]

    params = FilterCatalogParams()
    for name in key:
        entry = getattr(FilterCatalogParams.FilterCatalogs, name.upper(), None)
        if entry is None:
            print(f"[ADMET] Unknown catalog, ignored: {name}", file=sys.stderr)
            continue
        params.AddCatalog(entry)
    catalog = FilterCatalog.FilterCatalog(params)
    _catalog_cache[key] = catalog
    return catalog


def _check_structural_alerts(mol, catalogs: tuple = ("PAINS",)) -> List[str]:
    """
    Names of the alerts found, empty if none. The names are RDKit's, so
    each one can be traced back to the filter's original publication.
    """
    if mol is None or not catalogs:
        return []
    try:
        catalog = _get_catalog(tuple(catalogs))
    except Exception as e:
        print(f"[ADMET] FilterCatalog unavailable ({e}); no alerts reported.",
              file=sys.stderr)
        return []

    try:
        return sorted({m.GetDescription() for m in catalog.GetMatches(mol)})
    except Exception:
        return []
