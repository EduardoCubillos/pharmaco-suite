"""
ligand_fetcher.py — Obtener ligandos activos desde ChEMBL o desde archivo.

Dos fuentes:
  1. ChEMBL API   : dado un nombre o ChEMBL ID de target, descarga SMILES
                    de compuestos con actividad documentada (IC50/Ki/Kd).
  2. Archivo local: .smi (un SMILES por línea), .sdf, o .csv/.tsv (col 0).

Referencia ChEMBL API: https://www.ebi.ac.uk/chembl/api/data/
"""

import os
import sys
import requests
import time
from typing import List, Dict, Any, Optional


CHEMBL_BASE = "https://www.ebi.ac.uk/chembl/api/data"
ACTIVITY_TYPES = ["IC50", "Ki", "Kd", "EC50", "Inhibition"]


# ── Punto de entrada principal ────────────────────────────────────────────────

def fetch_ligands(
    source: str,
    n_max: int = 100,
    activity_cutoff: float = 10000,   # nM — filtro de potencia (default 10 µM)
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """
    Carga ligandos desde ChEMBL (nombre/ID de target) o desde archivo local.

    Parameters
    ----------
    source      : nombre del target (ej. "HIV protease"), ChEMBL ID (ej. "CHEMBL2074"),
                  o ruta a un archivo (.smi / .sdf / .csv)
    n_max       : máximo de ligandos a devolver
    activity_cutoff : potencia máxima en nM para incluir el compuesto (default 10 µM)
    verbose     : imprimir progreso

    Returns
    -------
    Lista de dicts: {'smiles', 'chembl_id', 'source', 'activity_type', 'activity_value'}
    """
    _log = print if verbose else lambda *a, **k: None

    if os.path.exists(source):
        _log(f"[Ligandos] Cargando desde archivo: {source}")
        return _load_from_file(source, n_max, verbose)
    else:
        _log(f"[Ligandos] Buscando en ChEMBL: '{source}'")
        return _fetch_from_chembl(source, n_max, activity_cutoff, verbose)


# ── Carga desde archivo ───────────────────────────────────────────────────────

def _load_from_file(path: str, n_max: int, verbose: bool) -> List[Dict[str, Any]]:
    """Soporta .smi, .sdf, .csv, .tsv"""
    from rdkit import Chem
    _log = print if verbose else lambda *a, **k: None

    ext = os.path.splitext(path)[1].lower()
    results = []

    if ext in (".smi", ".smiles"):
        with open(path) as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                smi   = parts[0]
                name  = parts[1] if len(parts) > 1 else f"mol_{i+1:04d}"
                mol   = Chem.MolFromSmiles(smi)
                if mol:
                    results.append({
                        "smiles"        : smi,
                        "chembl_id"     : name,
                        "source"        : "file",
                        "activity_type" : "unknown",
                        "activity_value": None,
                    })
                if len(results) >= n_max:
                    break

    elif ext == ".sdf":
        supplier = Chem.SDMolSupplier(path, removeHs=True)
        for i, mol in enumerate(supplier):
            if mol is None:
                continue
            smi  = Chem.MolToSmiles(mol)
            name = mol.GetProp("_Name") if mol.HasProp("_Name") else f"mol_{i+1:04d}"
            results.append({
                "smiles"        : smi,
                "chembl_id"     : name,
                "source"        : "file",
                "activity_type" : "unknown",
                "activity_value": None,
            })
            if len(results) >= n_max:
                break

    elif ext in (".csv", ".tsv"):
        sep = "\t" if ext == ".tsv" else ","
        with open(path) as f:
            for i, line in enumerate(f):
                if i == 0 and "smiles" in line.lower():
                    continue   # saltar header
                parts = line.strip().split(sep)
                if not parts:
                    continue
                smi = parts[0].strip()
                mol = Chem.MolFromSmiles(smi)
                if mol:
                    results.append({
                        "smiles"        : smi,
                        "chembl_id"     : parts[1].strip() if len(parts) > 1 else f"mol_{i:04d}",
                        "source"        : "file",
                        "activity_type" : "unknown",
                        "activity_value": None,
                    })
                if len(results) >= n_max:
                    break
    else:
        print(f"[Ligandos] Formato no reconocido: {ext}. Use .smi, .sdf, .csv o .tsv",
              file=sys.stderr)

    _log(f"[Ligandos] {len(results)} ligandos cargados desde archivo.")
    return results


# ── Fetch desde ChEMBL ────────────────────────────────────────────────────────

def _fetch_from_chembl(
    query: str,
    n_max: int,
    activity_cutoff: float,
    verbose: bool,
) -> List[Dict[str, Any]]:
    """
    1. Busca el target en ChEMBL (por nombre o ID).
    2. Descarga actividades del target.
    3. Filtra por tipo de actividad y potencia.
    4. Devuelve SMILES únicos.
    """
    _log = print if verbose else lambda *a, **k: None

    # Paso 1: resolver target ID
    target_id = _resolve_target_id(query, verbose)
    if not target_id:
        print(f"[Ligandos] ERROR: target '{query}' no encontrado en ChEMBL.",
              file=sys.stderr)
        print("[Ligandos] Sugerencia: prueba con el nombre en inglés o con el ID ChEMBL directamente (ej. CHEMBL2074).",
              file=sys.stderr)
        return []

    _log(f"[Ligandos] Target resuelto: {target_id}")

    # Paso 2: descargar actividades
    activities = _fetch_activities(target_id, n_max * 3, activity_cutoff, verbose)
    _log(f"[Ligandos] Actividades descargadas: {len(activities)}")

    # Paso 3: obtener SMILES únicos
    results = _activities_to_smiles(activities, n_max, verbose)
    _log(f"[Ligandos] {len(results)} ligandos únicos válidos.")
    return results


def _resolve_target_id(query: str, verbose: bool) -> Optional[str]:
    """Devuelve el ChEMBL ID del target, buscando por nombre o ID directo."""
    _log = print if verbose else lambda *a, **k: None

    # Si ya es un ID ChEMBL
    if query.upper().startswith("CHEMBL"):
        return query.upper()

    # Buscar por nombre
    url = f"{CHEMBL_BASE}/target/search.json"
    params = {"q": query, "limit": 10}
    try:
        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
        targets = r.json().get("targets", [])
        if not targets:
            return None

        query_lower = query.lower()

        # 1. Coincidencia exacta de nombre (case-insensitive)
        for t in targets:
            if t.get("pref_name", "").lower() == query_lower:
                _log(f"[ChEMBL] Coincidencia exacta: {t['pref_name']} ({t['target_chembl_id']})")
                return t["target_chembl_id"]

        # 2. Nombre contiene la query y es SINGLE PROTEIN
        for t in targets:
            name = t.get("pref_name", "").lower()
            if query_lower in name and t.get("target_type") == "SINGLE PROTEIN":
                _log(f"[ChEMBL] Encontrado: {t['pref_name']} ({t['target_chembl_id']})")
                return t["target_chembl_id"]

        # 3. Cualquier SINGLE PROTEIN
        for t in targets:
            if t.get("target_type") == "SINGLE PROTEIN":
                _log(f"[ChEMBL] Usando (SINGLE PROTEIN): {t['pref_name']} ({t['target_chembl_id']})")
                return t["target_chembl_id"]

        # 4. Fallback: primer resultado
        t = targets[0]
        _log(f"[ChEMBL] Usando (primer resultado): {t['pref_name']} ({t['target_chembl_id']})")
        return t["target_chembl_id"]

    except Exception as e:
        print(f"[ChEMBL] Error buscando target: {e}", file=sys.stderr)
        return None


def _fetch_activities(
    target_id: str,
    limit: int,
    activity_cutoff: float,
    verbose: bool,
) -> List[Dict]:
    """Descarga actividades de ChEMBL para el target dado."""
    _log = print if verbose else lambda *a, **k: None

    activities = []
    offset = 0
    page_size = 100

    while len(activities) < limit:
        url = f"{CHEMBL_BASE}/activity.json"
        params = {
            "target_chembl_id": target_id,
            "standard_type__in": ",".join(ACTIVITY_TYPES),
            "standard_relation__in": "=,<,<=",
            "limit": min(page_size, limit - len(activities)),
            "offset": offset,
        }
        try:
            r = requests.get(url, params=params, timeout=30)
            r.raise_for_status()
            data = r.json()
            batch = data.get("activities", [])
            if not batch:
                break
            activities.extend(batch)
            offset += len(batch)
            _log(f"  {len(activities)} actividades descargadas...")
            if len(batch) < page_size:
                break
            time.sleep(0.2)   # ser amable con la API
        except Exception as e:
            print(f"[ChEMBL] Error en descarga: {e}", file=sys.stderr)
            break

    return activities


def _activities_to_smiles(
    activities: List[Dict],
    n_max: int,
    verbose: bool,
) -> List[Dict[str, Any]]:
    """Convierte actividades a lista de SMILES únicos válidos."""
    from rdkit import Chem
    _log = print if verbose else lambda *a, **k: None

    seen_smiles = set()
    seen_chembl = set()
    results = []

    for act in activities:
        if len(results) >= n_max:
            break

        smi = act.get("canonical_smiles") or act.get("molecule_structures", {})
        if isinstance(smi, dict):
            smi = smi.get("canonical_smiles", "")
        if not smi:
            # Intentar obtener SMILES del compuesto
            mol_id = act.get("molecule_chembl_id")
            if mol_id and mol_id not in seen_chembl:
                smi = _get_molecule_smiles(mol_id)
                seen_chembl.add(mol_id)

        if not smi:
            continue

        # Validar con RDKit
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue

        canon_smi = Chem.MolToSmiles(mol)
        if canon_smi in seen_smiles:
            continue
        seen_smiles.add(canon_smi)

        # Extraer valor de actividad
        try:
            val = float(act.get("standard_value") or 0)
        except (TypeError, ValueError):
            val = None

        results.append({
            "smiles"        : canon_smi,
            "chembl_id"     : act.get("molecule_chembl_id", "?"),
            "source"        : "ChEMBL",
            "activity_type" : act.get("standard_type", "?"),
            "activity_value": val,
            "activity_units": act.get("standard_units", "nM"),
        })

    return results


def _get_molecule_smiles(chembl_id: str) -> Optional[str]:
    """Obtiene el SMILES canónico de un compuesto ChEMBL por su ID."""
    try:
        url = f"{CHEMBL_BASE}/molecule/{chembl_id}.json"
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        structs = r.json().get("molecule_structures") or {}
        return structs.get("canonical_smiles")
    except Exception:
        return None


# ── Guardar ligandos a archivo ────────────────────────────────────────────────

def save_ligands_smi(ligands: List[Dict], path: str) -> str:
    """Guarda los ligandos en formato .smi (SMILES + nombre + actividad)."""
    with open(path, "w") as f:
        f.write("# SMILES  ID  activity_type  activity_value\n")
        for lig in ligands:
            val = f"{lig['activity_value']:.2f}" if lig.get("activity_value") else "?"
            f.write(f"{lig['smiles']}\t{lig['chembl_id']}\t"
                    f"{lig.get('activity_type','?')}\t{val}\n")
    return path
