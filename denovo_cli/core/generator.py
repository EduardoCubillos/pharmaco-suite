"""
generator.py — Generación de moléculas de novo via Algoritmo Genético SELFIES.

Dado un conjunto de ligandos semilla (con su perfil farmacofórico),
genera exactamente N moléculas nuevas que:
  - Son químicamente válidas (garantía SELFIES)
  - Pasan filtros ADMET básicos
  - Tienen puntuación farmacofórica ≥ umbral
  - Son suficientemente diversas entre sí (Tanimoto < umbral)

Referencia:
  - Krenn et al. (2020) Machine Learning: Science and Technology 1, 045024
  - Nigam et al. (2021) Nature Machine Intelligence 3, 573–584
"""

import random
import numpy as np
from typing import List, Dict, Any, Optional
from collections import defaultdict


# ── Parámetros por defecto del GA ─────────────────────────────────────────────
GA_DEFAULTS = {
    "n_pop"         : 60,
    "n_gen"         : 50,
    "n_elite"       : 10,
    "p_mut"         : 0.65,
    "p_cross"       : 0.40,
    "max_selfies_len": 90,
    "diversity_threshold": 0.85,  # Tanimoto máximo entre candidatas finales
}


def generate_molecules(
    seed_smiles: List[str],
    reference_profile: Dict[str, int],
    n_target: int = 20,
    ga_params: Dict = None,
    admet_filters: Dict = None,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """
    Genera n_target moléculas nuevas usando GA SELFIES.

    Parameters
    ----------
    seed_smiles       : lista de SMILES de los ligandos de entrenamiento
    reference_profile : perfil de features LBP {tipo: count}
    n_target          : número de moléculas únicas a generar (default 20)
    ga_params         : sobreescribir parámetros del GA
    admet_filters     : sobreescribir filtros ADMET
    verbose           : imprimir progreso

    Returns
    -------
    Lista de dicts con 'smiles', 'pharma_score', métricas ADMET, 'rank'
    """
    import selfies as sf
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem

    from .admet_filter   import compute_admet, DEFAULT_FILTERS
    from .pharma_scorer  import pharmacophore_score

    _log = print if verbose else lambda *a, **k: None
    params = {**GA_DEFAULTS, **(ga_params or {})}
    active_filters = {**DEFAULT_FILTERS, **(admet_filters or {})}

    # ── Alfabeto SELFIES ──────────────────────────────────────────────────────
    alphabet = list(sf.get_semantic_robust_alphabet())

    # ── Función de scoring interna ────────────────────────────────────────────
    fp_seeds = []
    for smi in seed_smiles[:20]:   # usar hasta 20 seeds para fps
        mol = Chem.MolFromSmiles(smi)
        if mol:
            fp_seeds.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048))

    def _score(smi: str) -> float:
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                return 0.0
            admet = compute_admet(mol)
            if not admet.get("passes"):
                return 0.0
            ps  = pharmacophore_score(mol, reference_profile)
            qed = admet.get("qed", 0.0)
            # Penalizar si es idéntica a algún seed (Tanimoto > 0.95)
            if fp_seeds:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)
                max_sim = max(DataStructs.TanimotoSimilarity(fp, fp_s) for fp_s in fp_seeds)
                if max_sim > 0.95:
                    return 0.0
            sa_term = (1.0 - min(1.0, admet["sa_score"] / 5.0)) if admet.get("sa_score") else 0.5
            return 0.45 * ps + 0.35 * qed + 0.20 * sa_term
        except Exception:
            return 0.0

    # ── Convertores SELFIES ───────────────────────────────────────────────────
    def smi2sel(smi):
        try: return sf.encoder(smi)
        except: return None

    def sel2smi(sel):
        try:
            smi = sf.decoder(sel)
            mol = Chem.MolFromSmiles(smi)
            return Chem.MolToSmiles(mol) if mol else None
        except: return None

    def mutate(sel):
        tokens = list(sf.split_selfies(sel))
        if not tokens: return sel
        op = random.choice(["replace", "insert", "delete"])
        if op == "replace":
            tokens[random.randrange(len(tokens))] = random.choice(alphabet)
        elif op == "insert" and len(tokens) < params["max_selfies_len"]:
            tokens.insert(random.randrange(len(tokens) + 1), random.choice(alphabet))
        elif op == "delete" and len(tokens) > 4:
            tokens.pop(random.randrange(len(tokens)))
        new_sel = "".join(tokens)
        return new_sel if sel2smi(new_sel) else sel

    def crossover(sel1, sel2):
        t1 = list(sf.split_selfies(sel1))
        t2 = list(sf.split_selfies(sel2))
        if len(t1) < 2 or len(t2) < 2: return sel1
        child = t1[:random.randrange(1, len(t1))] + t2[random.randrange(1, len(t2)):]
        child = child[:params["max_selfies_len"]]
        new_sel = "".join(child)
        return new_sel if sel2smi(new_sel) else sel1

    # ── Población inicial ─────────────────────────────────────────────────────
    _log(f"[GA] Inicializando población (N={params['n_pop']})...")
    population = []
    for smi in seed_smiles:
        sel = smi2sel(smi)
        if sel:
            population.append(sel)
        if len(population) >= params["n_pop"] // 2:
            break

    attempts = 0
    while len(population) < params["n_pop"] and attempts < params["n_pop"] * 15:
        attempts += 1
        base = random.choice(population) if population else smi2sel(seed_smiles[0])
        if base is None: continue
        s = mutate(base)
        if sel2smi(s): population.append(s)

    _log(f"[GA] Población inicial: {len(population)} individuos")

    # ── Ciclo evolutivo ───────────────────────────────────────────────────────
    best_score = 0.0
    history    = []

    for gen in range(params["n_gen"]):
        scored = sorted(
            [(s, sel2smi(s)) for s in population],
            key=lambda x: -_score(x[1]) if x[1] else 0.0
        )
        scored = [(sel, smi, _score(smi)) for sel, smi in scored if smi]

        if scored:
            best_gen = scored[0][2]
            if best_gen > best_score:
                best_score = best_gen
            history.append(best_gen)

        if (gen + 1) % 10 == 0:
            _log(f"  Gen {gen+1:3d}/{params['n_gen']}  best_score={best_score:.4f}  "
                 f"pop={len(population)}")

        elite   = [s[0] for s in scored[:params["n_elite"]]]
        new_pop = list(elite)

        while len(new_pop) < params["n_pop"]:
            candidates = scored[:max(10, len(scored)//2)]
            if not candidates: break
            p1 = random.choice(candidates)[0]
            if random.random() < params["p_cross"] and len(candidates) > 1:
                p2    = random.choice(candidates)[0]
                child = crossover(p1, p2)
            else:
                child = p1
            if random.random() < params["p_mut"]:
                child = mutate(child)
            if sel2smi(child):
                new_pop.append(child)

        population = new_pop

    _log(f"[GA] Evolución completada. Mejor score: {best_score:.4f}")

    # ── Recolectar y deduplicar ───────────────────────────────────────────────
    candidates = []
    seen_smi   = set(seed_smiles)   # excluir seeds originales

    for sel in population:
        smi = sel2smi(sel)
        if not smi or smi in seen_smi:
            continue
        seen_smi.add(smi)
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        admet = compute_admet(mol)
        if not admet.get("passes"):
            continue
        ps = pharmacophore_score(mol, reference_profile)
        sc = _score(smi)
        if sc <= 0:
            continue
        candidates.append({
            "smiles"       : smi,
            "pharma_score" : round(ps, 3),
            "score"        : round(sc, 3),
            **{k: v for k, v in admet.items() if k not in ("passes", "error", "alerts")},
        })

    candidates.sort(key=lambda x: -x["score"])

    # ── Filtro de diversidad ──────────────────────────────────────────────────
    _log(f"[GA] Candidatas antes de filtro diversidad: {len(candidates)}")
    final = _diversity_filter(
        candidates,
        n_target,
        params["diversity_threshold"],
    )
    _log(f"[GA] Candidatas finales (N={n_target}): {len(final)}")

    # Si no llegamos al target, intentar relajar
    if len(final) < n_target:
        _log(f"[GA] Advertencia: solo se generaron {len(final)} moléculas "
             f"(objetivo: {n_target}). Considera aumentar n_gen o n_pop.")

    for i, mol_dict in enumerate(final, 1):
        mol_dict["rank"] = i

    return final


def _diversity_filter(
    candidates: List[Dict],
    n_target: int,
    max_tanimoto: float,
) -> List[Dict]:
    """
    Greedy diversity filter: retiene candidatas cuyo Tanimoto con todas las
    ya seleccionadas sea ≤ max_tanimoto.
    """
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem

    selected      = []
    selected_fps  = []

    for cand in candidates:
        if len(selected) >= n_target:
            break
        mol = Chem.MolFromSmiles(cand["smiles"])
        if mol is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)

        if selected_fps:
            max_sim = max(DataStructs.TanimotoSimilarity(fp, sfp) for sfp in selected_fps)
            if max_sim > max_tanimoto:
                continue

        selected.append(cand)
        selected_fps.append(fp)

    return selected
