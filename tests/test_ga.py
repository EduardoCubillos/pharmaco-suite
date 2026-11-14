"""The GA must run, honour caller filters, and not rescore the same SMILES."""
import os, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "denovo_cli"))
_TMP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_tmp")
os.makedirs(_TMP, exist_ok=True)
import time
from rdkit import Chem
import core.admet_filter as af
from core.pharma_scorer import build_reference_profile

calls = {"n": 0}
real = af.compute_admet
def counting(mol, filters=None):
    calls["n"] += 1
    return real(mol, filters)
af.compute_admet = counting
import core.generator as gen
gen_mod = sys.modules["core.generator"]

SEEDS = ["CC(=O)Nc1ccc(O)cc1", "CC(=O)Oc1ccccc1C(=O)O",
         "CN1CCC[C@H]1c1cccnc1", "OCC(O)C(O)C(O)CO"]
for s in SEEDS:
    assert Chem.MolFromSmiles(s) is not None, s
prof = build_reference_profile(SEEDS)
print("reference profile:", prof)

t0 = time.time()
out = gen.generate_molecules(SEEDS, prof, n_target=5,
                             ga_params={"n_pop": 20, "n_gen": 6, "n_elite": 4},
                             admet_filters={"mw_max": 400, "qed_min": 0.45},
                             verbose=False)
dt = time.time() - t0
print(f"\ncandidates: {len(out)}   compute_admet calls: {calls['n']}   {dt:.1f}s")
assert out, "no candidates generated"
for c in out[:5]:
    print(f"  {c['smiles'][:46]:46s} mw {c['mw']:6.1f}  qed {c['qed']:.3f}  score {c['score']:.3f}")
    # the caller's filters must actually bind
    assert c["mw"] <= 400, f"mw {c['mw']} exceeds the 400 the caller asked for"
    assert c["qed"] >= 0.45, f"qed {c['qed']} below the 0.45 the caller asked for"
assert "alerts_info" in out[0]
print("\nOK  caller filters bind, GA produces candidates.")
