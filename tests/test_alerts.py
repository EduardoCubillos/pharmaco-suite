"""Curated catalogs vs the six hand-written SMARTS."""
import os, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "denovo_cli"))
_TMP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_tmp")
os.makedirs(_TMP, exist_ok=True)
from rdkit import Chem
from core.admet_filter import _check_structural_alerts, compute_admet

OLD = ["[#6]-[NH]-[NH]-[#6]", "c1ccc(cc1)-[NH]-c1ccccc1", "[O;X2]-[N;X2]=O",
       "[S;X2]-[S;X2]", "[#6][C@@H]([NH2])C(=O)", "[nH]1cccc1"]
old_p = [(s, Chem.MolFromSmarts(s)) for s in OLD]
def old_alerts(m):
    return [s for s, p in old_p if p is not None and m.HasSubstructMatch(p)]

cases = [
    ("quinone (a canonical PAINS chemotype)", "O=C1C=CC(=O)C=C1"),
    ("bare pyrrole",                          "c1cc[nH]c1"),
    ("alpha-amino ketone, (R)",               "C[C@H](N)C(=O)C"),
    ("alpha-amino ketone, (S)",               "C[C@@H](N)C(=O)C"),
    ("alpha-amino ketone, no stereo",         "CC(N)C(=O)C"),
    ("simple amide",                          "CC(=O)NC"),
]
print(f"{'molecule':40s} {'old hand-written':>17s} {'PAINS':>7s}  BRENK/NIH (reported, not rejected)")
for label, smi in cases:
    m = Chem.MolFromSmiles(smi)
    a = compute_admet(m)
    print(f"{label:40s} {len(old_alerts(m)):>17d} {len(a['alerts']):>7d}  {a['alerts_info']}")

# the two enantiomers must be treated identically
r = compute_admet(Chem.MolFromSmiles("C[C@H](N)C(=O)C"))
s = compute_admet(Chem.MolFromSmiles("C[C@@H](N)C(=O)C"))
assert r["alerts"] == s["alerts"], "alerts still depend on chirality"
# the old pattern was written stereospecifically but HasSubstructMatch
# ignores chirality by default, so the @@ was dead specificity, not an
# asymmetry: it matched every alpha-amino carbonyl regardless of centre.
assert len(old_alerts(Chem.MolFromSmiles("CC(N)C(=O)C"))) == 1, \
       "the old pattern matched the achiral case too"

# bare pyrrole used to be rejected outright
pyr = Chem.MolFromSmiles("c1cc[nH]c1")
assert old_alerts(pyr), "old list should have flagged pyrrole"
assert not compute_admet(pyr)["alerts"], "PAINS should not flag bare pyrrole"

# a real PAINS chemotype must still be caught -- and the old list missed it
q_mol = Chem.MolFromSmiles("O=C1C=CC(=O)C=C1")
assert not old_alerts(q_mol), "old list unexpectedly caught the quinone"
q = compute_admet(q_mol)
assert q["alerts"], "quinone must be flagged"
print(f"\nquinone alert names: {q['alerts']}")
print("OK  curated catalog, chirality-independent, pyrrole no longer rejected.")
