"""The 3D fit must be frame-independent, reward the true ligand, reject others."""
import os, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "denovo_cli"))
_TMP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_tmp")
os.makedirs(_TMP, exist_ok=True)
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, ChemicalFeatures, RDConfig
from core.pharma_scorer import score_against_pdb_pharmacophore, FTYPE_MAP

factory = ChemicalFeatures.BuildFeatureFactory(
    os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef"))

LIG = "CC(=O)Nc1ccc(O)cc1"           # paracetamol
DECOY = "CCCCCCCCCCCC"               # dodecane: no polar groups at all

def pharmacophore_from(smi, shift=np.zeros(3), rot=None, seed=11):
    """Build a pharmacophore from one conformer's own feature centroids."""
    m = Chem.AddHs(Chem.MolFromSmiles(smi))
    p = AllChem.ETKDGv3(); p.randomSeed = seed
    AllChem.EmbedMolecule(m, p); AllChem.MMFFOptimizeMolecule(m)
    conf = m.GetConformer()
    pts = []
    for f in factory.GetFeaturesForMol(m):
        t = FTYPE_MAP.get(f.GetFamily())
        if not t: continue
        pos = np.array([list(conf.GetAtomPosition(i)) for i in f.GetAtomIds()]).mean(axis=0)
        if rot is not None: pos = rot @ pos
        pts.append({"type": t, "coords": tuple(pos + shift), "weight": 3})
    # dedupe co-located points of the same type
    out = []
    for q in pts:
        if not any(r["type"] == q["type"] and
                   np.linalg.norm(np.array(r["coords"]) - np.array(q["coords"])) < 1.0
                   for r in out):
            out.append(q)
    return out

ph_origin = pharmacophore_from(LIG)
th = 1.3
R = np.array([[np.cos(th),0,np.sin(th)],[0,1,0],[-np.sin(th),0,np.cos(th)]])
ph_moved = pharmacophore_from(LIG, shift=np.array([120., -80., 300.]), rot=R)

print(f"pharmacophore points: {len(ph_origin)}  "
      f"({[p['type'] for p in ph_origin]})\n")

for label, ph in [("at origin", ph_origin), ("translated+rotated", ph_moved)]:
    s_lig,  i_lig  = score_against_pdb_pharmacophore(Chem.MolFromSmiles(LIG),   ph, return_details=True)
    s_dec,  i_dec  = score_against_pdb_pharmacophore(Chem.MolFromSmiles(DECOY), ph, return_details=True)
    print(f"[{label}]")
    print(f"  true ligand : score {s_lig:.3f}  points {i_lig['n_points_matched']}/{i_lig['n_points_total']}  rmsd {i_lig['fit_rmsd']}")
    print(f"  decoy       : score {s_dec:.3f}  points {i_dec['n_points_matched']}/{i_dec['n_points_total']}  reason {i_dec['reason']}")
    assert s_lig > 0.8, f"true ligand should fit its own pharmacophore (got {s_lig})"
    assert s_dec < s_lig, "decoy outscored the true ligand"

s1 = score_against_pdb_pharmacophore(Chem.MolFromSmiles(LIG), ph_origin)
s2 = score_against_pdb_pharmacophore(Chem.MolFromSmiles(LIG), ph_moved)
print(f"\nframe invariance: {s1:.3f} vs {s2:.3f}")
assert abs(s1 - s2) < 1e-9, "score changed when the pharmacophore was moved"
print("OK  frame-independent, discriminating.")
