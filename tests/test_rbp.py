"""RBP: hydrophobics must sit on the side chain, GLY must not make one,
and the catalytic-dyad override must actually fire."""
import os, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "pharmacophore_cli"))
_TMP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_tmp")
os.makedirs(_TMP, exist_ok=True)
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from core.rbp import generate_rbp, _PROTONATION_OVERRIDES

def build(path):
    """Two ASP25 (chains A and B), plus PHE/TRP/GLY/VAL, plus a ligand."""
    blocks = []
    serial = 1
    for chain, seq, start in [("A", "DFW", 25), ("B", "DGV", 25)]:
        m = Chem.MolFromSequence(seq)
        mh = Chem.AddHs(m, addCoords=True)
        AllChem.EmbedMolecule(mh, randomSeed=5)
        AllChem.MMFFOptimizeMolecule(mh)
        m = Chem.RemoveHs(mh)
        conf = m.GetConformer()
        off = np.array([0.0, 0.0, 0.0]) if chain == "A" else np.array([9.0, 0.0, 0.0])
        for a in m.GetAtoms():
            ri = a.GetPDBResidueInfo()
            x, y, z = np.array(list(conf.GetAtomPosition(a.GetIdx()))) + off
            blocks.append(
                "ATOM  " + f"{serial:>5d}" + " " + f"{ri.GetName()[:4]:<4s}" + " "
                + f"{ri.GetResidueName():>3s}" + " " + chain
                + f"{start + ri.GetResidueNumber() - 1:>4d}" + "    "
                + f"{x:>8.3f}{y:>8.3f}{z:>8.3f}" + "  1.00  0.00          "
                + f"{a.GetSymbol():>2s}")
            serial += 1
    # a small ligand parked between the two chains so everything is "binding site"
    for n, (x, y, z) in enumerate([(4.5, 0.5, 0.5), (5.5, 1.5, 0.5), (3.5, -0.5, 1.5)], 1):
        blocks.append(
            "HETATM" + f"{serial:>5d}" + " " + f"{'C'+str(n):<4s}" + " "
            + "RIT" + " " + "A" + f"{901:>4d}" + "    "
            + f"{x:>8.3f}{y:>8.3f}{z:>8.3f}" + "  1.00  0.00          " + " C")
        serial += 1
    open(path, "w").write("\n".join(blocks) + "\nEND\n")

build(os.path.join(_TMP, "rbp_test.pdb"))
feats = generate_rbp(os.path.join(_TMP, "rbp_test.pdb"), ligand_resname="RIT", site_cutoff=25.0, verbose=False)

# 1. the dyad override fires, and differently per chain
dyad = {f["residue"]: f["label"] for f in feats if f["residue"].startswith("ASP25")}
print("ASP25 features by residue:")
for f in feats:
    if f["residue"].startswith("ASP25"):
        print(f"   {f['residue']:10s} {f['type']:14s} {f['label']}")
a_types = {f["type"] for f in feats if f["residue"] == "ASP25A"}
b_types = {f["type"] for f in feats if f["residue"] == "ASP25B"}
assert "DONOR" in a_types, f"ASP25A override did not fire (got {a_types})"
assert "NEG_IONIZABLE" in b_types, f"ASP25B should be deprotonated (got {b_types})"
assert a_types != b_types, "the two halves of the dyad came out identical"

# 2. GLY makes no hydrophobic feature
gly_hydro = [f for f in feats if f["residue"].startswith("GLY") and f["type"] == "HYDROPHOBIC"]
assert not gly_hydro, f"GLY got a hydrophobic feature: {gly_hydro}"

# 3. hydrophobics are off the CA, on the side chain
import Bio.PDB as bp
st = bp.PDBParser(QUIET=True).get_structure("t", os.path.join(_TMP, "rbp_test.pdb"))
ca = {}
for ch in st[0]:
    for r in ch:
        if "CA" in r:
            ca[f"{r.get_resname().strip()}{r.id[1]}{ch.id}"] = np.array(r["CA"].get_vector().get_array())
print("\nhydrophobic feature vs that residue's CA:")
for f in feats:
    if f["type"] != "HYDROPHOBIC":
        continue
    d = np.linalg.norm(np.array(f["coords"]) - ca[f["residue"]])
    print(f"   {f['residue']:10s} {d:5.2f} A from CA   {f['label']}")
    assert d > 0.5, f"{f['residue']} hydrophobic still sits on the CA"
print("\nOK  dyad override fires per chain, GLY excluded, hydrophobics on side chains.")
