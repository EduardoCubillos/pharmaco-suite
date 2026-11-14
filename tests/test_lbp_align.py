"""Does the LBP land in receptor space when atom order is scrambled?"""
import os, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "pharmacophore_cli"))
_TMP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_tmp")
os.makedirs(_TMP, exist_ok=True)
import random
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from core.lbp import generate_lbp

SMI = "CC(=O)Oc1ccccc1C(=O)O"          # aspirin
RESN = "LIG"

def crystal_pdb(path, scramble=True, rot=True):
    mol = Chem.AddHs(Chem.MolFromSmiles(SMI))
    AllChem.EmbedMolecule(mol, randomSeed=7)
    AllChem.MMFFOptimizeMolecule(mol)
    mol = Chem.RemoveHs(mol)
    conf = mol.GetConformer()
    xyz = np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])
    if rot:
        th = 0.9
        R = np.array([[np.cos(th),-np.sin(th),0],[np.sin(th),np.cos(th),0],[0,0,1.0]])
        xyz = xyz @ R.T + np.array([25.0, -12.0, 40.0])
    order = list(range(mol.GetNumAtoms()))
    if scramble:
        random.Random(3).shuffle(order)
    with open(path, "w") as fh:
        for n, i in enumerate(order, 1):
            el = mol.GetAtomWithIdx(i).GetSymbol()
            x, y, z = xyz[i]
            name = f"{el}{n}"
            rec = (
                "HETATM"                    # 1-6
                f"{n:>5d}"                  # 7-11  serial
                " "                         # 12
                f"{name:<4s}"               # 13-16 atom name
                " "                         # 17    altLoc
                f"{RESN:>3s}"               # 18-20 resName
                " "                         # 21
                "A"                         # 22    chainID
                f"{901:>4d}"                # 23-26 resSeq
                "    "                      # 27-30 iCode + pad
                f"{x:>8.3f}{y:>8.3f}{z:>8.3f}"   # 31-54
                "  1.00  0.00          "    # 55-76
                f"{el:>2s}"                 # 77-78 element
            )
            fh.write(rec + "\n")
        fh.write("END\n")
    return xyz

pdb = os.path.join(_TMP, "crystal.pdb")
xyz = crystal_pdb(pdb)
centre = xyz.mean(axis=0)

feats = generate_lbp([SMI], align_to_pdb=pdb, align_ligand_resname=RESN, verbose=True)
fc = np.array([f["coords"] for f in feats])
print(f"\ncrystal ligand centroid : {np.round(centre,2)}")
print(f"LBP feature centroid    : {np.round(fc.mean(axis=0),2)}")
print(f"frame                   : {feats[0]['frame']}")
d = np.linalg.norm(fc.mean(axis=0) - centre)
print(f"centroid offset         : {d:.3f} A")
assert feats[0]["frame"] == "receptor", "alignment was refused"
assert d < 1.0, f"features not in receptor space (offset {d:.2f} A)"

# every feature must sit inside the crystal ligand's own envelope
maxr = np.max(np.linalg.norm(xyz - centre, axis=1))
far = np.max(np.linalg.norm(fc - centre, axis=1))
print(f"ligand radius {maxr:.2f} A | farthest feature {far:.2f} A")
assert far < maxr + 1.5, "a feature fell outside the ligand envelope"

# wrong residue name -> must refuse, not silently return ligand-frame coords
feats2 = generate_lbp([SMI], align_to_pdb=pdb, align_ligand_resname="XXX", verbose=False)
assert feats2[0]["frame"] == "ligand", "should have refused alignment"

# a different molecule in the crystal -> no common substructure -> refuse
feats3 = generate_lbp(["c1ccncc1C(=O)NCCN"], align_to_pdb=pdb,
                      align_ligand_resname=RESN, verbose=False)
assert feats3[0]["frame"] == "ligand", "should have refused: different molecule"

print("\nOK  alignment recovers receptor space, and refuses when it cannot.")
