"""The fallback must not decide donor/acceptor from the element alone."""
import os, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "pharmacophore_cli"))
_TMP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_tmp")
os.makedirs(_TMP, exist_ok=True)
from core.sbp import _biopython_fallback

def write(path, with_h):
    """A pyridine-like N and a carbonyl O on the ligand, a SER OG opposite."""
    rows = []
    n = 1
    def atom(rec, name, res, chain, seq, xyz, el):
        nonlocal n
        x, y, z = xyz
        rows.append(f"{rec:<6s}{n:>5d} {name:<4s} {res:>3s} {chain}{seq:>4d}    "
                    f"{x:>8.3f}{y:>8.3f}{z:>8.3f}  1.00  0.00          {el:>2s}")
        n += 1
    # protein: one SER, its OG within H-bond range of both ligand polar atoms
    atom("ATOM  ", "N",  "SER", "A", 1, (0.0, 3.0, 0.0), "N")
    atom("ATOM  ", "CA", "SER", "A", 1, (1.0, 3.5, 0.0), "C")
    atom("ATOM  ", "C",  "SER", "A", 1, (2.0, 3.0, 0.0), "C")
    atom("ATOM  ", "O",  "SER", "A", 1, (2.5, 3.8, 0.0), "O")
    atom("ATOM  ", "CB", "SER", "A", 1, (1.2, 4.9, 0.0), "C")
    atom("ATOM  ", "OG", "SER", "A", 1, (1.0, 2.0, 0.0), "O")
    # ligand: N1 (no H -> acceptor only), O2 carbonyl, O3 hydroxyl (has H)
    atom("HETATM", "N1", "LIG", "A", 901, (1.0, -0.8, 0.0), "N")
    atom("HETATM", "C2", "LIG", "A", 901, (2.2, -1.4, 0.0), "C")
    atom("HETATM", "O3", "LIG", "A", 901, (-0.3, -0.9, 0.0), "O")
    if with_h:
        atom("HETATM", "H3", "LIG", "A", 901, (-0.6, -0.1, 0.5), "H")
    open(path, "w").write("\n".join(rows) + "\nEND\n")

for with_h, tag in [(False, "no hydrogens"), (True, "with an O-H")]:
    write(os.path.join(_TMP, "sbp_test.pdb"), with_h)
    feats = _biopython_fallback(os.path.join(_TMP, "sbp_test.pdb"), "LIG", verbose=False)
    print(f"\n[{tag}]")
    by_atom = {}
    for f in feats:
        key = f["label"].split(", ")[1]
        by_atom.setdefault(key, []).append(f["type"])
    for k in sorted(by_atom):
        print(f"   {k:4s} -> {sorted(set(by_atom[k]))}")
    if not with_h:
        # old behaviour: N -> DONOR, O -> ACCEPTOR, each a guess from the element
        assert sorted(set(by_atom["N1"])) == ["ACCEPTOR", "DONOR"], by_atom["N1"]
        assert sorted(set(by_atom["O3"])) == ["ACCEPTOR", "DONOR"], by_atom["O3"]
    else:
        assert by_atom["N1"] == ["ACCEPTOR"], f"N without H must only accept: {by_atom['N1']}"
        assert sorted(set(by_atom["O3"])) == ["ACCEPTOR", "DONOR"], by_atom["O3"]
print("\nOK  roles resolved from hydrogens when present, ambiguity kept visible when not.")
