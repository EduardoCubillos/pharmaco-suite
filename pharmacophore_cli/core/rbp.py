"""
rbp.py — Receptor-Based Pharmacophore (RBP).

Maps binding site residues to pharmacophoric features at a given pH.
RBP features describe what the *receptor* offers — the complementary
picture to the ligand-perspective models (SBP, LBP):

  receptor DONOR    → ligand must have an ACCEPTOR
  receptor ACCEPTOR → ligand must have a DONOR

Protonation states are derived from standard pKa values (Henderson-
Hasselbalch) with overrides for known catalytic residues. The one override
shipped here encodes the monoprotonated HIV-1 protease catalytic dyad, where
one Asp25 is protonated and the other is not; a residue whose pKa is shifted
by its own environment needs an entry of its own, since the bulk pKa table
cannot predict it. CITATION NEEDED before publication: pick the reference you
want to stand behind for the dyad's protonation state.

References:
  - Sliwoski et al. (2014) Pharmacol. Rev. 66, 334–395 — review of
    computational methods in drug discovery, incl. receptor-based
    pharmacophore construction
  - Koes & Camacho (2011) J. Chem. Inf. Model. 51, 1307–1314 — Pharmer, on
    searching with the kind of model this module produces
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional

# Standard side-chain pKa values at 25°C (Lehninger Biochemistry, 5th ed.)
_STANDARD_PKA: Dict[str, float] = {
    "ASP": 3.9,
    "GLU": 4.1,
    "HIS": 6.0,
    "CYS": 8.3,
    "TYR": 10.1,
    "LYS": 10.5,
    "ARG": 12.5,
}

# Hard overrides for known catalytic residues (ignores pKa-based calculation).
# HIV-1 protease: the catalytic dyad is monoprotonated — Asp25 of chain A
# donates, Asp25 of chain B accepts.
# Keyed by (resname, residue number, chain id). The chain matters: the two
# halves of the dyad are the same residue number in different chains, so a
# key without the chain cannot distinguish them.
_PROTONATION_OVERRIDES: Dict[Tuple[str, int, str], str] = {
    ("ASP", 25, "A"): "PROTONATED",
    ("ASP", 25, "B"): "DEPROTONATED",
}

# Backbone atoms, excluded when locating a side chain
_BACKBONE = {"N", "CA", "C", "O", "OXT"}

# Rings whose centroid is the right place for an aromatic hydrophobic feature
_RING_ATOMS: Dict[str, List[str]] = {
    "PHE": ["CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
    "TRP": ["CD2", "CE2", "CE3", "CZ2", "CZ3", "CH2"],   # benzo ring of indole
}

_BINDING_SITE_CUTOFF = 6.0   # Å from any ligand atom
_DEFAULT_PH          = 7.4


def generate_rbp(
    pdb_file: str,
    ligand_resname: str = "RIT",
    site_cutoff: float = _BINDING_SITE_CUTOFF,
    ph: float = _DEFAULT_PH,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """
    Generate RBP features for the binding site around the specified ligand.

    Parameters
    ----------
    pdb_file       : path to the PDB structure
    ligand_resname : residue name of the ligand
    site_cutoff    : radius (Å) defining the binding site around the ligand
    ph             : pH for protonation state calculation
    verbose        : print progress

    Returns
    -------
    List of features with 'type', 'coords', 'label', 'source', 'residue'
    """
    from Bio.PDB import PDBParser
    import warnings
    warnings.filterwarnings("ignore")

    _log = print if verbose else lambda *a, **k: None

    parser    = PDBParser(QUIET=True)
    structure = parser.get_structure("mol", pdb_file)
    mdl       = structure[0]

    lig_coords = _get_ligand_coords(mdl, ligand_resname)
    if not lig_coords:
        print(f"[RBP] ERROR: ligand '{ligand_resname}' not found in {pdb_file}.")
        return []

    _log(f"[RBP] Ligand: {len(lig_coords)} atoms. Binding site cutoff: {site_cutoff} Å")

    active_site = _binding_site_residues(mdl, lig_coords, site_cutoff)
    _log(f"[RBP] Binding site: {len(active_site)} residues.")

    features = []
    for res in active_site:
        features.extend(_residue_features(res, ph))

    for f in features:
        f["source"] = "RBP"

    _log(f"[RBP] {len(features)} RBP features generated.")
    return features


# ── Internals ─────────────────────────────────────────────────────────────────

def _get_ligand_coords(model, ligand_resname: str) -> List[np.ndarray]:
    name   = ligand_resname.strip().upper()
    coords = []
    for chain in model:
        for res in chain:
            if res.get_resname().strip() == name:
                for atom in res.get_atoms():
                    coords.append(atom.get_vector().get_array())
    return coords


def _binding_site_residues(model, lig_coords, cutoff: float):
    """Protein residues with at least one atom within `cutoff` Å of the ligand."""
    lig_arr = np.array(lig_coords)
    active  = []
    for chain in model:
        for res in chain:
            if res.id[0] != " ":   # skip HETATM and water
                continue
            for atom in res.get_atoms():
                ac = atom.get_vector().get_array()
                if np.min(np.linalg.norm(lig_arr - ac, axis=1)) < cutoff:
                    active.append(res)
                    break
    return active


def _residue_features(res, ph: float) -> List[Dict[str, Any]]:
    """
    Translate a residue into pharmacophoric features based on its
    protonation state at pH `ph`.
    """
    rname  = res.get_resname().strip()
    resseq = res.id[1]
    icode  = res.id[2].strip()
    resid  = f"{resseq}{icode}"
    chain  = res.get_parent().id

    # Use Cα as fallback position when specific atoms aren't available
    ca_coord = _atom_coord(res, ["CA"])
    if ca_coord is None:
        atoms = list(res.get_atoms())
        if not atoms:
            return []
        ca_coord = tuple(atoms[0].get_vector().get_array())

    prot_state = _protonation_state(rname, resseq, chain, ph)
    features   = []

    if rname in ("SER", "THR"):
        oh = _atom_coord(res, ["OG", "OG1", "OG2"]) or ca_coord
        features += [
            {"type": "DONOR",    "coords": oh, "label": f"{rname}{resid}{chain} OH donor"},
            {"type": "ACCEPTOR", "coords": oh, "label": f"{rname}{resid}{chain} OH acceptor"},
        ]

    elif rname == "ASN":
        nd2 = _atom_coord(res, ["ND2"])
        od1 = _atom_coord(res, ["OD1"])
        if nd2: features.append({"type": "DONOR",    "coords": nd2, "label": f"ASN{resid}{chain} ND2"})
        if od1: features.append({"type": "ACCEPTOR", "coords": od1, "label": f"ASN{resid}{chain} OD1"})

    elif rname == "GLN":
        ne2 = _atom_coord(res, ["NE2"])
        oe1 = _atom_coord(res, ["OE1"])
        if ne2: features.append({"type": "DONOR",    "coords": ne2, "label": f"GLN{resid}{chain} NE2"})
        if oe1: features.append({"type": "ACCEPTOR", "coords": oe1, "label": f"GLN{resid}{chain} OE1"})

    elif rname == "TYR":
        oh = _atom_coord(res, ["OH"]) or ca_coord
        if prot_state in (None, "PROTONATED"):
            features += [
                {"type": "DONOR",    "coords": oh, "label": f"TYR{resid}{chain} OH"},
                {"type": "ACCEPTOR", "coords": oh, "label": f"TYR{resid}{chain} O/OH"},
            ]
        else:
            features.append({"type": "ACCEPTOR", "coords": oh, "label": f"TYR{resid}{chain} O⁻"})

    elif rname in ("ASP", "GLU"):
        o_names = ["OD1", "OD2"] if rname == "ASP" else ["OE1", "OE2"]
        o_list  = [_atom_coord(res, [n]) for n in o_names]
        o_list  = [c for c in o_list if c]
        cg = o_list[0] if o_list else ca_coord
        if prot_state == "PROTONATED":
            features.append({"type": "DONOR",         "coords": cg, "label": f"{rname}{resid}{chain} COOH (protonated)"})
        else:
            # pH 7.4 >> pKa of Asp/Glu → almost always deprotonated
            features.append({"type": "NEG_IONIZABLE", "coords": cg, "label": f"{rname}{resid}{chain} COO⁻"})
            features.append({"type": "ACCEPTOR",      "coords": cg, "label": f"{rname}{resid}{chain} COO⁻ acceptor"})

    elif rname == "LYS":
        nz = _atom_coord(res, ["NZ"]) or ca_coord
        features += [
            {"type": "POS_IONIZABLE", "coords": nz, "label": f"LYS{resid}{chain} NH3⁺"},
            {"type": "DONOR",         "coords": nz, "label": f"LYS{resid}{chain} NH3⁺ donor"},
        ]

    elif rname == "ARG":
        cz = _atom_coord(res, ["CZ"]) or ca_coord
        features += [
            {"type": "POS_IONIZABLE", "coords": cz, "label": f"ARG{resid}{chain} guanidinium"},
            {"type": "DONOR",         "coords": cz, "label": f"ARG{resid}{chain} guanidinium donor"},
        ]

    elif rname == "HIS":
        nd1 = _atom_coord(res, ["ND1"])
        ne2 = _atom_coord(res, ["NE2"])
        if prot_state == "PROTONATED":
            if nd1: features.append({"type": "POS_IONIZABLE", "coords": nd1, "label": f"HIS{resid}{chain} ND1 (HIP)"})
            if ne2: features.append({"type": "DONOR",         "coords": ne2, "label": f"HIS{resid}{chain} NE2 (HIP)"})
        else:
            c = nd1 or ne2 or ca_coord
            features += [
                {"type": "DONOR",    "coords": c, "label": f"HIS{resid}{chain} N-H"},
                {"type": "ACCEPTOR", "coords": c, "label": f"HIS{resid}{chain} N:"},
            ]

    elif rname == "CYS":
        sg = _atom_coord(res, ["SG"]) or ca_coord
        if prot_state == "PROTONATED":
            features += [
                {"type": "DONOR",    "coords": sg, "label": f"CYS{resid}{chain} SH"},
                {"type": "ACCEPTOR", "coords": sg, "label": f"CYS{resid}{chain} S:"},
            ]
        else:
            features += [
                {"type": "NEG_IONIZABLE", "coords": sg, "label": f"CYS{resid}{chain} S⁻"},
                {"type": "ACCEPTOR",      "coords": sg, "label": f"CYS{resid}{chain} S⁻ acceptor"},
            ]

    elif rname in ("ALA", "VAL", "LEU", "ILE", "PRO", "PHE", "TRP", "MET"):
        # The hydrophobic contact a ligand can make is with the side chain,
        # not with the backbone: Cα is 2–4 Å away from where the apolar
        # surface actually is, and for TRP the whole indole. GLY is absent
        # from this list on purpose — it has no side chain to contact.
        hydro, where = _hydrophobic_centre(res, rname)
        if hydro is not None:
            features.append({
                "type"  : "HYDROPHOBIC",
                "coords": hydro,
                "label" : f"{rname}{resid}{chain} hydrophobic ({where})",
            })

    # Backbone H-bond capability (applies to all residues)
    n_coord = _atom_coord(res, ["N"])
    o_coord = _atom_coord(res, ["O"])
    if n_coord and rname != "PRO":   # proline has no NH
        features.append({"type": "DONOR",    "coords": n_coord, "label": f"{rname}{resid}{chain} backbone N-H"})
    if o_coord:
        features.append({"type": "ACCEPTOR", "coords": o_coord, "label": f"{rname}{resid}{chain} backbone C=O"})

    for f in features:
        f["residue"] = f"{rname}{resid}{chain}"

    return features


def _protonation_state(rname: str, resseq: int, chain: str,
                       ph: float) -> Optional[str]:
    """Return 'PROTONATED', 'DEPROTONATED', or None (use pKa rule)."""
    override = _PROTONATION_OVERRIDES.get((rname, resseq, chain))
    if override:
        return override
    pka = _STANDARD_PKA.get(rname)
    if pka is None:
        return None
    return "PROTONATED" if ph < pka else "DEPROTONATED"


def _hydrophobic_centre(res, rname: str):
    """
    Where to place the hydrophobic feature of an apolar residue, and a short
    tag naming the choice. Aromatic residues get their ring centroid; the
    rest get the centroid of their side-chain heavy atoms. Returns
    (coords, tag) or (None, None) when no side chain is resolvable.
    """
    ring = _RING_ATOMS.get(rname)
    if ring:
        coords = [_atom_coord(res, [n]) for n in ring]
        coords = [c for c in coords if c]
        if len(coords) >= 3:
            return tuple(np.mean(np.array(coords), axis=0)), "ring centroid"

    side = [
        atom.get_vector().get_array()
        for atom in res.get_atoms()
        if atom.get_name().strip() not in _BACKBONE
        and (atom.element or atom.get_name().strip()[0]).strip().upper() != "H"
    ]
    if not side:
        return None, None
    return tuple(np.mean(np.array(side), axis=0)), "side-chain centroid"


def _atom_coord(res, names: List[str]) -> Optional[tuple]:
    """Return the coordinate of the first found atom, or None."""
    for name in names:
        try:
            return tuple(res[name].get_vector().get_array())
        except KeyError:
            continue
    return None
