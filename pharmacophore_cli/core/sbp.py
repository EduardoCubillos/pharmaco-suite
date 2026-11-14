"""
sbp.py — Structure-Based Pharmacophore (SBP).

Strategy:
1. Try PLIP via CLI — generates XML reports and protonated PDB in the output dir.
2. If the CLI isn't available, fall back to the PLIP Python API.
3. If PLIP isn't installed at all, fall back to BioPython distance analysis.

The CLI-first order matters: the CLI generates the XML + protonated PDB
traceability files that the API alone does not produce.

Reference: Salentin et al. (2015) Nucleic Acids Res. 43, W443–W447 (PLIP).
"""

import os
import sys
import subprocess
import shutil
import xml.etree.ElementTree as ET
import numpy as np
from typing import List, Dict, Any, Optional

# BioPython fallback cutoffs
_HBOND_CUTOFF   = 3.5   # Å  donor-acceptor distance
_HYDROPH_CUTOFF = 4.0   # Å  carbon-carbon contact
_HBOND_ATOMS    = {"N", "O"}
_HYDROPH_ELEMS  = {"C", "S"}
_XH_BOND_MAX    = 1.3   # Å  — longest N-H / O-H treated as covalent


def generate_sbp(
    pdb_file: str,
    ligand_resname: str = "RIT",
    plip_outdir: str = "plip_output",
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """
    Generate SBP features for the specified ligand in the PDB.

    Returns
    -------
    List of dicts: {'type', 'coords', 'label', 'source'}
    """
    _log = print if verbose else lambda *a, **k: None

    os.makedirs(plip_outdir, exist_ok=True)
    features = []

    # Attempt 1: PLIP CLI (generates XML reports + protonated PDB)
    plip_exe = _find_plip()
    if plip_exe:
        _log(f"[SBP] Running PLIP CLI: {plip_exe}")
        features = _run_plip_cli(plip_exe, pdb_file, plip_outdir, verbose)

    # Attempt 2: PLIP Python API (if CLI failed but plip is importable)
    if not features:
        features = _run_plip_api(pdb_file, plip_outdir, verbose)

    # Fallback: BioPython distance analysis
    if not features:
        _log("[SBP] Falling back to BioPython distance analysis...")
        features = _biopython_fallback(pdb_file, ligand_resname, verbose)

    for f in features:
        f["source"] = "SBP"

    _log(f"[SBP] {len(features)} features extracted.")
    return features


# ── PLIP Python API ───────────────────────────────────────────────────────────

def _run_plip_api(
    pdb_file: str,
    outdir: str,
    verbose: bool,
) -> List[Dict[str, Any]]:
    """
    Run PLIP as a Python library. Requires: pip install plip --no-build-isolation
    (the no-build-isolation flag makes pip use conda's openbabel rather than
    trying to compile it from scratch — saves a lot of pain.)
    """
    try:
        from plip.structure.preparation import PDBComplex
    except ImportError:
        return []

    _log = print if verbose else lambda *a, **k: None

    try:
        _log("[SBP] Running PLIP (Python API)...")
        mol = PDBComplex()
        mol.load_pdb(pdb_file)   # no 'as_string' param in PLIP 3.x
        mol.analyze()

        features = []
        pdb_tag  = os.path.splitext(os.path.basename(pdb_file))[0]

        for site_id, ia in mol.interaction_sets.items():

            # Write traceability reports if possible
            try:
                from plip.exchange.report import BindingSiteReport
                import lxml.etree as ET2
                bsr      = BindingSiteReport(ia)
                tag      = str(site_id).replace(":", "_").replace(" ", "_")
                xml_path = os.path.join(outdir, f"{pdb_tag}_report.xml")
                tree     = ET2.ElementTree(bsr.xmlreport)
                tree.write(xml_path, pretty_print=True, encoding="utf-8",
                           xml_declaration=True)
                _log(f"[SBP] PLIP report → {xml_path}")
            except Exception:
                pass  # reports are nice to have but not required

            try:
                prot_path = os.path.join(outdir, f"{pdb_tag}_protonated.pdb")
                mol.protcomplex.write("pdb", prot_path, overwrite=True)
            except Exception:
                pass

            # Hydrophobic contacts
            for hc in ia.hydrophobic_contacts:
                try:
                    x, y, z = hc.ligatom.coords
                    features.append({
                        "type"  : "HYDROPHOBIC",
                        "coords": (float(x), float(y), float(z)),
                        "label" : "Hydrophobic contact (PLIP)",
                    })
                except Exception:
                    continue

            # Hydrogen bonds (ligand is donor)
            for hb in ia.hbonds_ldon:
                try:
                    x, y, z = hb.d.coords
                    features.append({
                        "type"  : "DONOR",
                        "coords": (float(x), float(y), float(z)),
                        "label" : f"H-bond donor (PLIP) d={hb.distance_ad:.2f}Å",
                    })
                except Exception:
                    continue

            # Hydrogen bonds (protein is donor → ligand is acceptor)
            for hb in ia.hbonds_pdon:
                try:
                    x, y, z = hb.a.coords
                    features.append({
                        "type"  : "ACCEPTOR",
                        "coords": (float(x), float(y), float(z)),
                        "label" : f"H-bond acceptor (PLIP) d={hb.distance_ad:.2f}Å",
                    })
                except Exception:
                    continue

            # π-stacking → hydrophobic
            for ps in ia.pistacking:
                try:
                    c = ps.ligandring.center
                    features.append({
                        "type"  : "HYDROPHOBIC",
                        "coords": (float(c[0]), float(c[1]), float(c[2])),
                        "label" : "π-stacking (PLIP)",
                    })
                except Exception:
                    continue

            # Cation-π → positive ionizable
            for pc in ia.pication_laro:
                try:
                    c = pc.ring.center
                    features.append({
                        "type"  : "POS_IONIZABLE",
                        "coords": (float(c[0]), float(c[1]), float(c[2])),
                        "label" : "Cation-π ligand aromatic (PLIP)",
                    })
                except Exception:
                    continue

            # Salt bridges
            for sb in ia.saltbridge_lneg:
                try:
                    c = sb.negative.center
                    features.append({
                        "type"  : "NEG_IONIZABLE",
                        "coords": (float(c[0]), float(c[1]), float(c[2])),
                        "label" : "Salt bridge NEG (PLIP)",
                    })
                except Exception:
                    continue

            for sb in ia.saltbridge_pneg:
                try:
                    c = sb.positive.center
                    features.append({
                        "type"  : "POS_IONIZABLE",
                        "coords": (float(c[0]), float(c[1]), float(c[2])),
                        "label" : "Salt bridge POS (PLIP)",
                    })
                except Exception:
                    continue

        if features:
            _log(f"[SBP] PLIP API: {len(features)} interactions found.")
        return features

    except Exception as e:
        if verbose:
            print(f"[SBP] PLIP API error: {e}")
        import traceback; traceback.print_exc()
        return []


# ── PLIP CLI ──────────────────────────────────────────────────────────────────

def _find_plip() -> Optional[str]:
    """Return path to the PLIP executable or entry-point script, or None."""
    cmd = shutil.which("plip")
    if cmd:
        return cmd
    for path in ["/content/plip/plip/plipcmd.py",
                 os.path.expanduser("~/plip/plip/plipcmd.py")]:
        if os.path.exists(path):
            return path
    return None


def _run_plip_cli(
    plip_cmd: str,
    pdb_file: str,
    outdir: str,
    verbose: bool,
) -> List[Dict[str, Any]]:
    """Call PLIP and parse the resulting XML."""
    if plip_cmd.endswith(".py"):
        cmd = [sys.executable, plip_cmd, "-f", pdb_file, "-x", "-o", outdir]
    else:
        cmd = [plip_cmd, "-f", pdb_file, "-x", "-o", outdir]

    result = subprocess.run(cmd, capture_output=True, text=True,
                            env=os.environ.copy())

    if result.returncode != 0:
        if verbose:
            err = (result.stderr or result.stdout or "no output").strip()
            print(f"[SBP] PLIP CLI error (code {result.returncode}):")
            for line in err.splitlines()[:15]:
                print(f"       {line}")
        return []

    return _parse_plip_xml(outdir, verbose)


def _parse_plip_xml(xml_dir: str, verbose: bool) -> List[Dict[str, Any]]:
    """Extract pharmacophoric features from PLIP XML output."""
    features  = []
    xml_files = [f for f in os.listdir(xml_dir) if f.endswith(".xml")]
    if not xml_files:
        return features

    for xf in xml_files:
        if verbose:
            print(f"[SBP] Parsing {xf}...")
        tree = ET.parse(os.path.join(xml_dir, xf))
        root = tree.getroot()

        for el in root.iter("hydrophobic_interaction"):
            try:
                x = float(el.findtext("ligc_x") or 0)
                y = float(el.findtext("ligc_y") or 0)
                z = float(el.findtext("ligc_z") or 0)
                if (x, y, z) != (0.0, 0.0, 0.0):
                    features.append({
                        "type"  : "HYDROPHOBIC",
                        "coords": (x, y, z),
                        "label" : f"Hydrophobic (PLIP) lig_atom={el.findtext('lig_atom_idx', '?')}",
                    })
            except (TypeError, ValueError):
                continue

        for el in root.iter("hydrogen_bond"):
            try:
                lig_is_donor = (el.findtext("protisdon") or "").lower() == "false"
                x    = float(el.findtext("lig_x") or 0)
                y    = float(el.findtext("lig_y") or 0)
                z    = float(el.findtext("lig_z") or 0)
                dist = el.findtext("dist_h-a") or el.findtext("dist_d-a") or "?"
                ang  = el.findtext("angle") or "?"
                ftype = "DONOR" if lig_is_donor else "ACCEPTOR"
                if (x, y, z) != (0.0, 0.0, 0.0):
                    features.append({
                        "type"  : ftype,
                        "coords": (x, y, z),
                        "label" : f"{ftype} (PLIP) d={dist}Å θ={ang}°",
                    })
            except (TypeError, ValueError):
                continue

        for el in root.iter("pi_stacking"):
            try:
                x = float(el.findtext("lig_x") or 0)
                y = float(el.findtext("lig_y") or 0)
                z = float(el.findtext("lig_z") or 0)
                if (x, y, z) != (0.0, 0.0, 0.0):
                    features.append({
                        "type"  : "HYDROPHOBIC",
                        "coords": (x, y, z),
                        "label" : "π-stacking (PLIP)",
                    })
            except (TypeError, ValueError):
                continue

        for el in root.iter("pi_cation_interaction"):
            try:
                x = float(el.findtext("lig_x") or 0)
                y = float(el.findtext("lig_y") or 0)
                z = float(el.findtext("lig_z") or 0)
                if (x, y, z) != (0.0, 0.0, 0.0):
                    features.append({
                        "type"  : "POS_IONIZABLE",
                        "coords": (x, y, z),
                        "label" : "Cation-π (PLIP)",
                    })
            except (TypeError, ValueError):
                continue

        for el in root.iter("salt_bridge"):
            try:
                x      = float(el.findtext("lig_x") or 0)
                y      = float(el.findtext("lig_y") or 0)
                z      = float(el.findtext("lig_z") or 0)
                charge = (el.findtext("lig_charge") or "pos").lower()
                ftype  = "POS_IONIZABLE" if "pos" in charge else "NEG_IONIZABLE"
                if (x, y, z) != (0.0, 0.0, 0.0):
                    features.append({
                        "type"  : ftype,
                        "coords": (x, y, z),
                        "label" : f"Salt bridge {ftype} (PLIP)",
                    })
            except (TypeError, ValueError):
                continue

    return features


# ── BioPython fallback ────────────────────────────────────────────────────────

def _biopython_fallback(
    pdb_file: str,
    ligand_resname: str,
    verbose: bool,
) -> List[Dict[str, Any]]:
    """
    Distance-based feature extraction with BioPython.
    Less accurate than PLIP (no angle criteria, no protonation) but
    has no extra dependencies and always works.
    """
    from Bio.PDB import PDBParser
    import warnings
    warnings.filterwarnings("ignore")

    parser    = PDBParser(QUIET=True)
    structure = parser.get_structure("mol", pdb_file)
    model     = structure[0]

    lig_atoms  = []
    prot_atoms = []

    for chain in model:
        for res in chain:
            rname = res.get_resname().strip()
            if rname == ligand_resname.strip().upper():
                lig_atoms.extend(res.get_atoms())
            elif res.id[0] == " ":
                prot_atoms.extend(res.get_atoms())

    if not lig_atoms:
        print(f"[SBP] WARNING: ligand '{ligand_resname}' not found in {pdb_file}.")
        print(f"[SBP] Available HETATM residues:")
        for chain in model:
            for res in chain:
                if res.id[0] not in (" ", "W"):
                    print(f"       Chain {chain.id} | {res.get_resname().strip()}")
        return []

    if verbose:
        print(f"[SBP] Ligand '{ligand_resname}': {len(lig_atoms)} atoms | Protein: {len(prot_atoms)} atoms")

    from Bio.PDB import NeighborSearch

    features = []
    search   = NeighborSearch(prot_atoms)

    # Can the ligand's own hydrogens be used to tell donors from acceptors?
    lig_hydrogens = [a for a in lig_atoms
                     if (a.element or a.get_name().strip()[:1]).strip().upper() == "H"]
    if verbose and not lig_hydrogens:
        print("[SBP] Note: the ligand carries no explicit hydrogens, so this "
              "fallback cannot tell a donor from an acceptor. Polar contacts "
              "are reported as both roles; install PLIP for a resolved "
              "assignment.")

    for la in lig_atoms:
        la_elem  = (la.element or la.get_name().strip()[:1]).strip().upper()
        la_coord = tuple(np.round(la.get_vector().get_array(), 3))

        if la_elem in _HBOND_ATOMS:
            partners = [pa for pa in search.search(la.coord, _HBOND_CUTOFF)
                        if (pa.element or pa.get_name().strip()[:1]).strip().upper()
                        in _HBOND_ATOMS]
            if partners:
                d = min(float(la - pa) for pa in partners)
                for ftype in _hbond_roles(la, lig_hydrogens):
                    features.append({
                        "type"  : ftype,
                        "coords": la_coord,
                        "label" : f"{ftype} (fallback, {la.get_name().strip()}, "
                                  f"d={d:.2f}Å)",
                    })

        elif la_elem in _HYDROPH_ELEMS:
            partners = [pa for pa in search.search(la.coord, _HYDROPH_CUTOFF)
                        if (pa.element or pa.get_name().strip()[:1]).strip().upper()
                        in _HYDROPH_ELEMS]
            if partners:
                d = min(float(la - pa) for pa in partners)
                features.append({
                    "type"  : "HYDROPHOBIC",
                    "coords": la_coord,
                    "label" : f"Hydrophobic (fallback, {la.get_name().strip()}, "
                              f"d={d:.2f}Å)",
                })

    return features


def _hbond_roles(atom, lig_hydrogens: list) -> List[str]:
    """
    Which H-bond roles a polar ligand atom can play.

    Element alone does not decide this: a pyridine nitrogen and a tertiary
    amine accept and cannot donate, a hydroxyl oxygen does both, and a
    carbonyl oxygen only accepts. When the structure carries explicit
    hydrogens the question is settled by looking for one covalently bound to
    the atom. When it does not — the common case for a crystal structure —
    both roles are returned, which keeps the ambiguity visible instead of
    resolving it by a coin flip on the element.
    """
    if not lig_hydrogens:
        return ["DONOR", "ACCEPTOR"]

    centre = atom.get_vector().get_array()
    has_h = any(
        float(np.linalg.norm(h.get_vector().get_array() - centre)) <= _XH_BOND_MAX
        for h in lig_hydrogens
    )
    elem = (atom.element or atom.get_name().strip()[:1]).strip().upper()
    if has_h:
        # An N-H or O-H donates; an oxygen with a hydrogen (hydroxyl, water)
        # still has a lone pair and accepts as well.
        return ["DONOR", "ACCEPTOR"] if elem == "O" else ["DONOR"]
    return ["ACCEPTOR"]
