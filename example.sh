#!/bin/bash
# example.sh — End-to-end example: HIV-1 protease pharmacophore modeling
#
# This script walks through the full pharmaco_suite pipeline using the
# HIV-1 protease / Ritonavir co-crystal structure (PDB: 1HXW) as a case study.
#
# What it does:
#   1. Generates SBP + LBP + RBP pharmacophore models from 1HXW / RIT
#   2. Builds a weighted consensus model
#   3. Generates an interactive 2D pharmacophore graph
#   4. Designs 5 new candidate molecules (small run — good for testing)
#
# Usage: bash example.sh
# Expected runtime: ~2 min (downloads 1HXW on first run, then runs the pipeline)

set -e

SUITE_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTDIR="$SUITE_DIR/example_output"

# Ritonavir (HIV-1 protease inhibitor, PDB ligand code: RIT)
SMILES_RITONAVIR="CC(C)C1=NC(=CS1)CN(C)C(=O)N[C@@H](C(C)C)C(=O)N[C@@H](CC2=CC=CC=C2)C[C@@H]([C@H](CC3=CC=CC=C3)NC(=O)OCC4=CN=CS4)O"

GREEN=$'\033[92m'
CYAN=$'\033[96m'
BOLD=$'\033[1m'
RESET=$'\033[0m'

mkdir -p "$OUTDIR"

echo ""
echo -e "${BOLD}══════════════════════════════════════════════════════${RESET}"
echo -e "${BOLD}   pharmaco_suite — HIV-1 Protease Example${RESET}"
echo -e "${BOLD}══════════════════════════════════════════════════════${RESET}"
echo ""
echo "  Structure: 1HXW (HIV-1 protease + Ritonavir co-crystal)"
echo "  Ligand:    RIT  (Ritonavir, an approved HIV protease inhibitor)"
echo "  Output:    $OUTDIR"
echo ""

# ─────────────────────────────────────────────────────────────────────────────
echo -e "${CYAN}[Step 1] Full pharmacophore pipeline: SBP + LBP + RBP + consensus${RESET}"
echo "  (downloads 1HXW if not already cached — ~200 KB)"
echo ""

python "$SUITE_DIR/pharmacophore_cli/pharmacophore.py" all \
    --pdb 1HXW \
    --ligand RIT \
    --smiles "$SMILES_RITONAVIR" \
    --outdir "$OUTDIR/pharmacophore" \
    --min-consensus-weight 1

echo ""
echo -e "  ${GREEN}Open the consensus viewer:${RESET}"
echo "    $OUTDIR/pharmacophore/pharmacophore_consensus.html"
echo ""
echo -e "  ${GREEN}Open the interactive graph:${RESET}"
echo "    $OUTDIR/pharmacophore/pharmacophore_consensus_graph.html"
echo ""

# ─────────────────────────────────────────────────────────────────────────────
echo -e "${CYAN}[Step 2] De novo design — generate 5 new candidate molecules${RESET}"
echo "  (uses a small population for speed; increase --pop-size and --n-gen for real use)"
echo ""

# Multi-ligand LBP: include a few related HIV protease inhibitors
cat > "$OUTDIR/hiv_inhibitors.smi" << 'SMI'
# HIV-1 protease inhibitors for multi-ligand LBP
CC(C)C1=NC(=CS1)CN(C)C(=O)NC(C(C)C)C(=O)NCC2=CC=CC=C2	Ritonavir_core
CC(C)[C@@H](NC(=O)[C@H](Cc1ccccc1)NC(=O)c1cc(NS(=O)(=O)c2ccc(N)cc2)ccc1OCC1CCCCO1)C(=O)N[C@@H](Cc1ccccc1)CC(O)Cc1ccccc1	Lopinavir_like
OC(CN1CCCCC1)(Cc1ccc(F)cc1)c1ccc(F)cc1	Hydrophobic_fragment
SMI

python "$SUITE_DIR/denovo_cli/denovo.py" run \
    --file "$OUTDIR/hiv_inhibitors.smi" \
    --n 5 \
    --n-gen 20 \
    --pop-size 30 \
    --pharmacophore "$OUTDIR/pharmacophore/pharmacophore_lbp.pdb" \
    --outdir "$OUTDIR/candidates"

echo ""
echo -e "${BOLD}══════════════════════════════════════════════════════${RESET}"
echo -e "${BOLD}${GREEN}  Example complete${RESET}"
echo -e "${BOLD}══════════════════════════════════════════════════════${RESET}"
echo ""
echo "Output files:"
echo ""
echo "  Pharmacophore models:"
echo "    $OUTDIR/pharmacophore/pharmacophore_sbp.html"
echo "    $OUTDIR/pharmacophore/pharmacophore_lbp.html"
echo "    $OUTDIR/pharmacophore/pharmacophore_rbp.html"
echo "    $OUTDIR/pharmacophore/pharmacophore_consensus.html"
echo "    $OUTDIR/pharmacophore/pharmacophore_consensus_graph.html"
echo ""
echo "  PLIP reports (if PLIP is installed):"
echo "    $OUTDIR/pharmacophore/plip_output/"
echo ""
echo "  Candidate molecules:"
echo "    $OUTDIR/candidates/candidates.sdf"
echo "    $OUTDIR/candidates/candidates_admet.csv"
echo ""
echo "  Next: prepare a receptor PDB for docking and run AutoDock Vina."
echo "  See README.md for the full docking workflow."
echo ""
