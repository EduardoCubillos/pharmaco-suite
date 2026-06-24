#!/bin/bash
# install.sh — Install dependencies for pharmaco_suite
# Usage: bash install.sh
# Recommended: run inside a conda environment (conda create -n pharmaco python=3.11)

set -e

echo "=== pharmaco_suite — dependency installation ==="
echo ""

# RDKit: best installed via conda-forge to avoid compilation issues
if command -v conda &> /dev/null; then
    echo "[1/4] Installing RDKit via conda-forge..."
    conda install -c conda-forge rdkit -y
else
    echo "[1/4] Installing RDKit via pip..."
    pip install rdkit
fi

echo "[2/4] Installing Python dependencies..."
pip install selfies biopython requests numpy pandas py3Dmol lxml scipy

echo "[3/4] Installing PLIP (recommended for structure-based pharmacophores)..."
# PLIP depends on openbabel. If conda is available, install openbabel there first
# so pip doesn't have to compile it from source (which requires SWIG and fails often).
if command -v conda &> /dev/null; then
    conda install -c conda-forge openbabel -y && \
        pip install plip && \
        echo "  PLIP installed with conda openbabel." || \
        echo "  PLIP not installed — SBP will use BioPython distance fallback."
else
    pip install openbabel-wheel plip && \
        echo "  PLIP installed with openbabel-wheel." || \
        echo "  PLIP not installed — SBP will use BioPython distance fallback."
fi

echo "[4/4] Installing meeko (optional — needed for PDBQT export for docking)..."
pip install meeko || echo "  meeko not installed — skip if you don't need docking output."

echo ""
echo "=== Installation complete ==="
echo ""
echo "Quick check:"
python3 -c "from rdkit import Chem; print('  RDKit      OK')"
python3 -c "import selfies;         print('  SELFIES    OK')"
python3 -c "from Bio.PDB import PDBParser; print('  BioPython  OK')"
python3 -c "import plip;            print('  PLIP       OK')" 2>/dev/null || echo "  PLIP       not available (will use fallback)"
echo ""
echo "Usage:"
echo "  python pharmacophore_cli/pharmacophore.py --help"
echo "  python denovo_cli/denovo.py --help"
