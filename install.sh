#!/bin/bash
# install.sh — Instala dependencias del pharmaco_suite
# Uso: bash install.sh
# Recomendado: ejecutar dentro de un entorno conda o venv

set -e

echo "=== pharmaco_suite — Instalación de dependencias ==="
echo ""

# RDKit (mejor desde conda-forge)
if command -v conda &> /dev/null; then
    echo "[1/3] Instalando RDKit via conda-forge..."
    conda install -c conda-forge rdkit -y
else
    echo "[1/3] Instalando RDKit via pip..."
    pip install rdkit
fi

echo "[2/3] Instalando dependencias Python..."
pip install selfies biopython requests numpy pandas py3Dmol lxml scipy meeko

echo "[3/3] Intentando instalar PLIP (opcional, para SBP mejorado)..."
pip install plip --no-deps || echo "  PLIP no instalado — se usará fallback BioPython para SBP"

echo "[Opcional] Para preparar PDBQT (necesario para docking con Vina):"
echo "  pip install meeko"

echo ""
echo "=== Instalación completada ==="
echo ""
echo "Verificación rápida:"
python3 -c "from rdkit import Chem; print('  RDKit OK')"
python3 -c "import selfies; print('  SELFIES OK')"
python3 -c "from Bio.PDB import PDBParser; print('  BioPython OK')"
echo ""
echo "Para usar:"
echo "  python pharmacophore_cli/pharmacophore.py --help"
echo "  python denovo_cli/denovo.py --help"
