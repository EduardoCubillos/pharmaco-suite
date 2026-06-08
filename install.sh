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

echo "[3/4] Instalando PLIP y openbabel (necesarios para SBP mejorado)..."
# openbabel es dependencia clave de PLIP — instalar primero vía conda si está disponible
if command -v conda &> /dev/null; then
    conda install -c conda-forge openbabel -y && \
        pip install plip && \
        echo "  PLIP instalado con openbabel (conda)" || \
        echo "  PLIP no instalado — se usará fallback BioPython para SBP"
else
    pip install openbabel-wheel plip && \
        echo "  PLIP instalado con openbabel-wheel" || \
        echo "  PLIP no instalado — se usará fallback BioPython para SBP"
fi

echo "[4/4] Instalando meeko (opcional, para preparar PDBQT para docking)..."
pip install meeko || echo "  meeko no instalado — saltar si no necesitas docking"

echo ""
echo "=== Instalación completada ==="
echo ""
echo "Verificación rápida:"
python3 -c "from rdkit import Chem; print('  RDKit OK')"
python3 -c "import selfies; print('  SELFIES OK')"
python3 -c "from Bio.PDB import PDBParser; print('  BioPython OK')"
python3 -c "import plip; print('  PLIP OK')" 2>/dev/null || echo "  PLIP no disponible (se usará fallback)"
echo ""
echo "Para usar:"
echo "  python pharmacophore_cli/pharmacophore.py --help"
echo "  python denovo_cli/denovo.py --help"
