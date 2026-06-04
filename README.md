# pharmaco_suite

Herramientas de línea de comandos para modelado de farmacóforos y diseño de novo molecular.

**Autor:** Eduardo Cubillos-Llantén  
**Programa:** Bioinformática — Biomedicina y Biofarmacéutica, Universidad de Talca

---

## Estructura

```
pharmaco_suite/
├── pharmacophore_cli/   Modelado de farmacóforos (SBP, LBP, RBP, consenso)
└── denovo_cli/          Diseño de novo molecular (LBP multi-ligando → GA SELFIES → SDF/PDBQT)
```

---

## Instalación

```bash
# Crear entorno conda (recomendado)
conda create -n pharmaco python=3.11 -y
conda activate pharmaco

# Instalar dependencias
bash install.sh
```

O manualmente:

```bash
conda install -c conda-forge rdkit -y
pip install selfies biopython requests numpy pandas py3Dmol lxml
pip install meeko          # opcional, para preparar PDBQT
pip install plip --no-deps # opcional, para SBP mejorado
```

---

## pharmacophore_cli — Modelado de Farmacóforos

Genera modelos SBP (basado en estructura), LBP (basado en ligando), RBP (basado en receptor) y un modelo consenso a partir de complejos cristalográficos y SMILES.

### Uso rápido

```bash
cd pharmacophore_cli

# Pipeline completo: descarga PDB, genera SBP+LBP+RBP+consenso
python pharmacophore.py all \
  --pdb 1HXW \
  --ligand RIT \
  --smiles "CC(C)C1=NC(=CS1)CN(C)C(=O)..." \
  --outdir resultados/

# Solo SBP con PDB local
python pharmacophore.py sbp --pdb-file mi_proteina.pdb --ligand LIG

# LBP desde múltiples SMILES (separar con |)
python pharmacophore.py lbp --smiles "SMILES1|SMILES2|SMILES3"

# LBP desde archivo
python pharmacophore.py lbp --smiles-file ligandos.smi

# RBP con pH personalizado
python pharmacophore.py rbp --pdb 1HXW --ligand RIT --ph 6.5

# Consenso desde PDBs ya generados
python pharmacophore.py consensus \
  --sbp pharmacophore_sbp.pdb \
  --lbp pharmacophore_lbp.pdb \
  --rbp pharmacophore_rbp.pdb
```

### Subcomandos

| Subcomando | Descripción |
|---|---|
| `sbp` | Farmacóforo Basado en Estructura (PLIP + fallback BioPython) |
| `lbp` | Farmacóforo Basado en Ligando (RDKit, uno o múltiples SMILES) |
| `rbp` | Farmacóforo Basado en Receptor (pH 7.4, sitio activo) |
| `all` | Ejecutar SBP + LBP + RBP + consenso automáticamente |
| `consensus` | Cargar modelos PDB ya generados y construir el consenso |

### Parámetros de clustering

Por defecto usa radios basados en literatura (Wolber & Langer, 2005):

| Tipo | Radio default | Argumento |
|---|---|---|
| DONOR / ACCEPTOR | 1.5 Å | `--radius-donor`, `--radius-acceptor` |
| HYDROPHOBIC | 2.5 Å | `--radius-hydro` |
| POS/NEG IONIZABLE | 2.0 Å | `--radius-pos`, `--radius-neg` |
| Global (todos) | — | `--cluster-radius` |

### Archivos generados

- `pharmacophore_sbp.pdb` — features SBP clusterizados
- `pharmacophore_lbp.pdb` — features LBP clusterizados
- `pharmacophore_rbp.pdb` — features RBP clusterizados
- `pharmacophore_consensus.pdb` — modelo consenso (B-factor = nivel de confianza)
- `pharmacophore_overlap.html` — visualización 3D interactiva (abrir en navegador)
- `pharmacophore_consensus.html` — visualización del consenso

---

## denovo_cli — Diseño de Novo Molecular

Genera moléculas candidatas nuevas a partir de ligandos activos conocidos, usando un farmacóforo LBP multi-ligando como guía y un Algoritmo Genético SELFIES para la generación.

### Uso rápido

```bash
cd denovo_cli

# Pipeline completo desde ChEMBL (usar ChEMBL ID del target)
# Buscar el ID en https://www.ebi.ac.uk/chembl/ → Target → copiar CHEMBL ID
python denovo.py run --target "CHEMBL247" --n 20 --outdir resultados/

# Pipeline desde archivo propio de ligandos
python denovo.py run --file mis_ligandos.smi --n 20 --outdir resultados/

# Con farmacóforo PDB del pharmacophore_cli (scoring 3D adicional)
python denovo.py run \
  --target "HIV protease" \
  --n 20 \
  --pharmacophore ../pharmacophore_cli/resultados/pharmacophore_lbp.pdb \
  --outdir resultados/

# Solo descargar ligandos activos desde ChEMBL (usar ID, no nombre libre)
python denovo.py fetch --target "CHEMBL247" --n 50 --out ligands.smi

# Filtrar ADMET de un SDF existente
python denovo.py filter --sdf moleculas.sdf --outdir filtradas/
```

### Pipeline completo (subcomando `run`)

```
Ligandos activos (ChEMBL o archivo)
    │
    ▼
Perfil LBP multi-ligando
(features presentes en ≥50% de los ligandos)
    │
    ▼
Algoritmo Genético SELFIES
(N generaciones, población de tamaño P)
    │
    ▼
Filtro ADMET (Lipinski, QED ≥ 0.3, SA ≤ 5, PSA ≤ 140 Å²)
    │
    ▼
Filtro de diversidad (Tanimoto < 0.85 entre candidatas)
    │
    ▼
Top N candidatas → SDF 3D + PDBQT (si Meeko instalado) + CSV métricas
```

### Subcomandos

| Subcomando | Descripción |
|---|---|
| `run` | Pipeline completo: ligandos → LBP → GA → ADMET → SDF + PDBQT |
| `fetch` | Descargar ligandos activos desde ChEMBL o cargar desde archivo |
| `filter` | Aplicar filtro ADMET a un SDF existente |

### Parámetros clave de `run`

| Argumento | Default | Descripción |
|---|---|---|
| `--target` | — | ChEMBL ID del target (ej. "CHEMBL247"). Buscar en ebi.ac.uk/chembl → Target |
| `--file` | — | Archivo local de ligandos (.smi, .sdf, .csv) |
| `--n` | 20 | Número de moléculas a generar |
| `--n-ligands` | 50 | Ligandos de entrenamiento a usar |
| `--pharmacophore` | — | PDB de farmacóforo para scoring 3D adicional |
| `--pop-size` | 60 | Tamaño de la población del GA |
| `--n-gen` | 50 | Generaciones del GA |
| `--diversity` | 0.85 | Tanimoto máximo entre candidatas finales |

### Archivos generados

- `candidates.sdf` — moléculas con conformaciones 3D y metadatos ADMET
- `candidates.pdbqt` — listos para AutoDock Vina (si Meeko instalado)
- `candidates_admet.csv` — tabla con todas las métricas
- `ligands_training.smi` — ligandos de entrenamiento usados
- `PREPARACION_DOCKING.md` — instrucciones para preparar el docking

---

## Flujo completo para el proyecto

```bash
# 1. Generar farmacóforos SBP + LBP
python pharmacophore_cli/pharmacophore.py all \
  --pdb 1HXW --ligand RIT \
  --smiles "SMILES_del_ligando" \
  --outdir farmacoforos/

# 2. Generar 20 moléculas nuevas usando el LBP como guía
python denovo_cli/denovo.py run \
  --target "HIV protease" \
  --n 20 \
  --pharmacophore farmacoforos/pharmacophore_lbp.pdb \
  --outdir candidatas/

# 3. Preparar receptor para docking
prepare_receptor4.py -r 1HXW_proteina.pdb -o 1HXW.pdbqt

# 4. Correr docking con AutoDock Vina
vina --receptor 1HXW.pdbqt \
     --ligands candidatas/candidates.pdbqt \
     --center_x -5 --center_y 4 --center_z 15 \
     --size_x 25 --size_y 25 --size_z 25 \
     --exhaustiveness 16
```

---

## Referencias

- Wolber & Langer (2005) *J. Chem. Inf. Model.* 45, 160–169 — radios de clustering
- Krenn et al. (2020) *Machine Learning: Science and Technology* 1, 045024 — SELFIES
- Lipinski et al. (2001) *Adv. Drug Deliv. Rev.* 46, 3–26 — Rule of Five
- Bickerton et al. (2012) *Nature Chemistry* 4, 90–98 — QED
- Ertl & Schuffenhauer (2009) *J. Cheminform.* 1, 8 — SA Score
- Salentin et al. (2015) *Nucleic Acids Res.* 43, W443–W447 — PLIP
- Eberhardt et al. (2021) *J. Chem. Inf. Model.* 61, 3891–3898 — AutoDock Vina
