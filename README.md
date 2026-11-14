# pharmaco_suite

A command-line toolkit for 3D pharmacophore modeling and structure-guided de novo molecular design. Built around the idea that a pharmacophore model is only as reliable as the evidence behind it — so instead of picking one approach, this tool runs three independently and lets them vote.

**Author:** Eduardo Cubillos

---

## What it does

`pharmaco_suite` has two independent modules:

```
pharmaco_suite/
├── pharmacophore_cli/   3D pharmacophore generation (SBP, LBP, RBP, consensus)
└── denovo_cli/          De novo molecule design (multi-ligand LBP → genetic algorithm → SDF)
```

---

## Installation

```bash
# Recommended: create a dedicated conda environment
conda create -n pharmaco python=3.11 -y
conda activate pharmaco

# Install all dependencies
bash install.sh
```

Manual install:

```bash
conda install -c conda-forge rdkit -y
pip install selfies biopython requests numpy pandas py3Dmol lxml scipy
pip install plip          # optional but recommended for structure-based pharmacophores
pip install meeko         # optional, needed for PDBQT export (docking)
```

---

## pharmacophore_cli — Pharmacophore Modeling

Generates structure-based (SBP), ligand-based (LBP), and receptor-based (RBP) pharmacophore models, then builds a weighted consensus from the three perspectives.

### Quick start

```bash
cd pharmacophore_cli

# Full pipeline: download structure, run SBP + LBP + RBP, build consensus
python pharmacophore.py all \
  --pdb 1HXW \
  --ligand RIT \
  --smiles "CC(C)C1=NC(=CS1)CN(C)C(=O)N[C@@H](C(C)C)C(=O)N[C@@H](CC2=CC=CC=C2)C[C@@H]([C@H](CC3=CC=CC=C3)NC(=O)OCC4=CN=CS4)O" \
  --outdir results/

# SBP only, from a local PDB file
python pharmacophore.py sbp --pdb-file my_protein.pdb --ligand LIG

# LBP from multiple SMILES (pipe-separated or from file)
python pharmacophore.py lbp --smiles "SMILES1|SMILES2|SMILES3"
python pharmacophore.py lbp --smiles-file active_compounds.smi

# RBP with custom binding site radius and pH
python pharmacophore.py rbp --pdb 1HXW --ligand RIT --site-cutoff 5.0 --ph 6.5

# Build consensus from previously generated PDB files
python pharmacophore.py consensus \
  --sbp pharmacophore_sbp.pdb \
  --lbp pharmacophore_lbp.pdb \
  --rbp pharmacophore_rbp.pdb
```

### Subcommands

| Subcommand | Description |
|---|---|
| `sbp` | Structure-Based Pharmacophore — uses PLIP to extract protein-ligand interactions from a crystal structure, falls back to BioPython distance analysis if PLIP is unavailable |
| `lbp` | Ligand-Based Pharmacophore — uses RDKit feature detection on one or more ligand SMILES; multi-ligand mode keeps features present in ≥50% of compounds |
| `rbp` | Receptor-Based Pharmacophore — maps the binding site residues to pharmacophoric features at physiological pH, accounting for protonation states |
| `all` | Runs all three models and builds the consensus automatically |
| `consensus` | Loads previously generated PDB files and builds the consensus from them |

### Clustering radii

Default radii follow Wolber & Langer (2005), which distinguish directional from non-directional interactions:

| Feature type | Default radius | Flag |
|---|---|---|
| DONOR / ACCEPTOR | 1.5 Å | `--radius-donor`, `--radius-acceptor` |
| HYDROPHOBIC | 2.5 Å | `--radius-hydro` |
| POS/NEG IONIZABLE | 2.0 Å | `--radius-pos`, `--radius-neg` |
| All types (global override) | — | `--cluster-radius` |

---

### Consensus model

The consensus is not a visual overlay of three models. It is a cross-validation process:

**Step 1 — SBP + LBP fusion (ligand perspective):**
Features of the same type (e.g., DONOR) within `--consensus-radius` Å of each other are merged into a single centroid. A feature present in both SBP and LBP gets weight ≥ 2.

**Step 2 — RBP validation (receptor perspective):**
For each merged ligand feature, the algorithm searches the RBP model for a chemically complementary feature in the same region (DONOR ↔ ACCEPTOR, HYDROPHOBIC ↔ HYDROPHOBIC, etc.). A match raises the weight to 3.

**Step 3 — Confidence filter:**
Features are retained based on `--min-consensus-weight` (default: 1 = keep everything).

#### Confidence levels

| Weight | Label | Meaning | Sphere in 3D viewer |
|--------|-------|---------|---------------------|
| ★★★ (3) | Triple consensus | SBP + LBP + RBP all agree | large, nearly opaque |
| ★★☆ (2) | Double consensus | 2 of 3 models agree | medium, semi-transparent |
| ★☆☆ (1) | Single model | Only 1 model — included by default | small, faint |

Higher-weight features are the most reliable for virtual screening (Baroni et al., 2007).

**Why keep single-model features (weight ★☆☆)?**

A feature that appears only in the SBP can represent a real crystal interaction detected by PLIP that the LBP simply missed — either because RDKit did not encode that group well from the SMILES, or because the interaction geometry doesn't overlap cleanly with the ligand atom position. Discarding it automatically would mean throwing away experimental evidence. The low weight signals lower confidence; the researcher decides what threshold to use for their application.

```bash
# Default: keep everything (★☆☆ to ★★★) — preserves potential crystal evidence
python pharmacophore.py all ...

# Strict screening: only features supported by ≥2 models
python pharmacophore.py all ... --min-consensus-weight 2

# Maximum confidence: triple consensus only
python pharmacophore.py all ... --min-consensus-weight 3
```

---

### 2D pharmacophore graph

The `all` subcommand also generates a graph representation of the consensus model:

- **Nodes** — pharmacophoric features, colored by type, sized proportionally to consensus weight
- **Edges** — minimum spanning tree (MST) of 3D pairwise distances; reduces N×(N-1)/2 connections to N-1, showing the essential topology without visual noise
- **Edge labels** — distances in Å on the 5 shortest edges; MST angles in the bottom-left inset
- **Interactive HTML** — drag nodes, hover edges for distances (vis.js, no external dependencies at runtime)

---

### Output files

```
pharmacophore_sbp.pdb              SBP features (clustered)
pharmacophore_lbp.pdb              LBP features (clustered)
pharmacophore_rbp.pdb              RBP features (clustered)
pharmacophore_consensus.pdb        Consensus model (B-factor = confidence level)
pharmacophore_sbp.html             Interactive 3D viewer — SBP
pharmacophore_lbp.html             Interactive 3D viewer — LBP
pharmacophore_rbp.html             Interactive 3D viewer — RBP
pharmacophore_overlap.html         3D overlay of all three clustered models
pharmacophore_consensus.html       3D viewer — consensus model
pharmacophore_consensus_graph.png  2D pharmacophore graph (static)
pharmacophore_consensus_graph.pdf  2D pharmacophore graph (vector)
pharmacophore_consensus_graph.html 2D pharmacophore graph (interactive)
plip_output/                       PLIP XML reports and protonated PDB (if PLIP available)
```

---

## denovo_cli — De Novo Molecular Design

Generates new candidate molecules from a set of known active compounds, using a multi-ligand LBP profile as a pharmacophoric fitness function and a SELFIES-based genetic algorithm for structure generation.

### Quick start

```bash
cd denovo_cli

# Full pipeline from ChEMBL (use target ChEMBL ID for reliable results)
# Find the ID at https://www.ebi.ac.uk/chembl/ → Target Search → copy CHEMBL ID
python denovo.py run --target "CHEMBL247" --n 20 --outdir results/

# Full pipeline from your own ligand file
python denovo.py run --file active_ligands.smi --n 20 --outdir results/

# With an LBP pharmacophore for additional 3D scoring
python denovo.py run \
  --target "CHEMBL247" \
  --n 20 \
  --pharmacophore ../pharmacophore_cli/results/pharmacophore_lbp.pdb \
  --outdir results/

# Download active compounds from ChEMBL (use ChEMBL ID)
python denovo.py fetch --target "CHEMBL247" --n 50 --out ligands.smi

# Filter ADMET properties on an existing SDF
python denovo.py filter --sdf molecules.sdf --outdir filtered/
```

### Pipeline overview

```
Active ligands (ChEMBL or local file)
        │
        ▼
Multi-ligand LBP profile
(features in ≥50% of training compounds)
        │
        ▼
SELFIES Genetic Algorithm
(N generations, population size P)
        │
        ▼
ADMET filter (Lipinski, QED ≥ 0.3, SA ≤ 5, PSA ≤ 140 Å²)
        │
        ▼
Diversity filter (Tanimoto < 0.85 between candidates)
        │
        ▼
Top N candidates → 3D SDF + PDBQT (if Meeko installed) + CSV metrics
```

### How the 3D pharmacophore fit is scored

With `--pharmacophore`, each candidate is scored on how much of the model it
can satisfy at once. The model lives in the receptor's coordinate frame and a
generated candidate lives in whatever frame its embedding produced, so the two
cannot be compared where they stand. The candidate is aligned onto the model
first: a conformer ensemble is generated, superpositions are proposed from
anchors whose internal distances agree with the model's, and under each
superposition every model point is matched to the nearest unused feature of its
own type inside the tolerance sphere.

The score is the best weighted coverage over the ensemble, so it depends only
on internal geometry and is invariant to rotating or translating either input.
Partial matches count as such: three results columns report the score, how many
points were matched out of how many, and the fit RMSD over the matched points.

```
pharma_score_3d,fit_points,fit_rmsd
0.875,4/5,0.879
```

### Subcommands

| Subcommand | Description |
|---|---|
| `run` | Full pipeline: ligands → LBP → GA → ADMET → SDF + PDBQT |
| `fetch` | Download active compounds from ChEMBL or load from local file |
| `filter` | Apply ADMET filters to an existing SDF |

### Key parameters for `run`

| Argument | Default | Description |
|---|---|---|
| `--target` | — | ChEMBL target ID (e.g., "CHEMBL247"). Find at ebi.ac.uk/chembl → Target |
| `--file` | — | Local ligand file (.smi, .sdf, .csv) |
| `--n` | 20 | Number of candidate molecules to generate |
| `--n-ligands` | 50 | Max training ligands to use |
| `--pharmacophore` | — | PDB pharmacophore file for additional 3D scoring |
| `--pop-size` | 60 | GA population size |
| `--n-gen` | 50 | GA generations |
| `--diversity` | 0.85 | Max Tanimoto similarity between final candidates |
| `--fit-tolerance` | 1.5 | Tolerance sphere radius (Å) when fitting candidates to the pharmacophore |

### Output files

```
candidates.sdf           Molecules with 3D conformations and ADMET metadata
candidates.pdbqt         Ready for AutoDock Vina (if Meeko installed)
candidates_admet.csv     Full metrics table
ligands_training.smi     Training ligands used
DOCKING_PREPARATION.md   Docking preparation instructions
```

---

## End-to-end example: HIV-1 protease inhibitor design

```bash
# 1. Generate pharmacophore models from the Ritonavir co-crystal structure (1HXW)
python pharmacophore_cli/pharmacophore.py all \
  --pdb 1HXW \
  --ligand RIT \
  --smiles "CC(C)C1=NC(=CS1)CN(C)C(=O)N[C@@H](C(C)C)C(=O)N[C@@H](CC2=CC=CC=C2)C[C@@H]([C@H](CC3=CC=CC=C3)NC(=O)OCC4=CN=CS4)O" \
  --outdir pharmacophores/

# Open the consensus viewer to inspect the model
open pharmacophores/pharmacophore_consensus.html

# 2. Design 20 new molecules guided by the LBP profile
python denovo_cli/denovo.py run \
  --target "CHEMBL247" \
  --n 20 \
  --pharmacophore pharmacophores/pharmacophore_lbp.pdb \
  --outdir candidates/

# 3. Prepare receptor for docking
prepare_receptor4.py -r 1HXW_protein.pdb -o 1HXW.pdbqt

# 4. Run AutoDock Vina
vina --receptor 1HXW.pdbqt \
     --ligands candidates/candidates.pdbqt \
     --center_x -5 --center_y 4 --center_z 15 \
     --size_x 25 --size_y 25 --size_z 25 \
     --exhaustiveness 16
```

---

## License

MIT — see [LICENSE](LICENSE). If you use this in published work, a citation to
the repository is appreciated.

---

## References

- Wolber & Langer (2005) *J. Chem. Inf. Model.* 45, 160–169 — LigandScout
- Wolber, Dornhofer & Langer (2006) *J. Comput.-Aided Mol. Des.* 20, 773–788 — overlay of small molecules onto 3D pharmacophores
- Dixon et al. (2006) *J. Comput.-Aided Mol. Des.* 20, 647–671 — PHASE
- Baroni et al. (2007) *J. Chem. Inf. Model.* 47, 279–294 — FLAP, a common reference frame for proteins and ligands
- Sanders et al. (2012) *J. Chem. Inf. Model.* 52, 1607–1620 — comparison of pharmacophore screening tools
- Koes & Camacho (2011) *J. Chem. Inf. Model.* 51, 1307–1314 — Pharmer
- Salentin et al. (2015) *Nucleic Acids Res.* 43, W443–W447 — PLIP
- Riniker & Landrum (2015) *J. Chem. Inf. Model.* 55, 2562–2574 — ETKDG conformer generation
- Sliwoski et al. (2014) *Pharmacol. Rev.* 66, 334–395 — receptor-based pharmacophores
- Baell & Holloway (2010) *J. Med. Chem.* 53, 2719–2740 — PAINS
- Brenk et al. (2008) *ChemMedChem* 3, 435–444 — structural alerts
- Krenn et al. (2020) *Mach. Learn.: Sci. Technol.* 1, 045024 — SELFIES
- Lipinski et al. (2001) *Adv. Drug Deliv. Rev.* 46, 3–26 — Rule of Five
- Bickerton et al. (2012) *Nature Chemistry* 4, 90–98 — QED
- Ertl & Schuffenhauer (2009) *J. Cheminform.* 1, 8 — SA Score
- Eberhardt et al. (2021) *J. Chem. Inf. Model.* 61, 3891–3898 — AutoDock Vina
