# Tests

Each file checks one claim the code makes about itself. They need nothing but
the runtime dependencies in `requirements.txt` — no network, no PDB downloads,
no reference data — because every structure is built on the spot with RDKit and
written out as a PDB. The whole set runs in well under a minute.

```bash
python tests/run_all.py          # or: python tests/test_fit.py
```

| Test | What it pins down |
|---|---|
| `test_lbp_align.py` | The LBP lands in receptor space when the crystal ligand's atom order does not match RDKit's, and the alignment is refused rather than silently skipped when the ligand is absent or is a different molecule. |
| `test_fit.py` | The 3D pharmacophore fit gives a molecule its own pharmacophore a score of 1.0, ranks a decoy below it, and returns the identical score after the model is rotated and translated. |
| `test_rbp.py` | Both halves of the HIV-1 protease dyad get different protonation states, GLY produces no hydrophobic feature, and hydrophobic features sit on side chains rather than on the Cα. |
| `test_sbp.py` | The BioPython fallback reads H-bond roles from the ligand's hydrogens when the structure has them, and reports both roles when it does not, instead of guessing from the element. |
| `test_alerts.py` | A quinone is flagged as PAINS, a bare pyrrole is not, and the alerts no longer depend on which enantiomer was drawn. Each case is compared against the six hand-written SMARTS this replaced. |
| `test_ga.py` | Filters passed by the caller actually bind on the generated candidates, and the GA produces candidates at all. |

A test fails by raising, so a silent run is a pass. The assertions carry the
numbers they expect, so a test that starts failing says what changed.
