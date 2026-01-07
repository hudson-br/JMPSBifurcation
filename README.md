# JMPS (revision) — Code + data package

**Paper:** *Damage localization as a bifurcation phenomenon and the resulting fracture patterns in soft materials*  
**Authors:** Hudson Borja da Rocha & Tal Cohen

This repository contains two complementary components:

1. **Bifurcation analysis (semi-analytical):**  
   Axisymmetric base state + circumferential bifurcation (Fourier mode \(n\)) using a symbolic derivation (SymPy) and a numerical ODE / boundary-condition solver. This generates the phase diagrams and critical thresholds reported in the paper.

2. **Finite element simulations (legacy FEniCS / `dolfin`):**  
   Stabilized mixed FE implementation for brittle fracture (plane stress). For the revision package, we include only the **reduced outputs needed to reproduce the paper figures** and omit large visualization files (`.h5`, `.xdmf`) because they are too large to distribute with the revision.

---

## Repository layout

```text
.
├── classes/
│   ├── symbolic_problem.py        # SymPy derivations (pushed-forward formulation used in SI notebook)
│   ├── equilibrium.py             # Base-state solver for α0(r) and p0(r)
│   ├── bifurcation_numeric.py     # Numeric bifurcation solver (matrix ODE + BC determinant)
│   ├── utils.py                   # Small helpers (I/O, plotting)
│   └── __init__.py
│
├── data/
│   ├── B_over_A/                  # Sweep outputs (JSON, plots) for varying B/A
│   ├── ell_over_A/                # Sweep outputs for varying ℓ/A
│   └── Gc_over_mu_ell/            # Sweep outputs for varying Gc/(μℓ)
│
├── FEM/
│   ├── mesh.py                    # Mesh generation using the gmsh Python API
│   ├── main-stabilized.py         # Legacy FEniCS (dolfin) stabilized FE solver
│   └── output/                    # Reduced FEM datasets included with the revision (see note below)
│
├── notebooks/
│   ├── data_analysis.ipynb
│   ├── data_analysis.html                         # HTML export (browser-viewable)
│   ├── SI_bifurcation_symbolic_walkthrough.ipynb
│   └── SI_bifurcation_symbolic_walkthrough.html   # HTML export (browser-viewable)│                                              
├── main_bifurcation.py            # Entry point to generate phase diagrams (writes to data/<sweep_type>/)
└── README.md
```

---

## Dependencies

### A) Bifurcation analysis + notebooks (pure Python)

You need a standard scientific Python stack:

- `numpy`
- `scipy`
- `sympy`
- `matplotlib`
- `jupyter` (for the notebooks)

Example install:
```bash
pip install numpy scipy sympy matplotlib jupyter
```

### B) FEM simulations (legacy FEniCS)

The FEM code is written for **legacy FEniCS** (imports `from dolfin import *`), not FEniCSx.

Typical requirements:

- legacy FEniCS / `dolfin`
- `gmsh` (Python API) — used in `FEM/mesh.py`
- `meshio` — used for mesh conversion inside `FEM/main-stabilized.py`
- `numpy`, `matplotlib`

Installation depends on OS; please follow the legacy FEniCS installation route appropriate for your system.

---

## Running the bifurcation phase diagrams

### 1) Generate sweep data (JSON + plots)

From the repository root:
```bash
python main_bifurcation.py
```

This script:
- builds the symbolic incremental problem (`SymbolicProblem`),
- solves the axisymmetric base state (`EquilibriumProblem`),
- computes the critical inner stretch \(\lambda_a^{crit}\) for each circumferential mode \(n\),
- sweeps one control parameter to generate phase diagrams:
  - `B_over_A` (thickness ratio)
  - `ell_over_A` (regularization length ratio)
  - `Gc_over_mu_ell` (dimensionless toughness \(Gc/(\mu \ell)\))
- saves results to `data/<sweep_type>/` as JSON and figure files.

**How to choose the sweep:**  
Edit the parameter block at the top of `main_bifurcation.py` (this project intentionally avoids `argparse`).

---

## Notebooks (recommended way to reproduce paper/SI figures)

### `notebooks/data_analysis.ipynb`

This notebook loads:
- phase-diagram outputs from `data/…/`
- reduced FEM outputs from `FEM/output/` (as provided)

and reproduces the plots used in the paper (pressure–stretch curves, critical points, phase-diagram summaries, etc.).  
Open and run:

```bash
jupyter lab
```

Then open:
- `notebooks/data_analysis.ipynb`

The notebook is organized so that each section explicitly states **which paper figure(s)** it corresponds to.

### `notebooks/SI_bifurcation_symbolic_walkthrough_improved.ipynb`

This notebook is the SI companion: it documents the bifurcation derivations **directly from the SymPy objects** in `classes/symbolic_problem.py`, so the SI equations do not drift from the implementation. It includes:

- model definitions (damage functions, energy, parameters),
- base-state ODEs (\(dp_0/dr\), \(\alpha_0''\)),
- incremental system construction and coefficient extraction (main-text matrix form),
- boundary-condition matrices.

---

## FEM notes (what is included / omitted)

### Included (revision package)

`FEM/output/` contains only the **reduced data files necessary to reproduce the figures** (e.g. pressure–stretch data, scalar diagnostics, small JSON/CSV summaries).

### Omitted

Large visualization outputs are not included (e.g. `.h5`, `.xdmf`) because they are too large for the revision package.

If you want the full visualization output:
1) install legacy FEniCS (`dolfin`) + dependencies,  
2) rerun the FEM scripts locally, and  
3) enable/restore any `.h5`/`.xdmf` output options inside `FEM/main-stabilized.py`.

---

## Running the FEM code (legacy FEniCS)

From the repository root:
```bash
python FEM/main-stabilized.py
```

Mesh generation is performed via the gmsh Python API (see `FEM/mesh.py`) and mesh conversion uses `meshio` (inside `FEM/main-stabilized.py`).  
As with the bifurcation script, the FEM script is configured by editing parameters near the top of the file.

---

## Data conventions

Phase-diagram outputs are stored under:
- `data/B_over_A/`
- `data/ell_over_A/`
- `data/Gc_over_mu_ell/`

Filenames encode key parameter values (e.g. `A=..._B=..._ell=..._Gc=..._mu=...`) for traceability.

---

## Contact

For questions about reproducing the figures or running the code, contact:  
Hudson Borja da Rocha at hudsonbr@mit.edu

## HTML versions of the notebooks

For convenience, we also provide **HTML-rendered versions** of each Jupyter notebook in the repository.  
These files allow the full content (markdown, equations, figures, and results) to be viewed directly in a standard web browser, **without requiring Jupyter or a Python installation**.

This is included **specifically for referee convenience**, as reviewers can easily inspect the notebooks without setting up a computational environment.  
The HTML files are direct exports of the corresponding notebooks and are intended for quick inspection and review by readers and referees. The authoritative, executable versions remain the `.ipynb` notebooks.