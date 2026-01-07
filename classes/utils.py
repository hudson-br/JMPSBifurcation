#!/usr/bin/env python3
"""
Combined script:

  - Read FEM results for a given simulation folder.
  - Plot FEM pressure–stretch curve P/μ vs λ_a.
  - Compute and plot the corresponding equilibrium (axisymmetric) pressure–stretch curve.
  - Run bifurcation analysis for the same (A, B, μ, ℓ, Gc) and modes n = 2,...,6.
  - Overlay the critical stretches (λ_a^crit, P^crit/μ) for each n on the same plot.

This is meant to help you check whether a bifurcation that nucleates at the
inner boundary is also "felt" at the outer boundary in the FEM results.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from classes.symbolic_problem import SymbolicProblem
from classes.equilibrium import EquilibriumProblem
from classes.bifurcation_numeric import BifurcationNumeric
# from plot_pressure_stretch import load_fem_pressure_curve


# -------------------------------------------------------------------------
# 1. User configuration
# -------------------------------------------------------------------------

# Path to FEM output folder (change to the one you want)
# FEM_FOLDER = (
#     "../../JMPS2025/MinimalPackV2/output/JMPS2025/circular_AT2C1_25.00_C2_1.00_R_void_0.20_mu_1.00_ell_0.200_Gc_5.0_ld-stps_1001_max-ld_2.00"
# )

# FEM_FOLDER = ("../../JMPS2025/MinimalPackV2/output/JMPS2025/circular_AT2C1_10.00_C2_0.50_R_void_0.20_mu_1.00_ell_0.100_Gc_1.0_ld-stps_1001_max-ld_1.00")
# FEM_FOLDER = ("../../JMPS2025/MinimalPackV2/output/JMPS2025/circular_AT2C1_10.00_C2_0.33_R_void_0.30_mu_1.00_ell_0.100_Gc_1.0_ld-stps_1001_max-ld_1.00")
FEM_FOLDER = ("../FEMstabilized_current_folder_for_paper/output/BifurcationPaper/circular_AT2C1_10.00_C2_0.33_R_void_0.30_mu_1.00_ell_0.100_Gc_1.0_ld-stps_501_max-ld_2.00")
FEM_FOLDER = ("../FEMstabilized_current_folder_for_paper/output/BifurcationPaper/circular_AT2C1_10.00_C2_0.50_R_void_0.20_mu_1.00_ell_0.100_Gc_1.0_ld-stps_501_max-ld_1.00")

# Material and damage model (consistent with SymbolicProblem / EquilibriumProblem)
MATERIAL = "NeoHookean"
DAMAGE_MODEL = "AT2"
Jm = 50.0   # Gent parameter (unused for NeoHookean, but harmless)

# Circumferential modes to analyze
MODES = [2, 3, 4, 5, 6]

# Bifurcation boundary condition at outer radius ("Neumann" from the paper)
BOUNDARY_CONDITION = "Neumann"

# Range of λ_a to compute the equilibrium curve
LAMBDA_A_MIN = 1.01
LAMBDA_A_MAX = 6.0
N_LAMBDA_A = 120  # resolution of equilibrium curve
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import os
import numpy as np


def load_fem_pressure_curve(
    folder: str,
    A: float,
    B: float,
    *,
    load_damage: bool = True,
    damage_filename: str = "damage.txt",
    boundary_damage_filename: str = "boundary_damage.txt",
    inner_boundary_damage_filename: str | None = None,   # if you have a separate file
    outer_boundary_damage_filename: str | None = None,   # if you have a separate file
    drop_zero_damage_rows: bool = True,
):
    """
    Load FEM simulation data and build λ_a, λ_b, and P/μ arrays for comparison
    with analytical pressure–stretch curves, and (optionally) load damage outputs.

    Assumes the folder contains:
        - forces.txt
        - displacements.txt
    Optionally (if load_damage=True):
        - damage.txt                 : bulk damage profiles in reference coordinate R
        - boundary_damage.txt        : (by default) boundary damage profiles vs theta
                                      for some boundary (often outer). If you have
                                      separate inner/outer boundary files, pass them
                                      via inner_boundary_damage_filename / outer_boundary_damage_filename.

    Parameters
    ----------
    folder : str
        Path to FEM output folder.
    A : float
        Reference inner radius.
    B : float
        Reference outer radius.
    load_damage : bool
        If True, attempts to load damage arrays.
    damage_filename : str
        Filename for bulk damage (2D: [step, radial_index]).
    boundary_damage_filename : str
        Default filename for boundary damage if only one boundary file exists.
    inner_boundary_damage_filename, outer_boundary_damage_filename : str | None
        If provided, loads these instead of (or in addition to) boundary_damage_filename.
    drop_zero_damage_rows : bool
        If True, removes rows that are entirely zero in damage arrays (common in your outputs).

    Returns
    -------
    lambda_a : ndarray
        Inner stretch λ_a for each load step.
    lambda_b : ndarray
        Outer stretch λ_b for each load step.
    P_over_mu : ndarray
        Pressure normalized by μ, from the FEM forces file.

    damage_bulk : ndarray | None
        Bulk damage profiles (shape: [n_steps, n_R]) or None if not available.
    damage_bd_inner : ndarray | None
        Inner boundary damage profiles vs theta (shape: [n_steps, n_theta]) or None.
    damage_bd_outer : ndarray | None
        Outer boundary damage profiles vs theta (shape: [n_steps, n_theta]) or None.

    Notes
    -----
    - This function uses the same nonzero-row mask derived from forces.txt to filter
      displacements and (when possible) damage arrays, to keep indexing aligned.
    - If your boundary/bulk damage files include extra zero rows that do NOT align
      with forces/displacements, we additionally remove all-zero rows from those
      damage arrays (controlled by drop_zero_damage_rows).
    """

    # -------------------------
    # 1) Load raw force/displ
    # -------------------------
    force = np.loadtxt(os.path.join(folder, "forces.txt"))
    displacements = np.loadtxt(os.path.join(folder, "displacements.txt"))

    # Remove all-zero rows (unused / uninitialized steps) based on forces
    mask_steps = ~np.all(force == 0, axis=1)
    force = force[mask_steps]
    displacements = displacements[mask_steps]

    # -------------------------
    # 2) Compute stretches
    # -------------------------
    lambda_b = displacements[:, 2] + 1.0
    lambda_a = np.sqrt(1.0 - (B / A) ** 2 * (1.0 - lambda_b**2))
    P_over_mu = force[:, 2]

    # -------------------------
    # 3) Optionally load damage
    # -------------------------
    damage_bulk = None
    damage_bd_inner = None
    damage_bd_outer = None

    if not load_damage:
        return lambda_a, lambda_b, P_over_mu, damage_bulk, damage_bd_inner, damage_bd_outer

    def _load_damage_2d(path: str) -> np.ndarray | None:
        if not os.path.exists(path):
            return None
        arr = np.loadtxt(path)
        # First try to align by the force-derived mask if lengths match
        if arr.ndim == 2 and arr.shape[0] == mask_steps.shape[0]:
            arr = arr[mask_steps]
        # Often you still have leading/trailing all-zero rows; remove them if requested
        if drop_zero_damage_rows and arr.ndim == 2:
            arr = arr[~np.all(arr == 0, axis=1)]
        return arr

    # Bulk damage profiles (R direction)
    damage_bulk = _load_damage_2d(os.path.join(folder, damage_filename))

    # Boundary damage(s) (theta direction)
    # Case A: separate inner/outer files explicitly provided
    if inner_boundary_damage_filename is not None:
        damage_bd_inner = _load_damage_2d(os.path.join(folder, inner_boundary_damage_filename))
    if outer_boundary_damage_filename is not None:
        damage_bd_outer = _load_damage_2d(os.path.join(folder, outer_boundary_damage_filename))

    # Case B: only one boundary file exists (your current workflow)
    if (damage_bd_inner is None) and (damage_bd_outer is None):
        bd = _load_damage_2d(os.path.join(folder, boundary_damage_filename))
        # We don't know whether it's inner or outer; return it as "outer" by default
        damage_bd_outer = bd

    return lambda_a, lambda_b, P_over_mu, damage_bulk, damage_bd_inner, damage_bd_outer


def plot_pressure_stretch_with_damage_inset(
    FEM_FOLDER: str,
    lambda_a_eq: np.ndarray,
    P_over_mu_eq: np.ndarray,
    lambda_a_fem: np.ndarray,
    P_over_mu_fem: np.ndarray,
    lambdab: np.ndarray,
    crit_data: dict,
    MODES,
    *,
    # --- styling / layout ---
    fig_size_cm=(7.0, 5.0),
    dpi=200,
    inset_rect=(0.35, 0.35, 0.30, 0.30),   # (left,bottom,width,height) in figure coords
    main_rect=(0.10, 0.10, 0.90, 0.90),
    font_size=10,
    # --- data selection ---
    lambda_a_max: float | None = 6.0,     # set None to disable cutting
    marker_stride: int = 10,              # every N indices
    marker_start: int = 10,
    # --- annotations ---
    A: float | None = None,
    B: float | None = None,
    Gc: float | None = None,
    mu: float | None = None,
    ell: float | None = None,
    text_pos_gc = None,    # e.g. text_pos_gc=(5.5, 0.85),
    text_pos_ab = None,
    # --- I/O ---
    damage_filename="boundary_damage.txt",
    savepath: str | None = None,
    show: bool = True,
):
    """
    Plot:
      - main axes: equilibrium pressure-stretch + FEM curve + colored markers
      - black markers: bifurcation critical points (crit_data)
      - inset axes: boundary damage profiles at the same marker indices

    Assumptions:
      - boundary_damage.txt rows align with lambda_a_fem / P_over_mu_fem / lambdab indices.
      - boundary_damage.txt rows are time/load steps; columns are theta samples.
      - Rows that are all zeros are removed (as in your current workflow).
    """

    # -----------------------
    # 0) Matplotlib styling
    # -----------------------
    matplotlib.rcParams["mathtext.fontset"] = "stix"
    matplotlib.rcParams["font.family"] = "STIXGeneral"
    plt.rcParams.update({"font.size": font_size})

    cent = 1 / 2.54
    fig = plt.figure(figsize=(fig_size_cm[0] * cent, fig_size_cm[1] * cent), dpi=dpi)
    ax1 = fig.add_axes(list(main_rect))

    # -----------------------
    # 1) Load + clean boundary damage
    # -----------------------
    damage_path = os.path.join(FEM_FOLDER, damage_filename)
    damage_bd = np.loadtxt(damage_path)

    # remove all-zero rows
    data_full = damage_bd[~np.all(damage_bd == 0, axis=1)]
    x_axis = np.linspace(0.0, 2.0 * np.pi, data_full.shape[1])

    # -----------------------
    # 2) Consistent "cut" to lambda_a <= lambda_a_max
    # -----------------------
    if lambda_a_max is not None:
        mask = np.asarray(lambda_a_fem) <= float(lambda_a_max)
        lambda_a_fem_c = np.asarray(lambda_a_fem)[mask]
        P_over_mu_fem_c = np.asarray(P_over_mu_fem)[mask]
        lambdab_c = np.asarray(lambdab)[mask]

        # IMPORTANT: boundary damage rows must align with FEM indices
        # If your pipeline guarantees that, this mask is correct.
        data_c = data_full[mask, :]
    else:
        lambda_a_fem_c = np.asarray(lambda_a_fem)
        P_over_mu_fem_c = np.asarray(P_over_mu_fem)
        lambdab_c = np.asarray(lambdab)
        data_c = data_full

    # Guard: marker range should be valid
    n_steps = len(lambda_a_fem_c)
    if n_steps == 0:
        raise ValueError("After cutting (lambda_a_max), no FEM points remain to plot.")

    # -----------------------
    # 3) Marker indices + colormap normalization
    # -----------------------
    Indexes = list(range(marker_start, n_steps, marker_stride))
    n_size = max(len(Indexes), 1)

    # -----------------------
    # 4) Main plot: equilibrium + FEM + colored markers
    # -----------------------
    ax1.plot(lambda_a_eq, P_over_mu_eq, label=r"Analytical", alpha=0.9)
    ax1.plot(lambda_a_fem_c, P_over_mu_fem_c, "-", alpha=0.5, label=r"FEM")

    for count, i in enumerate(Indexes):
        ax1.plot(
            lambda_a_fem_c[i],
            P_over_mu_fem_c[i],
            ".",
            marker="o",
            markersize=2,
            color=cm.coolwarm(count / n_size),
        )

    # Bifurcation critical points
    for n in MODES:
        if n not in crit_data:
            continue
        lambda_a_crit, P_over_mu_crit = crit_data[n]
        ax1.plot(
            lambda_a_crit,
            P_over_mu_crit,
            "o",
            color="k",
            markersize=3,
            label=rf"$\lambda_a^{{\mathrm{{crit}}}}$, n={n}",
        )

    # Optional annotations
    if (Gc is not None) and (mu is not None) and (ell is not None) and text_pos_gc is not None:
        ax1.text(
            text_pos_gc[0],
            text_pos_gc[1],
            r"$G_c /(\mu \ell) = $" + str(int(Gc / (mu * ell))),
        )
    if (A is not None) and (B is not None) and text_pos_ab is not None:
        ax1.text(
            text_pos_ab[0],
            text_pos_ab[1],
            r"$A/B = $" + str(A / B),
        )

    ax1.set_xlabel(r"Circumferential stretch: $\lambda_a$", fontsize=font_size)
    ax1.set_ylabel(r"Pressure: $P/\mu$", fontsize=font_size)

    # -----------------------
    # 5) Inset: damage profiles at marker indices
    # -----------------------
    ax_in = fig.add_axes(list(inset_rect))
    for count, i in enumerate(Indexes):
        ax_in.plot(
            x_axis,
            data_c[i, :],
            linewidth=0.5,
            color=cm.coolwarm(count / n_size),
            alpha=0.75,
        )

    ax_in.set_xlabel(r"polar coordinate: $\theta$", fontsize=font_size)
    ax_in.set_ylabel(r"damage: $\alpha$", fontsize=font_size)

    ax_in.set_xticks([0, np.pi, 2 * np.pi])
    ax_in.set_xticklabels(["0", r"$\pi$", r"$2 \pi$"])

    # -----------------------
    # 6) Save / show / return
    # -----------------------
    if savepath is not None:
        fig.savefig(savepath, bbox_inches="tight", dpi=600)

    if show:
        plt.show()

    return fig, ax1, ax_in

# def load_fem_pressure_curve(folder: str, A: float, B: float):
#     """
#     Load FEM simulation data and build λ_a, λ_b, and P/μ arrays for comparison
#     with the analytical pressure–stretch curves.

#     Assumes the folder contains:
#         - forces.txt         (columns: [?, ?, P/mu] or similar)
#         - displacements.txt  (columns: [?, u_a, u_b] where u_b is outer disp)

#     Parameters
#     ----------
#     folder : str
#         Path to the FEM output folder (e.g. 'FEMstabilized/output/...').
#     A : float
#         Reference inner radius.
#     B : float
#         Reference outer radius.

#     Returns
#     -------
#     lambda_a : ndarray
#         Inner stretch λ_a for each load step.
#     lambda_b : ndarray
#         Outer stretch λ_b for each load step.
#     P_over_mu : ndarray
#         Pressure normalized by μ, from the FEM forces file.
#     """

#     # --- Load raw data ---
#     force = np.loadtxt(os.path.join(folder, "forces.txt"))
#     displacements = np.loadtxt(os.path.join(folder, "displacements.txt"))

#     # Remove all-zero rows (unused / uninitialized steps)
#     mask = ~np.all(force == 0, axis=1)
#     force = force[mask]
#     displacements = displacements[mask]

#     # --- Compute stretches from displacements ---

#     # In your original script:
#     #   lambdab = (displacements[:,2] + 1)/1
#     # so we interpret displacements[:,2] as (λ_b - 1)
#     lambda_b = displacements[:, 2] + 1.0

#     # Incompressibility mapping between λ_a and λ_b:
#     #   λ_a^2 = 1 - (B/A)^2 (1 - λ_b^2)
#     lambda_a = np.sqrt(1.0 - (B / A) ** 2 * (1.0 - lambda_b**2))

#     # FEM “pressure” already normalized by μ (you were plotting force[:,2] as P/μ)
#     P_over_mu = force[:, 2]



#     return lambda_a, lambda_b, P_over_mu
# -------------------------------------------------------------------------
# 2. Helper: extract (A, μ, ℓ, Gc) from FEM folder name
# -------------------------------------------------------------------------

def parse_parameters_from_folder(folder: str):
    """
    Parse A (R_void), μ, ℓ, and Gc from the folder name, assuming the pattern:

        ... R_void_<A>_mu_<mu>_ell_<ell>_Gc_<Gc>_ld-stps_...

    B is taken as 1.0 by convention.

    Returns
    -------
    A, B, mu, ell, Gc
    """
    base = os.path.basename(folder)

    def extract(prefix: str, suffix: str) -> float:
        i0 = base.index(prefix) + len(prefix)
        i1 = base.index(suffix, i0)
        return float(base[i0:i1])

    A = extract("R_void_", "_mu")
    mu = extract("mu_", "_ell")
    ell = extract("ell_", "_Gc")
    Gc = extract("Gc_", "_ld")

    B = 1.0  # default outer reference radius (as in your scripts)
    return A, B, mu, ell, Gc


# -------------------------------------------------------------------------
# 3. Equilibrium pressure–stretch curve (axisymmetric base state)
# -------------------------------------------------------------------------

def compute_equilibrium_curve(A, B, mu, ell, Gc, damage_model, Jm,
                              lambda_a_min, lambda_a_max, n_lambda):
    """
    Compute the axisymmetric equilibrium curve P(λ_a)/μ:

      - Builds SymbolicProblem and EquilibriumProblem.
      - Sweeps λ_a in [lambda_a_min, lambda_a_max].
      - Uses eq.compute_inner_pressure_from_lambda_a(λ_a).

    Returns
    -------
    lambda_a_array : ndarray
    P_over_mu_array : ndarray
    eq : EquilibriumProblem       (reused later for bifurcation)
    sym : SymbolicProblem
    """
    # Symbolic problem
    sym = SymbolicProblem(material=MATERIAL, damage_model=damage_model)

    # Equilibrium base-state solver
    eq = EquilibriumProblem(
        A=A,
        B=B,
        mu=mu,
        Gc=Gc,
        ell=ell,
        damage_model=damage_model,
        Jm=Jm,
        dda_fun=sym.dda_f,
    )

    lambda_a_list = np.linspace(lambda_a_min, lambda_a_max, n_lambda)

    lambda_a_valid = []
    P_over_mu_valid = []

    for lam_a in lambda_a_list:
        # print(f"[Equilibrium] λ_a = {lam_a:.3f}")
        try:
            P = eq.compute_inner_pressure_from_lambda_a(lam_a)
        except Exception as e:
            print(f"    >> skipping λ_a = {lam_a:.3f} (failed: {e})")
            continue

        lambda_a_valid.append(eq.lambda_a)  # eq might slightly adjust λ_a
        P_over_mu_valid.append(P / mu)

    return np.array(lambda_a_valid), np.array(P_over_mu_valid), eq, sym


# -------------------------------------------------------------------------
# 4. Bifurcation analysis for given parameters
# -------------------------------------------------------------------------

def compute_bifurcation_points(eq, sym, modes, boundary_condition,
                               x0=2.0, dx=0.05, solver="ridder", tol=1e-4):
    """
    For each mode n in 'modes', find the critical stretch λ_a^crit and
    corresponding pressure P^crit/μ.

    Uses:
        BifurcationNumeric.find_critical_stretch_auto(...)

    Returns
    -------
    crit_data : dict
        crit_data[n] = (lambda_a_crit, P_over_mu_crit)
        for modes where the root was successfully found.
    """
    crit_data = {}
    mu = eq.mu

    for n in modes:
        print(f"\n=== Bifurcation analysis for mode n = {n} ===")

        bif = BifurcationNumeric(
            equilibrium=eq,
            symbolic=sym,
            n=n,
            boundary_condition=boundary_condition,
        )

        try:
            lambda_a_crit = bif.find_critical_stretch_auto(
                x0=x0, dx=dx, solver=solver, tol=tol
            )
            # Once we know λ_a^crit, recompute the base state and pressure
            P_crit = eq.compute_inner_pressure_from_lambda_a(lambda_a_crit)
            P_over_mu_crit = P_crit / mu

            print(
                f"  -> λ_a^crit (n={n}) = {lambda_a_crit:.4f}, "
                f"P^crit/μ = {P_over_mu_crit:.4f}"
            )

            crit_data[n] = (lambda_a_crit, P_over_mu_crit)

        except Exception as e:
            print(f"  !! Failed to find critical stretch for n={n}: {e}")
            continue

    return crit_data


# -------------------------------------------------------------------------
# 5. Main driver
# -------------------------------------------------------------------------

def main():
    # --- Parse parameters from FEM folder ---
    A, B, mu, ell, Gc = parse_parameters_from_folder(FEM_FOLDER)
    print("Parameters from FEM folder:")
    print(f"  A   = {A}")
    print(f"  B   = {B}")
    print(f"  mu  = {mu}")
    print(f"  ell = {ell}")
    print(f"  Gc  = {Gc}")
    print()

    # --- Load FEM pressure–stretch curve ---
    # lambda_a_fem, lambda_b_fem, P_over_mu_fem = load_fem_pressure_curve(
    #     FEM_FOLDER, A, B
    # )

    lambda_a_fem, lambda_b_fem, P_over_mu_fem, damage_R, damage_inner, damage_outer = load_fem_pressure_curve(
        FEM_FOLDER, A, B,
        load_damage=True,
        boundary_damage_filename="boundary_damage.txt",   # if only one exists
    )


    # --- Compute equilibrium (axisymmetric) curve ---
    lambda_a_eq, P_over_mu_eq, eq, sym = compute_equilibrium_curve(
        A=A,
        B=B,
        mu=mu,
        ell=ell,
        Gc=Gc,
        damage_model=DAMAGE_MODEL,
        Jm=Jm,
        lambda_a_min=LAMBDA_A_MIN,
        lambda_a_max=LAMBDA_A_MAX,
        n_lambda=N_LAMBDA_A,
    )

    # --- Run bifurcation analysis for modes n = 2,...,6 ---
    crit_data = compute_bifurcation_points(
        eq=eq,
        sym=sym,
        modes=MODES,
        boundary_condition=BOUNDARY_CONDITION,
        x0=2.0,
        dx=0.05,
        solver="ridder",
        tol=1e-4,
    )

    # ------------------------------------------------------------------
    # 6. Plot: FEM curve, equilibrium curve, and critical points
    # ------------------------------------------------------------------
    plt.figure(figsize=(6, 4), dpi=300)

    # FEM curve
    plt.plot(
        lambda_a_fem,
        P_over_mu_fem,
        "k-",
        label="FEM",
        linewidth=1.5,
    )

    # Equilibrium (axisymmetric) curve
    plt.plot(
        lambda_a_eq,
        P_over_mu_eq,
        "r--",
        label="Equilibrium (axisymmetric)",
        linewidth=1.3,
    )

    # Critical points for each mode n
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(MODES)))
    for (i, n) in enumerate(MODES):
        if n not in crit_data:
            continue
        lambda_a_crit, P_over_mu_crit = crit_data[n]
        plt.plot(
            lambda_a_crit,
            P_over_mu_crit,
            "o",
            color=colors[i],
            label=rf"$\lambda_a^{{\mathrm{{crit}}}}$, n={n}",
        )

    plt.xlabel(r"Inner stretch $\lambda_a$")
    plt.ylabel(r"Pressure $P/\mu$")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()

    out_name = "pressure_stretch_fem_equilibrium_bifurcation.png"
    plt.savefig(out_name, dpi=300)
    print(f"\nSaved figure: {out_name}")


if __name__ == "__main__":
    main()
