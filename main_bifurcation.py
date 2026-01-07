# main_bifurcation.py
"""
Phase-diagram generator for the circumferential bifurcation problem.

This script:
  - Builds the symbolic incremental problem (SymbolicProblem)
  - Builds the base-state equilibrium solver (EquilibriumProblem)
  - Uses BifurcationNumeric to find the critical inner stretch λ_a^crit
    for each circumferential mode n
  - Sweeps over one control parameter to generate a phase diagram:
        * B/A           (relative thickness)
        * ell/A         (regularization length over inner radius)
        * Gc/(mu * ell) (dimensionless toughness)
  - Saves results to JSON
  - Plots λ_a^crit vs the chosen parameter for several n.

Assumes the following local modules exist:
    symbolic_problem.py     (class SymbolicProblem)
    equilibrium.py          (class EquilibriumProblem)
    bifurcation_numeric.py  (class BifurcationNumeric)
"""

import os
import json
import logging
from datetime import date

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from classes.symbolic_problem import SymbolicProblem
from classes.equilibrium import EquilibriumProblem
from classes.bifurcation_numeric import BifurcationNumeric


# -------------------------------------------------------------------------
# 1. User configuration
# -------------------------------------------------------------------------

# Material / damage model
MATERIAL = "NeoHookean"        # currently only NeoHookean is wired in
DAMAGE_MODEL = "AT2"           # "AT2" or "AT1" (AT1 may need a different λ_a bracket)

# Base/reference parameters (used as anchors for the sweeps)
BASE_B = 1.0                   # reference outer radius
BASE_A = (1/5) * BASE_B          # reference inner radius (so BASE_B/BASE_A = 5)
MU = 1.0                       # shear modulus
BASE_ELL_OVER_A = 1.0          # default ℓ/A
BASE_C = 10.0                  # default C = Gc / (μ ℓ)
Jm = 1000.0                    # Gent parameter (unused for NeoHookean but OK)

# -------------------------------------------------------------------------
# Derived "fixed" dimensional parameters (for sweeps that hold ell and Gc fixed)
# -------------------------------------------------------------------------
ELL_FIXED = BASE_ELL_OVER_A * BASE_A        # a fixed length (based on your base A)
GC_FIXED  = BASE_C * MU * ELL_FIXED         # fixed toughness corresponding to BASE_C
B_FIXED   = BASE_B                          # fixed outer radius

B_OVER_A_FIXED = BASE_B / BASE_A          # keep thickness ratio fixed
ELL_FIXED      = BASE_ELL_OVER_A * BASE_A # pick a dimensional ell anchor
GC_FIXED       = BASE_C * MU * ELL_FIXED  # fixed because C and ell are fixed



# Bifurcation options
BOUNDARY_CONDITION = "Neumann"   # "Neumann" or "Dirichlet"
MODES = [2, 3, 4, 5, 6]          # circumferential modes to include in the plot

# Phase-diagram choice:
#   "B_over_A"       → vary B/A   (relative thickness)
#   "ell_over_A"     → vary ℓ/A
#   "Gc_over_mu_ell" → vary C = Gc / (μ ℓ)

PHASE_DIAGRAM_TYPE = "B_over_A"  # change this to switch diagrams

NUM_POINTS = 21  # resolution along the x-axis parameter

# Bracketing / root-finding for λ_a
LAMBDA_A_START = 2.0      # global starting guess for λ_a
LAMBDA_A_STEP = 0.1       # step for *global* bracketing search
ROOT_SOLVER = "ridder"    # "ridder", "brenth", or "bisection"
ROOT_TOL = 1e-4

# Output directory for JSON + figures
RAW_DATA_ROOT = "data/"


# -------------------------------------------------------------------------
# 2. Helper: JSON writer
# -------------------------------------------------------------------------

def write_to_json(savedir: str, filename: str, data: dict) -> None:
    """Write a Python dict to JSON, creating directory if needed."""
    os.makedirs(savedir, exist_ok=True)
    path = os.path.join(savedir, filename)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved JSON: {path}")


# -------------------------------------------------------------------------
# 3. Helper: set sweep over the chosen parameter
# -------------------------------------------------------------------------

def get_sweep_arrays(phase_diagram_type: str):
    """
    Returns:
        param_values : 1D array of parameter values to sweep (ascending)
        x_label      : str, for plot
        title_suffix : str, added to plot title
        x_name       : str, key name used in JSON (x-axis label)
    The actual (A, B, ell, Gc) for each param are computed later.
    """

    if phase_diagram_type == "B_over_A":
        # Sweep relative thickness B/A from slightly >1 to, say, 10
        param_values = np.linspace(1.01, 10.0, NUM_POINTS)
        param_values = np.linspace(1.2, 10.0, NUM_POINTS)
        x_label = r"$B/A$"
        title_suffix = rf"$ G_c / (\mu \ell) = {BASE_C:.2f}$, " \
                       rf"$\ell/A = {BASE_ELL_OVER_A:.2f}$"
        x_name = "B_over_A"

    elif phase_diagram_type == "ell_over_A":
        # Sweep ℓ/A from 0.05 to 1.0
        param_values = np.linspace(0.05, 1., NUM_POINTS)
        x_label = r"$\ell/A$"
        title_suffix = rf"$ G_c / (\mu \ell) = {BASE_C:.2f}$, " \
                       rf"$B/A = {BASE_B/BASE_A:.2f}$"
        x_name = "ell_over_A"

    elif phase_diagram_type == "Gc_over_mu_ell":
        # Sweep C = Gc / (μ ℓ) on a log scale
        param_values = np.logspace(0, 3, NUM_POINTS)
        x_label = r"$G_c / (\mu \ell)$"
        title_suffix = rf"$B/A = {BASE_B/BASE_A:.2f}$, " \
                       rf"$\ell/A = {BASE_ELL_OVER_A:.2f}$"
        x_name = "Gc_over_mu_ell"

    else:
        raise ValueError(f"Unknown PHASE_DIAGRAM_TYPE: {phase_diagram_type}")

    return param_values, x_label, title_suffix, x_name


def compute_parameters_from_sweep_value(param: float, phase_diagram_type: str):
    """
    Given a sweep value 'param' and the phase_diagram_type, return
    the concrete (A, B, ell, Gc) for this point.

    We use:
        BASE_A, BASE_B, BASE_ELL_OVER_A, BASE_C, MU
    as anchors.

    Returns:
        A, B, ell, Gc
    """
    if phase_diagram_type == "B_over_A":
        # A fixed, vary B to change B/A; keep ℓ/A and C constant
        A = BASE_A
        B = param * A                          # param = B/A
        ell = BASE_ELL_OVER_A * A
        Gc = BASE_C * MU * ell                 # keep C = Gc/(μℓ) constant

    elif phase_diagram_type == "ell_over_A":
        # A, B fixed, vary ℓ/A; keep C constant
        A = BASE_A
        B = BASE_B
        ell = param * A                        # param = ℓ/A
        Gc = BASE_C * MU * BASE_ELL_OVER_A * A                 

    elif phase_diagram_type == "Gc_over_mu_ell":
        # A, B fixed, ℓ fixed (via BASE_ELL_OVER_A), vary C = Gc/(μℓ)
        A = BASE_A
        B = BASE_B
        ell = BASE_ELL_OVER_A * A
        C = param                              # param = C
        Gc = C * MU * ell

    else:
        raise ValueError(f"Unknown PHASE_DIAGRAM_TYPE: {phase_diagram_type}")

    return A, B, ell, Gc


# -------------------------------------------------------------------------
# 4. Main phase-diagram driver
# -------------------------------------------------------------------------

def run_phase_diagram():
    """
    Main driver to build a phase diagram:
      x-axis  : chosen parameter (B/A, ℓ/A, or Gc/(μℓ))
      y-axis  : critical inner stretch λ_a^crit
      curves  : different circumferential modes n.

    Also saves:
      - JSON with x, λ_a^crit, λ_b^crit, and P^crit/μ for each mode n.
      - Two figures (EPS and PNG) with λ_a^crit vs x.
    """
    # Set up logging (in case of errors during sweeps)
    logging.basicConfig(
        filename="bifurcation_phase_diagram.log",
        filemode="w",
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # --- Sweep definition (ascending) ---
    param_values, x_label, title_suffix, x_name = get_sweep_arrays(PHASE_DIAGRAM_TYPE)

    # For stability in B/A diagrams, we *compute* from large B/A downwards
    # but will sort the results before plotting/saving.
    if PHASE_DIAGRAM_TYPE == "B_over_A":
        param_sweep = param_values[::-1]  # start from high B/A
    else:
        param_sweep = param_values[::-1]
    # param_sweep = param_values

    # --- Symbolic problem (can be reused for all parameter sets) ---
    symbolic = SymbolicProblem(material=MATERIAL, damage_model=DAMAGE_MODEL)

    # --- Results container ---
    results = {
        "meta": {
            "material": MATERIAL,
            "damage_model": DAMAGE_MODEL,
            "mu": MU,
            "base_A": BASE_A,
            "base_B": BASE_B,
            "base_ell_over_A": BASE_ELL_OVER_A,
            "base_C_Gc_over_mu_ell": BASE_C,
            "phase_diagram_type": PHASE_DIAGRAM_TYPE,
            "x_axis_key": x_name,
            "boundary_condition": BOUNDARY_CONDITION,
            "modes": MODES,
        },
        "data": {}
    }

    # --- Plot setup ---
    cent = 1.0 / 2.54    # centimeters in inches
    fig, ax = plt.subplots(figsize=(10 * cent, 7 * cent), dpi=600)
    plt.rcParams.update({"font.size": 10})

    # color index
    n_size = max(len(MODES) - 1, 1)

    # --- Outer loop over circumferential modes ---
    for i_mode, n in enumerate(MODES):
        print(f"\n=== Solving for mode n = {n} ===")
        x_list = []
        lambda_a_list = []
        lambda_b_list = []
        P_over_mu_list = []

        # Continuation: use previous λ_a^crit as the next initial guess
        lambda_guess = LAMBDA_A_START

        # Inner loop over parameter sweep
        for j, param in enumerate(param_sweep):
            # Translate the sweep parameter into (A, B, ell, Gc)
            A, B, ell, Gc = compute_parameters_from_sweep_value(
                param, PHASE_DIAGRAM_TYPE
            )

            print(f"  [{j+1}/{len(param_sweep)}] param = {param:.4g}, "
                  f"A={A:.4g}, B={B:.4g}, ell={ell:.4g}, Gc={Gc:.4g}, n={n:d}")

            try:
                # --- Build equilibrium problem ---
                eq = EquilibriumProblem(
                    A=A,
                    B=B,
                    mu=MU,
                    Gc=Gc,
                    ell=ell,
                    damage_model=DAMAGE_MODEL,
                    Jm=Jm,
                    dda_fun=symbolic.dda_f,
                )

                # --- Bifurcation numerics for this (A,B,ell,Gc,n) ---
                bif = BifurcationNumeric(
                    equilibrium=eq,
                    symbolic=symbolic,
                    n=n,
                    boundary_condition=BOUNDARY_CONDITION,
                )

                # -------------------------------------------------
                # 1) Local continuation-based bracketing and root
                # -------------------------------------------------
                local_dx = 0.2  # small local step around previous root

                try:
                    # Local bracket around the previous λ_a^crit
                    # kjh
                    if n == 2 and PHASE_DIAGRAM_TYPE == "ell_over_A":
                        local_dx=0.1
                        pass  
                    a_loc, b_loc = bif.bracket_root(
                        x0=lambda_guess -  1 * local_dx,
                        dx=local_dx,
                        max_iter=10,
                    )
                    # a_loc = 1.8
                    # b_loc = 5.
                    lambda_a_crit = bif.find_critical_stretch(
                        a_loc, b_loc, solver=ROOT_SOLVER, tol=ROOT_TOL
                    )

                except Exception as e_local:
                    # -------------------------------------------------
                    # 2) Fallback: global bracket from fixed starting λ_a
                    # -------------------------------------------------
                    print()
                    logging.warning(
                        "Local bracket failed at n=%d, param=%.4g: %s. "
                        "Trying global bracket.",
                        n, param, e_local,
                    )
                    
                    a_glob, b_glob = bif.bracket_root(
                        x0=LAMBDA_A_START,
                        dx=LAMBDA_A_STEP,
                        max_iter=50,
                    )
                    lambda_a_crit = bif.find_critical_stretch(
                        a_glob, b_glob, solver=ROOT_SOLVER, tol=ROOT_TOL
                    )

                # Update continuation guess for next parameter
                lambda_guess = lambda_a_crit

                # EquilibriumProblem has been updated to this λ_a^crit
                lambda_b_crit = eq.lambda_b
                P_crit_over_mu = eq.P / MU

                # Store in lists
                x_list.append(param)
                lambda_a_list.append(lambda_a_crit)
                lambda_b_list.append(lambda_b_crit)
                P_over_mu_list.append(P_crit_over_mu)

            except Exception as e:
                msg = (f"Error at mode n={n}, param={param:.4g}, "
                       f"(A,B,ell,Gc)=({A:.4g},{B:.4g},{ell:.4g},{Gc:.4g}): {e}")
                print("   !!!", msg)
                logging.error(msg, exc_info=True)
                # Skip this point if we can't find a root
                continue

        # Convert to arrays for convenience
        x_arr = np.array(x_list)
        La = np.array(lambda_a_list)
        Lb = np.array(lambda_b_list)
        Pmu = np.array(P_over_mu_list)

        # Sort by x so plots & JSON have monotone x-axis,
        # even though we may have computed from high→low.
        if len(x_arr) > 0:
            sort_idx = np.argsort(x_arr)
            x_arr = x_arr[sort_idx]
            La = La[sort_idx]
            Lb = Lb[sort_idx]
            Pmu = Pmu[sort_idx]

        # Save in results dict
        results["data"][f"n={n}"] = {
            x_name: x_arr.tolist(),
            "lambda_a_crit": La.tolist(),
            "lambda_b_crit": Lb.tolist(),
            "P_over_mu_crit": Pmu.tolist(),
        }

        # Plot λ_a^crit vs x
        if len(x_arr) > 0:
            color = cm.coolwarm(i_mode / max(n_size, 1))
            ax.plot(
                x_arr,
                La,
                "-",
                label=rf"$n = {n}$",
                color=color,
            )

    # --- Finalize plot ---
    ax.set_ylabel(r"Critical inner stretch: $\lambda_a^{\mathrm{crit}}$", fontsize=12)
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_title(title_suffix, fontsize=11)
    ax.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left")

    # --- Save results ---
    today = date.today()
    filename_base = (
        f"{BOUNDARY_CONDITION}_PD_{PHASE_DIAGRAM_TYPE}_{today}"
        f"_A={BASE_A}_B={BASE_B}_ell_over_A={BASE_ELL_OVER_A}_C={BASE_C}_mu={MU}"
    )

    savedir = os.path.join(RAW_DATA_ROOT, PHASE_DIAGRAM_TYPE)
    json_name = filename_base + ".json"
    write_to_json(savedir, json_name, results)

    eps_name = filename_base + "_CriticalInnerStretch.eps"
    png_name = filename_base + "_CriticalInnerStretch.png"

    fig.tight_layout()
    fig.savefig(os.path.join(savedir, eps_name), dpi=600, bbox_inches="tight")
    fig.savefig(os.path.join(savedir, png_name), dpi=600, bbox_inches="tight")
    print(f"Saved figures:\n  {os.path.join(savedir, eps_name)}\n  {os.path.join(savedir, png_name)}")


# -------------------------------------------------------------------------
# 5. Entry point
# -------------------------------------------------------------------------

if __name__ == "__main__":
    run_phase_diagram()
