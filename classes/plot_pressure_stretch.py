# plot_pressure_stretch.py
"""
Quick script to generate a pressure–stretch curve for the cavity problem.

It uses:
  - SymbolicProblem.dda_f for the base-state damage ODE.
  - EquilibriumProblem to compute:
        * geometry (a, b, λ_b)
        * damage α_0(r)
        * inner pressure P(λ_a)

Output:
  - A plot of P/μ vs λ_a.
  - (Optional) a second curve vs λ_b.

Run with:
    python plot_pressure_stretch.py
"""

import numpy as np
import matplotlib.pyplot as plt

from symbolic_problem import SymbolicProblem
from equilibrium import EquilibriumProblem

import os
import numpy as np


def load_fem_pressure_curve(folder: str, A: float, B: float):
    """
    Load FEM simulation data and build λ_a, λ_b, and P/μ arrays for comparison
    with the analytical pressure–stretch curves.

    Assumes the folder contains:
        - forces.txt         (columns: [?, ?, P/mu] or similar)
        - displacements.txt  (columns: [?, u_a, u_b] where u_b is outer disp)

    Parameters
    ----------
    folder : str
        Path to the FEM output folder (e.g. 'FEMstabilized/output/...').
    A : float
        Reference inner radius.
    B : float
        Reference outer radius.

    Returns
    -------
    lambda_a : ndarray
        Inner stretch λ_a for each load step.
    lambda_b : ndarray
        Outer stretch λ_b for each load step.
    P_over_mu : ndarray
        Pressure normalized by μ, from the FEM forces file.
    """

    # --- Load raw data ---
    force = np.loadtxt(os.path.join(folder, "forces.txt"))
    displacements = np.loadtxt(os.path.join(folder, "displacements.txt"))

    # Remove all-zero rows (unused / uninitialized steps)
    mask = ~np.all(force == 0, axis=1)
    force = force[mask]
    displacements = displacements[mask]

    # --- Compute stretches from displacements ---

    # In your original script:
    #   lambdab = (displacements[:,2] + 1)/1
    # so we interpret displacements[:,2] as (λ_b - 1)
    lambda_b = displacements[:, 2] + 1.0

    # Incompressibility mapping between λ_a and λ_b:
    #   λ_a^2 = 1 - (B/A)^2 (1 - λ_b^2)
    lambda_a = np.sqrt(1.0 - (B / A) ** 2 * (1.0 - lambda_b**2))

    # FEM “pressure” already normalized by μ (you were plotting force[:,2] as P/μ)
    P_over_mu = force[:, 2]

    return lambda_a, lambda_b, P_over_mu

def main():
    # --------------------------------------------------------
    # 1. Parameters (tune these to match your paper)
    # --------------------------------------------------------
    A = 0.2          # reference inner radius
    B = 1.0          # reference outer radius
    mu = 1.0         # shear modulus
    Gc = 1.0        # fracture energy
    # ell_factor = 1  # ℓ = ell_factor * A
    ell = 0.1
    damage_model = "AT2"   # "AT1" or "AT2"
    Jm = 50.0              # Gent parameter (unused for NeoHookean but harmless)

    # Range of inner stretches λ_a to explore
    lambda_a_min = 1.01
    lambda_a_max = 7.0
    n_lambda = 100
    lambda_a_list = np.linspace(lambda_a_min, lambda_a_max, n_lambda)

    # --------------------------------------------------------
    # 2. Symbolic & equilibrium problems
    # --------------------------------------------------------
    # Symbolic: used only for the base-state damage ODE α''(r)
    sym = SymbolicProblem(material="NeoHookean", damage_model=damage_model)

    # Equilibrium: base-state geometry, damage, pressure
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

    # Storage for results
    lambda_a_valid = []
    lambda_b_valid = []
    P_over_mu_valid = []

    # --------------------------------------------------------
    # 3. Sweep over λ_a and compute P(λ_a)/μ
    # --------------------------------------------------------
    for lambda_a in lambda_a_list:
        print(f"Computing base state for λ_a = {lambda_a:.3f} ...")
        try:
            P = eq.compute_inner_pressure_from_lambda_a(lambda_a)
        except Exception as e:
            print(f"  Skipping λ_a = {lambda_a:.3f} (failed: {e})")
            continue

        # Store values only if everything succeeded
        lambda_a_valid.append(eq.lambda_a)
        lambda_b_valid.append(eq.lambda_b)
        P_over_mu_valid.append(P / mu)

    lambda_a_valid = np.array(lambda_a_valid)
    lambda_b_valid = np.array(lambda_b_valid)
    P_over_mu_valid = np.array(P_over_mu_valid)

    # --------------------------------------------------------
    # 4. Plot P/μ vs λ_a (and λ_b if desired)
    # --------------------------------------------------------
    plt.figure(figsize=(5, 4), dpi=300)
    plt.plot(lambda_a_valid, P_over_mu_valid, "o-", label=r"$P/\mu$ vs. $\lambda_a$")
    plt.xlabel(r"Inner stretch $\lambda_a$")
    plt.ylabel(r"Inner pressure $P/\mu$")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig("pressure_vs_lambda_a.png", dpi=300)
    print("Saved figure: pressure_vs_lambda_a.png")

    # Optional second figure: P/μ vs λ_b
    plt.figure(figsize=(5, 4), dpi=300)
    plt.plot(lambda_b_valid, P_over_mu_valid, "o-", label=r"$P/\mu$ vs. $\lambda_b$")
    plt.xlabel(r"Outer stretch $\lambda_b$")
    plt.ylabel(r"Inner pressure $P/\mu$")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig("pressure_vs_lambda_b.png", dpi=300)
    print("Saved figure: pressure_vs_lambda_b.png")

    # Uncomment if you want to see plots interactively:
    # plt.show()


if __name__ == "__main__":
    main()


from symbolic_problem import SymbolicProblem
from equilibrium import EquilibriumProblem
# from load_fem import load_fem_pressure_curve  # or paste function above directly

import matplotlib.pyplot as plt
import numpy as np

# --- Parameters and analytical curve (from your previous script) ---
A = 0.2
B = 1.0
mu = 1.0
Gc = 1.0
ell = 0.1
Jm = 50.0
damage_model = "AT2" 

sym = SymbolicProblem(material="NeoHookean", damage_model=damage_model)
eq = EquilibriumProblem(A, B, mu, Gc, ell, damage_model, Jm=Jm, dda_fun=sym.dda_f)

lambda_a_list = np.linspace(1.05, 7.0, 25)
lambda_a_analytical = []
lambda_b_analytical = []
P_over_mu_analytical = []

for lam_a in lambda_a_list:
    P = eq.compute_inner_pressure_from_lambda_a(lam_a)
    lambda_a_analytical.append(eq.lambda_a)
    lambda_b_analytical.append(eq.lambda_b)
    P_over_mu_analytical.append(P / mu)

lambda_a_analytical = np.array(lambda_a_analytical)
P_over_mu_analytical = np.array(P_over_mu_analytical)

# --- FEM data overlay ---
folder = "../FEM/output/circular_AT2C1_10.00_C2_0.50_R_void_0.20_mu_1.00_ell_0.100_Gc_1.0_ld-stps_1001_max-ld_1.00"
lambda_a_fem, lambda_b_fem, P_over_mu_fem = load_fem_pressure_curve(folder, A, B)

# --- Plot ---
plt.figure(figsize=(5, 4), dpi=300)
plt.plot(lambda_a_fem, P_over_mu_fem, "k-", label="FEM")
plt.plot(lambda_a_analytical, P_over_mu_analytical, "r--", label="Analytical")

plt.xlabel(r"Inner stretch $\lambda_a$")
plt.ylabel(r"Pressure $P/\mu$")
plt.grid(True, linestyle="--", alpha=0.4)
plt.legend()
plt.tight_layout()
# plt.savefig("pressure_vs_lambda_a_with_FEM.png", dpi=300)
# plt.show()

