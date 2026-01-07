# bifurcation_numeric.py
"""
Numerical bifurcation analysis for circumferential modes n.

This module separates the *numerics* of the bifurcation problem from:
  - the *symbolic* derivation of the incremental equations
      (symbolic_problem.SymbolicProblem)
  - the *base-state* equilibrium solver
      (equilibrium.EquilibriumProblem)

It provides:
  - BifurcationNumeric: a class that
      * builds the 6D first-order ODE system in the current frame
        for f = [U, U', U'', U''', B, B'](r),
      * assembles boundary-condition matrices at r = a and r = b,
      * evaluates the determinant D(λ_a) which vanishes at bifurcation,
      * finds the critical inner stretch λ_a via root-finding.

It also provides a bracketing helper (bracket_root) that mimics the old
boundary_finding(func) used in your parameter-sweep scripts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Optional

import numpy as np
from scipy.integrate import odeint
from scipy.optimize import brenth, ridder
from scipy.integrate import solve_ivp

from classes.equilibrium import EquilibriumProblem
from classes.symbolic_problem import SymbolicProblem


BoundaryType = Literal["Neumann", "Dirichlet"]


@dataclass
class BifurcationNumeric:
    """
    Numerical bifurcation solver for a given circumferential mode `n`.

    Parameters
    ----------
    equilibrium : EquilibriumProblem
        Base-state solver providing geometry (a, b), damage field α(r),
        and base pressure p0(r). Must be configured with dda_f.
    symbolic : SymbolicProblem
        Symbolic problem providing lambdified incremental equations
        (dudr4_f, dalphadr2_f) and boundary condition functions Eij_f.
    n : int
        Circumferential mode number.
    boundary_condition : {"Neumann", "Dirichlet"}
        Type of boundary condition at the outer radius:
          - "Neumann"   : traction-free at r = b (usual case).
          - "Dirichlet" : fixed displacement at r = b.
    """

    equilibrium: EquilibriumProblem
    symbolic: SymbolicProblem
    n: int
    boundary_condition: BoundaryType = "Neumann"

    # -------------------------
    # Initialization helpers
    # -------------------------
    def __post_init__(self) -> None:
        # Shortcuts to parameters
        self.A = self.equilibrium.A
        self.B = self.equilibrium.B
        self.mu = self.equilibrium.mu
        self.Gc = self.equilibrium.Gc
        self.ell = self.equilibrium.ell
        self.Jm = self.equilibrium.Jm
        self.damage_model = self.equilibrium.damage_model

        # Radial grid (will be set later)
        self._r_grid = None
        self._r_grid_ab = None


        # Lambdified incremental equations from SymbolicProblem
        self.dudr4_f = self.symbolic.dudr4_f
        self.dalphadr2_f = self.symbolic.dalphadr2_f

        # Boundary-condition coefficients E_ij(r)
        self.E11_f = self.symbolic.E11_f
        self.E12_f = self.symbolic.E12_f
        self.E13_f = self.symbolic.E13_f
        self.E14_f = self.symbolic.E14_f
        self.E15_f = self.symbolic.E15_f
        self.E16_f = self.symbolic.E16_f

        self.E21_f = self.symbolic.E21_f
        self.E22_f = self.symbolic.E22_f
        self.E23_f = self.symbolic.E23_f
        self.E24_f = self.symbolic.E24_f
        self.E25_f = self.symbolic.E25_f
        self.E26_f = self.symbolic.E26_f

        # Boundary matrices / basis ICs will be set later
        self.BCA: Optional[np.ndarray] = None
        self.BCB: Optional[np.ndarray] = None
        self.BC1: Optional[np.ndarray] = None
        self.BC2: Optional[np.ndarray] = None
        self.BC3: Optional[np.ndarray] = None

    # -------------------------
    # Convenience geometry props
    # -------------------------
    @property
    def a(self) -> float:
        return self.equilibrium.a

    @property
    def b(self) -> float:
        return self.equilibrium.b

    # -------------------------
    # Incremental ODE system
    # -------------------------
    def _ode_system(self, f: np.ndarray, r: float) -> np.ndarray:
        """
        First-order ODE system for the perturbation fields in the current frame.

        State vector
        ------------
        f = [U, U', U'', U''', B, B'](r),
        where U(r) is the radial displacement amplitude and
              B(r) is the damage perturbation amplitude.

        The underlying equations are:
          - 4th order in U:     U'''' = dudr4_f(...)
          - 2nd order in B:     B''   = dalphadr2_f(...)
        """
        # Incompressibility: R^2 = r^2 - a^2 + A^2
        A = self.A
        a = self.a
        R = np.sqrt(r**2 - a**2 + A**2)
        lbda = r / R  # circumferential stretch λ(r)

        dfdr = np.zeros_like(f)

        # Trivial parts
        dfdr[0] = f[1]  # U'
        dfdr[1] = f[2]  # U''
        dfdr[2] = f[3]  # U'''

        # Base-state fields in current frame
        alpha = self.equilibrium.alpha(r)
        alpha_r = self.equilibrium.d_alpha(r)
        p_base = self.equilibrium.pressure(r)

        # 4th-order displacement equation: U'''' = dudr4_f(...)
        dfdr[3] = self.dudr4_f(r, lbda, self.n, f[0], f[1], f[2], f[3], alpha, alpha_r, f[4], f[5],self.ell, self.mu, self.Gc, self.Jm, p_base)

        # Damage perturbation equation: B'' = dalphadr2_f(...)
        dfdr[4] = f[5]
        dfdr[5] = self.dalphadr2_f(r, lbda, self.n, f[0], f[1], f[2], alpha, alpha_r, f[4], f[5], self.ell, self.mu, self.Gc, self.Jm)


        return dfdr

    # -------------------------
    # Boundary-condition matrices
    # -------------------------
    def _bc_matrix_at(self, r: float) -> np.ndarray:
        """
        Build the 3×6 boundary-condition matrix at radius r.

        Each row corresponds to a scalar boundary functional applied to
        the state vector f = [U, U', U'', U''', B, B'].

        In terms of the E_ij coefficients:

            [E11 E12 E13 E14 E15 E16] · f = 0
            [E21 E22 E23 E24 E25 E26] · f = 0
            [ 0   0   0   0   0   1 ] · f = 0  (simple B'-type condition)
        """
        A = self.A
        a = self.a

        R = np.sqrt(r**2 - a**2 + A**2)
        lbda = r / R

        alpha = self.equilibrium.alpha(r)
        alpha_r = self.equilibrium.d_alpha(r)
        p_base = self.equilibrium.pressure(r)

        # First row
        E11 = self.E11_f(self.n, r, lbda, alpha, alpha_r, self.mu, self.Jm)
        E12 = self.E12_f(self.n, r, lbda, alpha, alpha_r, self.mu, self.Jm)
        E13 = self.E13_f(self.n, r, lbda, alpha, alpha_r, self.mu, self.Jm)
        E14 = self.E14_f(self.n, r, lbda, alpha, alpha_r, self.mu, self.Jm)
        E15 = self.E15_f(self.n, r, lbda, alpha, alpha_r, self.mu, self.Jm, p_base)
        E16 = self.E16_f(self.n, r, lbda, alpha, alpha_r, self.mu, self.Jm)

        # Second row
        E21 = self.E21_f(self.n, r, lbda)
        E22 = self.E22_f(self.n, r, lbda)
        E23 = self.E23_f(self.n, r, lbda)
        E24 = self.E24_f(self.n, r, lbda)
        E25 = self.E25_f(self.n, r, lbda)
        E26 = self.E26_f(self.n, r, lbda)

        # Third row: [0 0 0 0 0 1]
        E31 = 0.0
        E32 = 0.0
        E33 = 0.0
        E34 = 0.0
        E35 = 0.0
        E36 = 1.0

        BCM = np.array(
            [
                [E11, E12, E13, E14, E15, E16],
                [E21, E22, E23, E24, E25, E26],
                [E31, E32, E33, E34, E35, E36],
            ]
        )
        return BCM

    def _define_bc_basis(self) -> None:
        """
        Construct three linearly independent initial conditions at r = a
        that satisfy the inner boundary conditions.

        Idea:
          - Let BCA be the 3×6 BC matrix at r = a.
          - We construct three vectors BC1, BC2, BC3 in ℝ⁶ such that
                BCA · BCk = 0  for each k,
            and the resulting solutions form a basis of the solution space
            of the ODE.
        """
        if self.BCA is None:
            raise RuntimeError("Boundary matrix at r=a not set.")

        # Initial guesses for [U, U', U'', U''', B, B']
        BC1 = np.array([1., 0., 0., 0., 1., 0.])
        BC2 = np.array([0., 1., 0., 0., 1., 0.])
        BC3 = np.array([0., 0., 0., 0., 1., 0.])

        # We solve for unknown second and third derivatives at r = a
        # (entries index 2 and 3) so that the first two BC rows are satisfied.
        tempbc = np.array(
            [
                [self.BCA[0, 2], self.BCA[0, 3]],
                [self.BCA[1, 2], self.BCA[1, 3]],
            ]
        )

        templ1 = np.array(
            [
                -self.BCA[0, 0] - self.BCA[0, 4],
                -self.BCA[1, 0],
            ]
        )
        templ2 = np.array(
            [
                -self.BCA[0, 1] - self.BCA[0, 4],
                -self.BCA[1, 1],
            ]
        )
        templ3 = np.array(
            [
                -self.BCA[0, 4],
                0.0,
            ]
        )

        BC1[2], BC1[3] = np.linalg.solve(tempbc, templ1)
        BC2[2], BC2[3] = np.linalg.solve(tempbc, templ2)
        BC3[2], BC3[3] = np.linalg.solve(tempbc, templ3)

        self.BC1 = BC1
        self.BC2 = BC2
        self.BC3 = BC3
        # for BC in (BC1, BC2, BC3):
        #     norm = np.linalg.norm(BC)
        #     if norm > 0:
        #         BC /= norm

        # self.BC1 = BC1
        # self.BC2 = BC2
        # self.BC3 = BC3
    # -------------------------
    # Determinant function D(λ_a)
    # -------------------------
    
    def _ode_system_ivp(self, r: float, f: np.ndarray) -> np.ndarray:
        # Just swap argument order and reuse the existing implementation
        return self._ode_system(f, r)

    def determinant(self, lambda_a: float) -> float:
        """
        Determinant D(λ_a) measuring compatibility of outer boundary conditions.

        For a given inner stretch λ_a:
          1. Compute base state (geometry, damage, pressure) via equilibrium.
          2. Build BC matrices at r = a and r = b.
          3. Construct three basis solutions (ODEs) starting from r = a.
          4. Evaluate BCs at r = b and build a 3×3 matrix D whose determinant
             must vanish at bifurcation.

        Returns
        -------
        detD : float
            Determinant of the boundary-condition matrix at r = b in the
            basis of three solutions. detD = 0 at bifurcation.
        """
        # Step 1: base state for this λ_a
        self.equilibrium.compute_inner_pressure_from_lambda_a(lambda_a)

        # Step 2: BC matrices at a and b
        self.BCA = self._bc_matrix_at(self.a)
        self.BCB = self._bc_matrix_at(self.b)

        # Step 3: basis of initial conditions at r = a
        self._define_bc_basis()

        # # Step 4: integrate the ODE from r = a to r = b for each basis

        ab = (self.a, self.b)
        if self._r_grid is None or self._r_grid_ab != ab:
            self._r_grid = np.linspace(self.a, self.b, 201)
            self._r_grid_ab = ab

        r_grid = self._r_grid

        

        # ...
        # r_grid = np.linspace(self.a, self.b, 501)
        opts = dict(rtol=1e-9, atol=1e-9, method="DOP853")  # DOP853, try "Radau" if stiff

        sol1 = solve_ivp(self._ode_system_ivp, (self.a, self.b), self.BC1, t_eval=r_grid, **opts)
        sol2 = solve_ivp(self._ode_system_ivp, (self.a, self.b), self.BC2, t_eval=r_grid, **opts)
        sol3 = solve_ivp(self._ode_system_ivp, (self.a, self.b), self.BC3, t_eval=r_grid, **opts)

        if not (sol1.success and sol2.success and sol3.success):
            raise RuntimeError(
                f"IVP failed: {sol1.message=} {sol2.message=} {sol3.message=}"
            )

        # Convert to odeint-like arrays: (n_points, n_states)
        Y1 = sol1.y.T
        Y2 = sol2.y.T
        Y3 = sol3.y.T


        # r_grid = np.linspace(self.a, self.b, 501)
        # common_opts = dict(rtol=1e-9, atol=1e-9)
        # sol1 = odeint(self._ode_system, self.BC1, r_grid, **common_opts)
        # sol2 = odeint(self._ode_system, self.BC2, r_grid, **common_opts)
        # sol3 = odeint(self._ode_system, self.BC3, r_grid, **common_opts)



        # Evaluate outer BCs at r = b
        if self.boundary_condition == "Neumann":
            D = np.array(
                [
                    self.BCB @ Y1[-1, :],
                    self.BCB @ Y2[-1, :],
                    self.BCB @ Y3[-1, :],
                ]
            )
        elif self.boundary_condition == "Dirichlet":
            # Fixed displacement at r = b:
            # use components [U, U', B] at r = b as rows
            idx = [0, 1, 4]
            s1 = Y1[-1, idx]
            s2 = Y2[-1, idx]
            s3 = Y3[-1, idx]
            D = np.array([s1, s2, s3])
        else:
            raise ValueError(f"Unknown boundary_condition: {self.boundary_condition}")

        return float(np.linalg.det(D))


    # Alias so your old boundary_finding(bifurcation.func) still works
    def func(self, lambda_a: float) -> float:
        """Alias for determinant(λ_a), for backwards compatibility."""
        return self.determinant(lambda_a)

    # -------------------------
    # Bracketing helper (like boundary_finding)
    # -------------------------
    def bracket_root(
        self,
        x0: float = 2.5,
        dx: float = 0.05,
        max_iter: int = 100,
        func: Optional[Callable[[float], float]] = None,
    ) -> tuple[float, float]:
        """
        Simple bracketing routine to find [x_min, x_max] where func changes sign.

        This mimics your previous boundary_finding(func) logic:

            x = 2.5; dx = 0.05
            temp = sign(func(x))
            while sign(func(x)) == temp: x += dx

        Parameters
        ----------
        x0 : float
            Starting guess for λ_a.
        dx : float
            Increment in λ_a used to search for the sign change.
        max_iter : int
            Maximum number of increments.
        func : callable, optional
            Function of λ_a. If None, uses self.determinant.

        Returns
        -------
        (x_min, x_max) : tuple of floats
            Interval where func changes sign: func(x_min) and func(x_max)
            have opposite signs.
        """
        if func is None:
            func = self.determinant

        x = x0
        f0 = func(x)
        s0 = np.sign(f0) if f0 != 0.0 else 1.0

        for _ in range(max_iter):
            x_new = x + dx
            f_new = func(x_new)
            s_new = np.sign(f_new) if f_new != 0.0 else s0

            if s_new != s0:
                # Sign change between x and x_new
                return (x, x_new)

            x = x_new

        raise RuntimeError(
            f"Failed to bracket a root in [{x0}, {x0 + max_iter*dx}] "
            "— determinant did not change sign."
        )

    # -------------------------
    # Root-finding: critical stretch (with given bracket)
    # -------------------------
    def find_critical_stretch(
        self,
        lambda_a_min: float,
        lambda_a_max: float,
        solver: Literal["brenth", "bisection", "ridder"] = "brenth",
        tol: float = 1e-4,
    ) -> float:
        """
        Find the critical inner stretch λ_a at which det(D(λ_a)) = 0.

        Parameters
        ----------
        lambda_a_min, lambda_a_max : float
            Bracketing interval for λ_a where the determinant changes sign.
        solver : {"brenth", "bisection", "ridder"}
            Root-finding scheme. Default "brenth".
        tol : float
            Absolute tolerance for the root.

        Returns
        -------
        lambda_a_crit : float
            Critical inner stretch λ_a where the bifurcation occurs.
        """
        f: Callable[[float], float] = self.determinant
        a, b = lambda_a_min, lambda_a_max

        if solver == "brenth":
            lambda_a_crit = brenth(f, a, b, xtol=tol, maxiter=200)
        elif solver == "ridder":
            lambda_a_crit = ridder(f, a, b, xtol=tol, maxiter=200)
        elif solver == "bisection":
            lambda_a_crit = self._bisection_method(f, a, b, tol=tol, max_iter=200)
        else:
            raise ValueError(f"Unknown solver: {solver}")

        # Update geometry for the critical stretch
        self.equilibrium.set_geometry_from_lambda_a(lambda_a_crit)
        return float(lambda_a_crit)

    # -------------------------
    # Root-finding: critical stretch with auto-bracket
    # -------------------------
    def find_critical_stretch_auto(
        self,
        x0: float = 2.,
        dx: float = 0.05,
        solver: Literal["brenth", "ridder", "bisection"] = "ridder",
        tol: float = 1e-4,
        max_iter_bracket: int = 100,
    ) -> float:
        """
        Convenience method: bracket + solve for the critical stretch λ_a.

        This mirrors your old pattern:

            a, b = boundary_finding(bifurcation.func)
            bifurcation.get_bifucation_point(a, b, solver)

        Now done as:

            lambda_a_crit = bifurcation.find_critical_stretch_auto()

        Parameters
        ----------
        x0 : float
            Initial guess for λ_a.
        dx : float
            Increment used to find a sign change in determinant(λ_a).
        solver : {"brenth", "ridder", "bisection"}
            Root-finding method to use after bracketing.
        tol : float
            Tolerance for the root.
        max_iter_bracket : int
            Maximum iterations to bracket the root.

        Returns
        -------
        lambda_a_crit : float
            Critical inner stretch λ_a where det(D(λ_a)) = 0.
        """
        a, b = self.bracket_root(x0=x0, dx=dx, max_iter=max_iter_bracket)
        return self.find_critical_stretch(a, b, solver=solver, tol=tol)

    # -------------------------
    # Simple bisection helper (optional)
    # -------------------------
    def _bisection_method(
        self,
        f: Callable[[float], float],
        a: float,
        b: float,
        tol: float = 1e-5,
        max_iter: int = 100,
    ) -> float:
        """
        Basic bisection method for scalar root finding.
        """
        fa = f(a)
        fb = f(b)
        if np.sign(fa) == np.sign(fb):
            raise ValueError(
                "Bisection error: f(a) and f(b) have the same sign; "
                "interval does not bracket a root."
            )

        for _ in range(max_iter):
            m = 0.5 * (a + b)
            fm = f(m)
            if abs(fm) < tol or 0.5 * (b - a) < tol:
                return m
            if np.sign(fm) == np.sign(fa):
                a, fa = m, fm
            else:
                b, fb = m, fm

        return 0.5 * (a + b)
