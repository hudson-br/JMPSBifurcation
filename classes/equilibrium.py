# equilibrium.py
"""
Axisymmetric base-state (equilibrium) solver.

Responsibilities:
- Given (A, B, μ, Gc, ℓ, damage_model) and a damage ODE RHS (dda_f),
  solve for:
    * geometry (a, b) from a prescribed inner stretch λ_a
    * symmetric damage profile α_0(r)
    * base-state pressure P(λ_a)

This corresponds to the radial cavity expansion base state in the paper.
"""

from typing import Callable, Optional

import numpy as np
from scipy.integrate import solve_bvp, odeint, quad
from scipy.interpolate import CubicSpline
from scipy.optimize import fsolve

# Optional: pretty-printing when running in a notebook.
# In a plain Python script, we silently fall back to no-op functions.
try:
    from IPython.display import display, Math
except Exception:
    def display(*args, **kwargs):
        """Fallback display: do nothing in non-IPython environments."""
        return None

    def Math(x):
        """Fallback Math: just return the input."""
        return x


class EquilibriumProblem:
    """
    Base-state (axisymmetric) equilibrium for the cavity problem.

    Parameters
    ----------
    A : float
        Reference inner radius.
    B : float
        Reference outer radius.
    mu : float
        Shear modulus μ.
    Gc : float
        Fracture energy G_c.
    ell : float
        Length-scale parameter ℓ.
    damage_model : str
        'AT1' or 'AT2'. Controls how the damage domain is treated.
    Jm : float, optional
        Gent extensibility parameter. Only used if the material model needs it.
    dda_fun : callable, optional
        Lambdified RHS of the base-state damage ODE α''(r), with signature
            dda_fun(r, lambda_theta, alpha, alpha_r, ell, mu, Gc, Jm).
        Typically this is `symbolic_problem.dda_f`. If None, damage
        computation will raise an error when invoked.
    """

    def __init__(
        self,
        A: float,
        B: float,
        mu: float,
        Gc: float,
        ell: float,
        damage_model: str,
        Jm: float = 1.0,
        dda_fun: Optional[Callable] = None,
    ):
        # Geometric & material parameters
        self.A = A
        self.B = B
        self.mu = mu
        self.Gc = Gc
        self.ell = ell
        self.damage_model = damage_model
        self.Jm = Jm

        # Symbolic damage RHS (from SymbolicProblem.dda_f)
        self.dda_f = dda_fun

        # Will be updated by set_geometry_from_lambda_a
        self.lambda_a: Optional[float] = None  # inner stretch λ_a = a/A
        self.lambda_b: Optional[float] = None  # outer stretch λ_b = b/B
        self.a: Optional[float] = None         # current inner radius
        self.b: Optional[float] = None         # current outer radius

        # Damaged region end-point for AT1 (current radius)
        self.damage_domain: Optional[float] = None

        # These will be filled by get_damage() and get_pressure()
        self._alpha_sol = None           # solve_bvp solution for α(r)
        self.P: Optional[float] = None   # inner pressure corresponding to λ_a

        # Pressure interpolation: p0(r)
        self._p_r_grid = None
        self._p_interpolated: Optional[CubicSpline] = None

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------
    def set_geometry_from_lambda_a(self, lambda_a: float) -> None:
        """
        Set current inner and outer radii from the control parameter λ_a.

        Uses incompressibility in plane strain:
            r^2(R) = a^2 - A^2 + R^2,
        so for the outer boundary R = B we have
            b^2 = a^2 - A^2 + B^2.

        Also stores λ_a = a/A and λ_b = b/B.
        """
        self.lambda_a = float(lambda_a)
        self.a = self.lambda_a * self.A
        # incompressibility: r(R)^2 = a^2 - A^2 + R^2
        self.b = float(np.sqrt(self.a**2 - self.A**2 + self.B**2))
        self.lambda_b = self.b / self.B

    def r_of_lambda(self, lam: float) -> float:
        """
        Current radius r(λ) from incompressibility.

        For a given circumferential stretch λ at some radius r, the
        incompressibility relation gives:

            r(λ) = λ * sqrt( (a^2 - A^2) / (λ^2 - 1) ).

        This is used when integrating in λ instead of r.
        """
        if self.a is None or self.A is None:
            raise RuntimeError(
                "Geometry not set. Call set_geometry_from_lambda_a() first."
            )
        lam = float(lam)
        num = self.a**2 - self.A**2
        return lam * np.sqrt(num / (lam**2 - 1.0))

    # ------------------------------------------------------------------
    # Damage BVP: α(r)
    # ------------------------------------------------------------------
    def bvp_alpha0(
        self,
        r: np.ndarray,
        y: np.ndarray,
        mu: float,
        Gc: float,
        ell: float,
        Jm: float,
    ) -> np.ndarray:
        """
        Current-frame damage ODE for solve_bvp.

        Parameters
        ----------
        r : array_like
            Current radius (independent variable).
        y : array_like, shape (2, N)
            State vector with
                y[0] = α(r),
                y[1] = dα/dr.
        mu, Gc, ell, Jm : floats
            Material/damage parameters passed through to dda_f.

        Returns
        -------
        dydr : ndarray, shape (2, N)
            Right-hand side [α_r, α_rr] of the first-order system.
        """
        if self.dda_f is None:
            raise RuntimeError(
                "dda_f is not set. Pass SymbolicProblem.dda_f as dda_fun "
                "when constructing EquilibriumProblem."
            )

        # Map to reference radius R only if needed to get λθ = r/R
        R = np.sqrt(r**2 - self.a**2 + self.A**2)  # from incompressibility
        lambda_theta = r / R

        alpha = y[0]
        alpha_r = y[1]

        # current-frame RHS α_rr = f(r, λθ, α, α_r)
        alpha_rr = self.dda_f(
            r,
            lambda_theta,
            alpha,
            alpha_r,
            ell,
            mu,
            Gc,
            Jm,
        )

        return np.vstack((alpha_r, alpha_rr))

    def bc_bvp_alpha0(self, ya: np.ndarray, yb: np.ndarray) -> np.ndarray:
        """
        Neumann boundary conditions for the damage BVP.

        We impose zero-flux (natural) BCs:
            α'(a) = 0,  α'(b) = 0.
        """
        return np.array([ya[1], yb[1]])

    def objective_shooting_method(self, r_star: np.ndarray) -> float:
        """
        Objective for shooting method in the AT1 case.

        For AT1 we assume damage is confined to [a, r*], with α(r*) = 0.
        This method:
          - solves the BVP on [a, r*],
          - returns α(r*) as the shooting residual.

        Parameters
        ----------
        r_star : array_like (length 1)
            Guess for the end of the damaged region in the current frame.

        Returns
        -------
        residual : float
            Value of α(r*) for the current guess r*.
        """
        r = np.linspace(self.a, r_star[0], 200)
        y = np.zeros((2, r.size))  # initial guess

        sol = solve_bvp(
            lambda r, y: self.bvp_alpha0(
                r, y,
                mu=self.mu, Gc=self.Gc,
                ell=self.ell, Jm=self.Jm,
            ),
            self.bc_bvp_alpha0,
            r, y,
        )

        # Shooting condition: α(r*) = 0
        return sol.sol(r_star)[0]

    def get_damage(self) -> None:
        """
        Solve for the base-state damage profile α(r).

        - For AT1:
            * Determine the damaged region [a, r_damage_domain] using a
              shooting method enforcing α(r*) = 0.
            * Solve the BVP only on [a, r_damage_domain].
            * α and α' are taken as zero outside this region.
        - For AT2:
            * Solve the BVP on the full interval [a, b].

        The solution is stored in `self._alpha_sol`, which is the SciPy
        `solve_bvp` solution object.
        """
        if self.a is None or self.b is None:
            raise RuntimeError(
                "Geometry not set. Call set_geometry_from_lambda_a() first."
            )

        if self.damage_model == "AT1":
            try:
                # 1) Find the damage front r* close to b
                self.damage_domain = fsolve(
                    self.objective_shooting_method,
                    [0.99 * self.b],
                )
                r = np.linspace(self.a, self.damage_domain[0], 200)
                y = np.zeros((2, r.size))

                # 2) Solve BVP on [a, r*]
                self._alpha_sol = solve_bvp(
                    lambda r, y: self.bvp_alpha0(
                        r, y,
                        mu=self.mu, Gc=self.Gc,
                        ell=self.ell, Jm=self.Jm,
                    ),
                    self.bc_bvp_alpha0,
                    r, y,
                )
            except Exception:
                # Degenerate case: no damage, α ≡ 0
                self.damage_domain = self.a
                self._alpha_sol = None
        else:  # AT2
            r = np.linspace(self.a, self.b, 400)
            y = np.zeros((2, r.size))

            self._alpha_sol = solve_bvp(
                lambda r, y: self.bvp_alpha0(
                    r, y,
                    mu=self.mu, Gc=self.Gc,
                    ell=self.ell, Jm=self.Jm,
                ),
                self.bc_bvp_alpha0,
                r, y,
            )

    def alpha(self, r: float) -> float:
        """
        Evaluate the base-state damage α(r).

        For AT1:
          - α(r) is taken from the BVP solution if r < damage_domain,
          - α(r) = 0 otherwise.

        For AT2:
          - α(r) is taken directly from the BVP solution on [a, b].
        """
        if self.damage_model == "AT1":
            if self._alpha_sol is None or self.damage_domain is None:
                # No damage: α ≡ 0
                return 0.0
            return float(self._alpha_sol.sol(r)[0]) if r < self.damage_domain else 0.0
        else:
            if self._alpha_sol is None:
                raise RuntimeError("Damage not solved yet; call get_damage() first.")
            return float(self._alpha_sol.sol(r)[0])

    def d_alpha(self, r: float) -> float:
        """Evaluate the radial derivative α'(r)."""
        if self.damage_model == "AT1":
            if self._alpha_sol is None or self.damage_domain is None:
                return 0.0
            return float(self._alpha_sol.sol(r)[1]) if r < self.damage_domain else 0.0
        else:
            if self._alpha_sol is None:
                raise RuntimeError("Damage not solved yet; call get_damage() first.")
            return float(self._alpha_sol.sol(r)[1])


    # ------------------------------------------------------------------
    # Base-state radial stress σ_rr(r) and pressure p0(r)
    # ------------------------------------------------------------------

    def sigma_rr(self, r: float) -> float:
        """
        Compute the radial Cauchy stress σ_rr^0(r) in the base state via
        the integral representation

            σ_rr^0(r) = -∫_{s=a}^{r} (1 - α(s))^2 μ (-1/λ(s) - 1/λ(s)^3) (dλ/ds)(s) ds,

        with
            λ(s)   = s / sqrt(s^2 - a^2 + A^2),
            dλ/ds  = λ(s) (1 - λ(s)^2) / s,

        and inner boundary condition σ_rr^0(a) = 0 (stress-free cavity wall).
        """
        if self.a is None or self.b is None:
            raise RuntimeError(
                "Geometry not set. Call set_geometry_from_lambda_a() first."
            )

        def integrand_s(s):
            # Incompressibility: R^2 = s^2 - a^2 + A^2
            chi = s**2 - self.a**2 + self.A**2       # = R^2
            lam = s / np.sqrt(chi)                   # λ(s)
            dlam_ds = lam * (1.0 - lam**2) / s       # dλ/ds

            alpha_s = self.alpha(s)

            return ((1.0 - alpha_s)**2) * self.mu * (
                -1.0/lam - 1.0/lam**3
            ) * dlam_ds

        # Integrate from s = a to s = r, using σ_rr^0(a) = 0
        sigma, _ = quad(integrand_s, self.a, r)
        return sigma


    def get_pressure(self) -> None:
        """
        Compute the base-state pressure p0(r) from σ_rr^0(r) using the
        constitutive relation

            σ_rr^0(r) = (1 - α(r))^2 [ μ / λ(r)^2 - p0(r) ],

        i.e.

            p0(r) = μ / λ(r)^2 - σ_rr^0(r) / (1 - α(r))^2.

        This builds an interpolant p0(r) over [a, b].
        """
        if self.a is None or self.b is None:
            raise RuntimeError(
                "Geometry not set. Call set_geometry_from_lambda_a() first."
            )

        # Ensure damage has been computed
        if self._alpha_sol is None:
            raise RuntimeError(
                "Damage not solved yet; call get_damage() first."
            )

        # Radial grid (avoid exact endpoints for numerical robustness)
        r_grid = np.linspace(self.a + 1e-6, self.b - 1e-6, 400)
        p_vals = np.zeros_like(r_grid)

        for i, r in enumerate(r_grid):
            # Kinematics from incompressibility: R^2 = r^2 - a^2 + A^2
            chi = r**2 - self.a**2 + self.A**2
            lam_sq = r**2 / chi  # λ(r)^2

            alpha_r = self.alpha(r)
            a_fac = (1.0 - alpha_r)**2

            # Radial Cauchy stress σ_rr^0(r) from the integral formula
            sigma_rr_r = self.sigma_rr(r)

            if a_fac < 1e-10:
                # Fully damaged / nearly zero stiffness: avoid division by zero.
                # In this limit σ_rr^0 ≈ 0, so take p0 ≈ μ/λ^2.
                p_vals[i] = self.mu / lam_sq
            else:
                p_vals[i] = self.mu / lam_sq - sigma_rr_r / a_fac

        self._p_r_grid = r_grid
        self._p_interpolated = CubicSpline(r_grid, p_vals)

    # def pressure(self, r: float) -> float:
    #     """Base-state pressure p0(r) in the current configuration."""
    #     if self._p_interpolated is None:
    #         raise RuntimeError(
    #             "Pressure not solved yet; call get_pressure() first."
    #         )
    #     return float(self._p_interpolated(self.a+self.b - r))
    


    def pressure(self, r: float) -> float:
        """
        Base-state pressure p0(r) in the current configuration.

        Uses the constitutive relation
            σ_rr^0(r) = (1 - α(r))^2 [ μ / λ(r)^2 - p0(r) ],
        i.e.
            p0(r) = μ / λ(r)^2 - σ_rr^0(r) / (1 - α(r))^2.
        """
        if self.a is None or self.b is None:
            raise RuntimeError(
                "Geometry not set. Call set_geometry_from_lambda_a() first."
            )

        # circumferential stretch λ(r)^2 from incompressibility:
        # r^2 = a^2 - A^2 + R^2  ⇒  λ^2 = r^2 / (r^2 - a^2 + A^2)
        r = float(r)
        chi = r**2 - self.a**2 + self.A**2
        lam_sq = r**2 / chi

        alpha_r = self.alpha(r)
        a_fac = (1.0 - alpha_r)**2

        sigma_rr_r = self.sigma_rr(r)

        if a_fac < 1e-10:
            # Almost fully damaged: stiffness ~ 0 ⇒ σ_rr ≈ 0,
            # so take p0 ≈ μ / λ^2 to avoid division by ~0.
            return self.mu / lam_sq

        return self.mu / lam_sq - sigma_rr_r / a_fac


    def compute_inner_pressure_from_lambda_a(self, lambda_a: float) -> float:
        """
        Set geometry and damage for a given λ_a, build p0(r), and define
        the inner pressure as

            P = σ_rr^0(b),

        with σ_rr^0 computed from the integral representation and
        σ_rr^0(a) = 0. Also performs a small diagnostic check of the
        constitutive relation at r = b.
        """
        # 1. Geometry and damage
        self.set_geometry_from_lambda_a(lambda_a)
        self.get_damage()

        # 2. Build p0(r) (and implicitly σ_rr(r))
        self.get_pressure()

        # 3. Inner pressure from cavity traction balance
        sigma_a = self.sigma_rr(self.a)
        sigma_b = self.sigma_rr(self.b)
        self.P = sigma_b - sigma_a  # = -σ_rr(a) since σ_rr(b) = 0

        # 4. Tiny diagnostic at r = a: check σ_rr^0(a) ≈ (1-α(a))^2 (μ/λ_a^2 - p0(a))
        alpha_a = self.alpha(self.a)
        chi_a = self.a**2 - self.a**2 + self.A**2  # = A^2
        lam_a_sq = self.a**2 / chi_a               # should equal (a/A)^2 = λ_a^2
        a_fac = (1.0 - alpha_a)**2
        p0_a = self.pressure(self.a)

        lhs = sigma_a
        rhs = a_fac * (self.mu / lam_a_sq - p0_a)
        rel_err = 0.0
        if abs(rhs) > 1e-12:
            rel_err = abs(lhs - rhs) / abs(rhs)

        # print(
        #     f"[diagnostic] r=a: sigma_rr(a) = {lhs:.6e}, "
        #     f"(1-α)^2(μ/λ^2 - p0) = {rhs:.6e}, "
        #     f"relative error = {rel_err:.3e}"
        # )

        return self.P
