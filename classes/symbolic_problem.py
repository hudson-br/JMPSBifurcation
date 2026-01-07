# symbolic_problem.py
"""
Symbolic derivation for the cavity bifurcation problem.

This module is meant to be purely symbolic:

- It defines the base-state (axisymmetric) kinematics, material model and
  damage functions.
- It derives:
    * the base-state pressure equation dp/dr,
    * the damage equilibrium equation α''(r),
- It derives the incremental equilibrium equations and boundary conditions for
  circumferential perturbations with mode number n.
- It exposes lambdified functions that are used by the numerical code:

    Base-state / damage:
        dpdr_f(r, lambda(r), alpha(r), alpha_r(r), ell, mu, Gc, Jm)
        dda_f(r, lambda(r), alpha(r), alpha_r(r), ell, mu, Gc, Jm)
        p0_for_bc_f(mu, Jm, lambda)

    Bifurcation boundary-condition coefficients:
        E11_f(...), ..., E26_f(...)

    Incremental 4th-order displacement and damage equation:
        dudr4_f(r, lambda, n, U0, U1, U2, U3,
                alpha, alpha_r, B, dB, ell, mu, Gc, Jm, p0)
        dalphadr2_f(r, lambda, n, U0, U1, alpha, B, dB, ell, mu, Gc, Jm)

This file is the “symbolic layer” of the code; it should not contain any
numerical integration or root-finding (that belongs in your numeric modules).
"""

import sympy as smp
# from IPython.display import display, Math

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



class SymbolicProblem:
    """
    SymbolicProblem(material, damage_model)

    Parameters
    ----------
    material : str
        Name of the hyperelastic model. Currently "NeoHookean" and "Gent"
        are supported in the original implementation.
    damage_model : str
        "AT1" or "AT2", controlling the form of the local damage dissipation.
    """

    # ------------------------------------------------------------------
    # INITIALIZATION AND BASIC SYMBOLS
    # ------------------------------------------------------------------
    def __init__(self, material: str, damage_model: str):
        # ------------------------------------------------------------------
        # Coordinates: current (r, θ), reference (R)
        # ------------------------------------------------------------------
        self.r = smp.symbols("r", real=True, positive=True)
        self.theta = smp.symbols("theta", real=True, positive=True)
        self.R = smp.symbols("R", real=True, positive=True)

        # Geometry parameters (inner radius in current & reference configs)
        self.a = smp.symbols("a", real=True, positive=True)
        self.A = smp.symbols("A", real=True, positive=True)

        # Material and damage parameters
        self.material = material
        self.damage_model = damage_model

        self.mu = smp.symbols("mu", real=True, positive=True)
        self.Jm = smp.symbols("J_m", real=True, positive=True)
        self.ell = smp.symbols("ell", real=True, positive=True)

        self.Gc = smp.symbols("G_c", real=True, positive=True)
        self.w1 = smp.symbols("w_1", real=True, positive=True)

        # Mode number and some frequently used combination
        self.n = smp.symbols("n", real=True, positive=True)
        self.Cte = smp.symbols("C", real=True, positive=True)  # e.g., Gc/(mu * ell)

        # Base-state hydrostatic pressure and its increment
        self.p0, self.p1 = smp.symbols("p_0 p_1", cls=smp.Function)
        self.p0 = self.p0(self.r)                  # p0 = p0(r)
        self.p1 = self.p1(self.r, self.theta)      # p1 = p1(r, θ)

        # Base-state stretch λ(r)
        self.lbda = smp.symbols("lambda", cls=smp.Function)
        self.lbda = self.lbda(self.r)

        # Base-state damage α(r)
        self.damage = smp.symbols("alpha", cls=smp.Function)
        self.damage = self.damage(self.r)

        # Incremental displacement field (functions of r, θ)
        self.vr, self.vtheta = smp.symbols("v_r v_theta", cls=smp.Function)
        self.vr = self.vr(self.r, self.theta)
        self.vtheta = self.vtheta(self.r, self.theta)

        # Damage perturbation β(r, θ)
        self.damage_perturbation = smp.symbols("beta", cls=smp.Function)
        self.damage_perturbation = self.damage_perturbation(self.r, self.theta)

        # Radial “ansatz” functions for mode n:
        #   vr = U(r) sin(n θ), vθ = V(r) cos(n θ), p1 = Q(r) sin(n θ), β = B(r) cos(n θ)
        self.U, self.V, self.Q, self.B = smp.symbols("U V Q B", cls=smp.Function)
        self.U = self.U(self.r)
        self.V = self.V(self.r)
        self.Q = self.Q(self.r)
        self.B = self.B(self.r)

        # Gradient of the incremental displacement in current coordinates
        self.Gamma = smp.Matrix(
            [
                [
                    smp.diff(self.vr, self.r),
                    1 / self.r * (smp.diff(self.vr, self.theta) - self.vtheta),
                ],
                [
                    smp.diff(self.vtheta, self.r),
                    1 / self.r * (smp.diff(self.vtheta, self.theta) + self.vr),
                ],
            ]
        )
        self.show_equations = False
        # Build symbolic structure
        self.set_kinematics()
        self.set_damage_functions()
        self.set_material()
        self.set_equilibrium_equations()
        self.incremental_elastic_equilibrium()
        self.incremental_damage_equilibrium()
        self.boundary_conditions()
        print("Symbolic problem initialized.")

    # ------------------------------------------------------------------
    # BASE-STATE KINEMATICS & MATERIAL
    # ------------------------------------------------------------------
    def set_kinematics(self):
        """
        Define base-state kinematics for the cavity problem.

        The base state is purely circumferential stretch λ(r) with
            F = diag(λ_r, λ_θ) = diag(1/λ, λ),
        so that incompressibility det F = 1 holds.
        """
        self.F = smp.Matrix(
            [
                [1 / self.lbda, 0],
                [0, self.lbda],
            ]
        )
        self.C = self.F.T * self.F

        self.I1 = smp.trace(self.C)
        self.I2 = smp.Rational(1, 2) * (
            smp.trace(self.C) ** 2 - smp.trace(self.C * self.C)
        )

    def set_damage_functions(self):
        """
        Define degradation function a(α) and local dissipation w(α),
        and compute the associated constant c_w needed in the
        1D damage equilibrium.
        """
        # Degradation function a(α)
        self.a = (1 - self.damage) ** 2

        # Local energy w(α) depends on damage model
        if self.damage_model == "AT1":
            # AT1: w(α) ~ α
            self.w = self.damage
            z = smp.Symbol("z", positive=True)
            self.c_w = float(4 * smp.integrate(smp.sqrt(z), (z, 0, 1)))
        elif self.damage_model == "AT2":
            # AT2: w(α) ~ α²
            self.w = self.damage ** 2
            z = smp.Symbol("z", positive=True)
            self.c_w = float(
                4 * smp.integrate(smp.sqrt(self.w), (self.damage, 0, 1))
            )
        else:
            raise NotImplementedError(
                f"Unknown damage model {self.damage_model}; expected 'AT1' or 'AT2'."
            )

    def set_material(self):
        """
        Define the strain-energy density ψ for the chosen material model.

        Currently specialized to:
            - NeoHookean
            - Gent (limited extensibility parameter Jm)
        """
        if self.material == "NeoHookean":
            # ψ = μ/2 (I1 - 2)
            self.strain_energy_density = self.mu / 2 * (self.I1 - 2)
            self.purely_elastic_cauchy =  self.mu * self.F * self.F.T - self.p0 * smp.eye(2)  # Stress tensor: T_0
            
            self.incremental_purely_elastic_cauchy = self.mu * ( self.Gamma * self.F * self.F.T + self.F * self.F.T * self.Gamma.T ) - self.p1 * smp.eye(2) # T_1

            self.dWdF = self.mu * self.F # Used in the incremental damage equation

            # display(self.strain_energy_density)
        elif self.material == "Gent":
            # ψ = - (μ Jm)/2 ln(1 - (I1 - 2)/Jm)
            self.strain_energy_density = (
                -self.mu * self.Jm / 2 * smp.log(1 - (self.I1 - 2) / self.Jm)
            )
            self.purely_elastic_cauchy =  self.mu * (self.Jm / (self.Jm - self.I1 + 2)) * self.F * self.F.T - self.p0 * smp.eye(2)  # Stress tensor: T_0
            
            self.incremental_purely_elastic_cauchy = (
                                                        2 * self.mu * (self.Jm / (self.Jm - self.I1 + 2)**2 ) * smp.trace( self.F * self.F.T * self.Gamma.T) * self.F * self.F.T 
                                                        + self.mu * (self.Jm / (self.Jm - self.I1 + 2) ) * ( self.Gamma * self.F * self.F.T + self.F * self.F.T * self.Gamma.T ) 
                                                        - self.p1 * smp.eye(2) # T_1
                                                    )
            self.dWdF = self.mu * (self.Jm / (self.Jm - self.I1 + 2) ) * self.F # Used in the incremental damage equation
            
            # display(self.strain_energy_density)
        else:
            raise NotImplementedError(
                f"Unknown material {self.material}; "
                "extend set_material() to handle it."
            )


        
        self.sigma0 = self.a * self.purely_elastic_cauchy
        self.sigma1 = self.a * self.incremental_purely_elastic_cauchy + smp.diff(self.a, self.damage) * self.damage_perturbation * self.purely_elastic_cauchy
        if self.show_equations:
            # Display main information about the model            
            print("Setting the hyperelastic material " + self.material)
            display(Math(r'W = ' + smp.latex(self.strain_energy_density) ))
            display(Math(r'T = ' + smp.latex(self.purely_elastic_cauchy) ))
            # display(Math(r'T_1 = ' + smp.latex(self.incremental_purely_elastic_cauchy) ))


            # # Uncomment to show the forms of the incremental stress tensors and the functional form of the boundary conditions
            
            display(Math(r'\sigma^{(1)} = ' + smp.latex(self.sigma1) ))
            display(Math(r'\sigma^{(0)} n_0 = ' + smp.latex(self.sigma0 * smp.Matrix([[1],[0]])) ))
            display(Math(
                        r'\sigma^{(1)} n_0 - \sigma^{(0)} \Gamma^T n_0 = ' + smp.latex( 
                                                                                        ( self.sigma1  - self.sigma0 * self.Gamma.T ) * smp.Matrix([[1],[0]]) 
                                                                                        )   
                        )  
                )   
            # Simplified boundary conditions

            self.P = smp.symbols("P", real=True, positive=True)

            display(Math(
                        r'\sigma^{(1)} n_0 - (\sigma^{(0)}-PI) \Gamma^T n_0 = ' + smp.latex( 
                                                                                        smp.simplify((( self.sigma1  - (self.sigma0 - self.P * smp.eye(2)) * self.Gamma.T ) * smp.Matrix([[1],[0]])).subs(self.p0, self.mu/self.lbda**2)/self.a) 
                                                                                        )   
                        )  
                )   
            
            display(Math(
                        r'\sigma^{(1)} n_0 - \sigma^{(0)} \Gamma^T n_0 = ' + smp.latex( 
                                                                                        smp.simplify((( self.sigma1  - (self.sigma0 - self.P * smp.eye(2)) * self.Gamma.T ) * smp.Matrix([[1],[0]])).subs(self.p0, self.mu/self.lbda**2- self.P/self.a)) 
                                                                                        )   
                        )  
                )   

    # ------------------------------------------------------------------
    # BASE-STATE EQUILIBRIUM (p0(r), α(r))
    # ------------------------------------------------------------------
    def set_equilibrium_equations(self):
        """
        Derive:
        - base-state equilibrium div σ⁰ = 0, which gives the ODE for dp0/dr
        - damage equilibrium (1D) giving α''(r) in terms of α, α', λ, etc.

        This method also builds lambdified functions:
            p0_for_bc_f(mu, Jm, lambda)
            dpdr_f(p0, r, lambda, alpha, alpha_r, ell, mu, Gc, Jm)
            dda_f(r, lambda, alpha, alpha_r, ell, mu, Gc, Jm)
        """

        # Divergence of σ⁰ in cylindrical coordinates (current frame)
        self.ds0 = self.divergence(self.sigma0)
        if self.show_equations:
            print("Setting equilibrium equations:")
            print("Divergence of sigma_0:")
            display(self.ds0[0])

        # Pressure at the traction-free boundary (used for BC at r = b)
        self.p0_for_bc = smp.solve(self.sigma0[0, 0], self.p0)[0]
        self.p0_for_bc_f = smp.lambdify(
            (self.mu, self.Jm, self.lbda),
            self.p0_for_bc,
        )
        if self.show_equations:
            
            display(Math(r"p(R=A) = " + smp.latex(self.p0_for_bc)))

        # div σ⁰ = 0 gives dp0/dr
        self.dpdr = smp.solve(self.ds0[0], smp.diff(self.p0, self.r))[0]

        # Replace λ'(r) using incompressibility relation for planar cavity
        #   λ(r) is given by the usual cavity mapping:
        #   r² - a² = λ⁻² (R² - A²)  ⇒ λ'(r) = λ (1 - λ²)/r
        self.dpdr = self.dpdr.subs(
            [(smp.diff(self.lbda, self.r), self.lbda * (1 - self.lbda**2) / self.r)]
        ).doit()
        # display(Math(r"\frac{dp}{dr} = " + smp.latex(self.dpdr)))

        # Lambdify dp/dr; numeric code will wrap this in a 1D ODE solver.
        self.dpdr_f = smp.lambdify((self.p0, self.r, self.lbda, self.damage, smp.diff(self.damage, self.r), self.ell, self.mu, self.Gc, self.Jm), self.dpdr,)
        # print("dp/dr function defined.")

        

        # Damage equilibrium equation α''(r) from the *pushed-forward* reference functional.
        # We use Eq. (75) in the SI:
        #   a'(α0) W(F0) + (Gc/(c_w ℓ)) [ w'(α0) - 2 ℓ^2 ∇_e^2 α0 ] = 0,
        # with the geometric Laplacian
        #   ∇_e^2 α0 = div_x( F0 F0^T ∇_x α0 ).
        #
        # In the present axisymmetric setting (α0 = α0(r)), we construct this symbolically
        # and solve for α0''(r) to obtain an explicit ODE α'' = dda(...).
        FFt0 = self.F * self.F.T
        # display(FFt0)
        # grad_x α0 = (∂α0/∂r, 0)^T in cylindrical coordinates
        grad_alpha0 = smp.Matrix(
            [
                smp.diff(self.damage, self.r),
                0,
            ]
        )

        q0 = FFt0 * grad_alpha0
        # display(q0)

        # Geometric Laplacian ∇_e^2 α0 = div_x(q0) in cylindrical coordinates:
        #   div_x(q0) = (1/r) ∂_r ( r q0_r ) + (1/r) ∂_θ q0_θ.
        # The base state is axisymmetric, so the θ-derivative vanishes.
        geo_lap_alpha0 = (
            1 / self.r * smp.diff(self.r * q0[0], self.r)
            + 1 / self.r * smp.diff(q0[1], self.theta)
        )

        # Base-state damage equilibrium a'(α0) W(F0) + (Gc/(c_w ℓ)) [ w'(α0) - 2 ℓ^2 geo_lap_alpha0 ] = 0
        damage_eq0 = (
            smp.diff(self.a, self.damage) * self.strain_energy_density
            + self.Gc / (self.c_w * self.ell)
            * (smp.diff(self.w, self.damage) - 2 * self.ell**2 * geo_lap_alpha0)
        )

        # Solve for α''(r)
        self.dda = smp.solve(damage_eq0, smp.diff(self.damage, self.r, 2))[0]

        # Eliminate λ'(r) using the incompressibility relation λ'(r) = λ (1 − λ²) / r
        self.dda = self.dda.subs(
            smp.diff(self.lbda, self.r), self.lbda * (1 - self.lbda**2) / self.r
        ).doit()
        if self.show_equations:       
            print("Damage equilibrium equation:")   
            display(Math(r"\frac{d^2 \alpha}{d r^2} = " + smp.latex(self.dda)))
    
        self.dda_f = smp.lambdify(
            (self.r, self.lbda, self.damage, smp.diff(self.damage, self.r),
             self.ell, self.mu, self.Gc, self.Jm),
            self.dda,
        )
        # print("Damage equilibrium function defined. Testing lambdfy function: self.dda_f(1, 2, 0.5, 0.1, 0.01, 1, 0.1, 100) = ",self.dda_f(1, 2, 0.5, 0.1, 0.01, 1, 0.1, 100))

    # ------------------------------------------------------------------
    # GENERIC DIVERGENCE IN CYLINDRICAL COORDINATES
    # ------------------------------------------------------------------
    def divergence(self, T):
        """
        Divergence of a 2×2 tensor T in cylindrical coordinates
        (current configuration, assuming no z-dependence).
        Returns a 2×1 vector (div T)_i.
        """
        return smp.Matrix(
            [
                smp.diff(T[0, 0], self.r)
                + smp.diff(T[0, 1], self.theta) / self.r
                + (T[0, 0] - T[1, 1]) / self.r,
                smp.diff(T[1, 0], self.r)
                + smp.diff(T[1, 1], self.theta) / self.r
                + (T[1, 0] + T[0, 1]) / self.r,
            ]
        )

    # ------------------------------------------------------------------
    # INCREMENTAL ELASTIC EQUILIBRIUM (dudr4_f)
    # ------------------------------------------------------------------
    def incremental_elastic_equilibrium(self):
        """
        Derive the incremental elastic equilibrium equations for the
        circumferential mode n, linearized around the base state.

        Starting point:
          - We have the incremental Cauchy stress tensor σ¹[r, θ].
          - The incremental equilibrium equation is div σ¹ = 0.

        Steps:
          1. Compute div σ¹ in cylindrical coordinates.
          2. Insert the circumferential mode ansatz:
               v_r(r,θ)      =  U(r) sin(n θ)
               v_θ(r,θ)      =  V(r) cos(n θ)
               p¹(r,θ)       =  Q(r) sin(n θ)
               β(r,θ)        =  B(r) sin(n θ)
          3. Eliminate V using incompressibility:
               V(r) = (r U'(r) + U(r)) / n
          4. Replace λ'(r) using the base-state incompressibility relation:
               λ'(r) = λ(r) (1 - λ(r)²) / r
          5. From the θ-equilibrium equation, solve for Q(r).
          6. Differentiate Q(r) to get Q'(r) and substitute both Q and Q'
             back into the radial equilibrium equation.
          7. Rearrange the radial equation into a 4th-order ODE for U(r),
             and isolate U⁽⁴⁾(r).
          8. Build and store the lambdified function:

               dudr4_f(
                   r, lambda, n,
                   U0, U1, U2, U3,
                   alpha, alpha_r,
                   B, dB,
                   ell, mu, Gc, Jm, p0
               )

           which will be used by the numerical solver.
        """

        # ------------------------------------------------------------
        # 1. Compute divergence of the incremental stress: div σ¹
        # ------------------------------------------------------------
        self.ds = self.divergence(self.sigma1)        
        
        # ------------------------------------------------------------
        # 2. Insert circumferential mode ansatz into div σ¹ = 0
        #
        #   v_r      = U(r) sin(n θ)
        #   v_θ      = V(r) cos(n θ)
        #   p¹       = Q(r) sin(n θ)
        #   β        = B(r) sin(n θ)
        #
        # After substitution, ds becomes a vector whose θ-dependence
        # is carried through sin(nθ), cos(nθ).
        # ------------------------------------------------------------

        self.ds = self.ds.subs([
                                (self.vr,       self.U * smp.sin(self.theta * self.n)), 
                                (self.vtheta,   self.V * smp.cos(self.theta*self.n)), 
                                (self.p1,       self.Q * smp.sin(self.theta*self.n)), 
                                (self.damage_perturbation, self.B * smp.sin(self.theta * self.n))
                                ]).doit() # need to call doit

        # ------------------------------------------------------------
        # 3. Enforce incompressibility on the incremental displacement:
        #
        #   V(r) = (r U'(r) + U(r)) / n
        #
        # This eliminates V from the equations.
        # ------------------------------------------------------------
        self.ds = self.ds.subs([(self.V, (self.r * smp.diff(self.U, self.r) + self.U)/self.n )]).doit() # need to call doit
        
        
        # ------------------------------------------------------------
        # 4. Replace λ'(r) using the base-state incompressibility relation:
        #
        #   λ'(r) = λ(r) (1 − λ(r)²) / r
        #
        # This is consistent with the cavity mapping; it removes λ' from
        # the incremental equations in favor of λ itself.
        # ------------------------------------------------------------
        self.ds = self.ds.subs([(smp.diff(self.lbda,self.r), self.lbda * (1 - self.lbda**2)/self.r ) ]).doit() # need to call doit
        # display(self.ds)
        

        # Split the two scalar equilibrium equations:
        #   Eq_r     : radial component of div σ¹ = 0
        #   Eq_theta : circumferential component of div σ¹ = 0
        self.Eq_r = self.ds[0]
        self.Eq_theta = self.ds[1]
        
        # ------------------------------------------------------------
        # 5. From θ-equilibrium, solve for Q(r)
        #
        # Eq_theta = 0  ⇒  Q = Q_sols(r, U, U', U'', B, B', α, α', …)
        # ------------------------------------------------------------
        self.Q_sols = smp.solve(self.Eq_theta, self.Q, simplify = False)[0]
        
        # Compute Q'(r) and again eliminate λ'(r) using the same relation as in step 4
        self.Q_sols_r = smp.diff(self.Q_sols, self.r)
        self.Q_sols_r = self.Q_sols_r.subs([(smp.diff(self.lbda,self.r), self.lbda * (1 - self.lbda**2)/self.r ) ]).doit() # need to call doit

        # self.Final_equation = smp.collect(self.Eq_r.subs(
        #                             [(smp.diff(self.Q,self.r),self.Q_sols_r), (self.Q, self.Q_sols) ]
        #                             ), smp.diff(self.U))

        # display(self.Q_sols_r)
        # print("Final Q_sols_r")
         
        # print(" Behold the final equation!! ")
        # Fn = smp.collect(smp.expand((n**2 * r**2 * lbda**2*Final_equation/ (-smp.sin(n*theta)*r**4)).simplify()),smp.diff(U))
        # # display(smp.simplify(r**4*Fn))
        
        # print("Final equation")
        # self.Fn2 = (self.n**2 * self.r**2 * self.lbda**2 * self.Final_equation/ (-smp.sin( self.n * self.theta) * self.r**4))

        # ------------------------------------------------------------
        # 6. Substitute Q and Q' into the radial equilibrium Eq_r = 0
        #
        # This gives the final 4th-order ODE (before isolating U⁽⁴⁾).
        # ------------------------------------------------------------
        if self.material == "NeoHookean":
            # For neo-Hookean, we keep the full θ-dependence and normalize later
            self.Final_equation = smp.expand(
                                    self.Eq_r.subs(
                                                        [(smp.diff(self.Q,self.r),self.Q_sols_r), (self.Q, self.Q_sols) ]
                                                        ))
            self.Final_equation = smp.expand(
                                            self.Eq_r.subs(
                                                            [
                                                                (smp.diff(self.Q, self.r), self.Q_sols_r),
                                                                (self.Q, self.Q_sols),
                                                            ]
                                                        )
                                            )
       
            # Normalize the equation by factoring out sin(nθ) and σ scaling.
            # The factor is chosen to match the form used in the numerical code.
            # self.Fn2 = smp.collect(smp.expand((self.n**2 * self.r**2 * self.lbda**2 * self.Final_equation/ (-smp.sin( self.n * self.theta) * self.r**4)).simplify()),smp.diff(self.U))
            
            self.Fn2 = smp.collect(
                smp.expand(
                    (
                        self.n**2 * self.r**2 * self.lbda**2 * self.Final_equation
                        / (-smp.sin(self.n * self.theta) * self.r**4)
                    ).simplify()
                ),
                smp.diff(self.U),
            )

            # --------------------------------------------------------
            # 7. Isolate the highest derivative term d⁴U/dr⁴
            #
            # Fn2(U, U', U'', U''', U'''', B, B', α, α', …) = 0.
            # Solve algebraically for d⁴U/dr⁴.
            # --------------------------------------------------------

            self.solsu = smp.solve(self.Fn2, smp.diff(self.U, self.r,4))
            self.dudr4 = self.solsu[0]

            # self.solsu = smp.solve(self.Fn2, smp.diff(self.U,self.r,4),
            #                 simplify=False, rational=False)

            # print("dudr4")
            # display(self.dudr4)


            # --------------------------------------------------------
            # 8. Introduce dummy symbols for lower-order quantities to
            #    make lambdify cleaner and avoid nested derivatives.
            #    These will be replaced by numeric values in the solver.
            # --------------------------------------------------------

            self.U3, self.U2, self.U1, self.U0 = smp.symbols('U_3 U_2 U_1 U_0', real = True, positive = True)  
            self.dB, self.da = smp.symbols('B_d alpha_d', real = True, positive = True)  
            

            # Replace U derivatives by dummy variables U0, U1, U2, U3
            self.dudr4 = self.solsu[0].subs(
                [
                    (smp.diff(self.U, self.r, 3), self.U3), 
                    (smp.diff(self.U, self.r, 2), self.U2), 
                    (smp.diff(self.U, self.r, 1), self.U1), 
                    (self.U, self.U0)
                ]
            )

            # Replace damage and B derivatives:
            #   α''(r)   → self.dda   (base-state 2nd derivative)
            #   α'(r)    → da
            #   B'(r)    → dB
            self.dudr4 = self.dudr4.subs(
                [
                    (smp.diff(self.damage, self.r,2), self.dda), 
                    (smp.diff(self.damage, self.r), self.da), 
                    (smp.diff( self.B, self.r ), self.dB)
                ]
            )

            # Lambdified function for d⁴U/dr⁴ in the neo-Hookean case

            self.dudr4_f = smp.lambdify(
                (
                    self.r, 
                    self.lbda, 
                    self.n, 
                    self.U0, 
                    self.U1, 
                    self.U2, 
                    self.U3, 
                    self.damage, 
                    self.da, 
                    self.B, 
                    self.dB, 
                    self.ell, 
                    self.mu, 
                    self.Gc, 
                    self.Jm, 
                    self.p0  
                ), 
                self.dudr4
                )
            
        elif self.material == "Gent":
            # --------------------------------------------------------
            # Gent material: the algebra is slightly different.
            # We simplify Eq_r with the ansatz and then isolate U⁽⁴⁾.
            # --------------------------------------------------------

            # Substitute Q and Q', and set sin(nθ) = 1 (we use the mode shape
            # only as a multiplicative factor here)
            self.Final_equation = smp.collect(
                self.Eq_r.subs(
                    [
                        (smp.diff(self.Q, self.r), self.Q_sols_r),
                        (self.Q, self.Q_sols),
                        (smp.sin(self.n * self.theta), 1),
                    ]
                ).doit(),
                smp.diff(self.U),
            )

            self.Fn2 = self.Final_equation

            # Dummy variables for U and its derivatives up to 4th order
            self.U4, self.U3, self.U2, self.U1, self.U0 = smp.symbols(
                "U_4 U_3 U_2 U_1 U_0", real=True, positive=True
            )
            self.dB, self.da = smp.symbols("B_d alpha_d", real=True, positive=True)

            # First, replace α'' by its base-state expression (self.dda),
            # and dp0/dr by its equilibrium expression (self.dpdr)
            self.Fn2 = self.Fn2.subs(
                [
                    (smp.diff(self.damage, self.r, 2), self.dda),
                    (smp.diff(self.p0, self.r, 1), self.dpdr),
                ]
            ).doit()

            # Replace U derivatives and damage/B derivatives by dummies
            self.Fn2 = self.Fn2.subs(
                [
                    (smp.diff(self.U, self.r, 4), self.U4),
                    (smp.diff(self.U, self.r, 3), self.U3),
                    (smp.diff(self.U, self.r, 2), self.U2),
                    (smp.diff(self.U, self.r, 1), self.U1),
                    (self.U, self.U0),
                    (smp.diff(self.damage, self.r, 2), self.dda),
                    (smp.diff(self.damage, self.r), self.da),
                    (smp.diff(self.B, self.r), self.dB),
                ]
            ).doit()

            # Coefficient of U⁽⁴⁾(r)
            dudr4_coeff = self.Fn2.subs(
                [
                    (self.B, 0),
                    (self.dB, 0),
                    (self.U4, 1),
                    (self.U3, 0),
                    (self.U2, 0),
                    (self.U1, 0),
                    (self.U0, 0),
                ]
            )

            # Isolate d⁴U/dr⁴
            self.dudr4 = -self.Fn2.subs([(self.U4, 0)]) / dudr4_coeff

            # Lambdified function for d⁴U/dr⁴ in the Gent case
            self.dudr4_f = smp.lambdify(
                (
                    self.r,
                    self.lbda,
                    self.n,
                    self.U0,
                    self.U1,
                    self.U2,
                    self.U3,
                    self.damage,
                    self.da,
                    self.B,
                    self.dB,
                    self.ell,
                    self.mu,
                    self.Gc,
                    self.Jm,
                    self.p0,
                ),
                self.dudr4,
            )
        

        # ------------------------------------------------------------
        # 9. Lambdify the base-state damage equilibrium equation α''(r)
        #
        # self.dda was already defined in set_equilibrium_equations as
        # the base-state damage ODE. Here we introduce dummy variables
        # so we can use it numerically as well.
        # ------------------------------------------------------------

        # Turningthe simbolic equilibrium damage equation into a function
        self.dda = self.dda.subs(
            [
                (smp.diff(self.U, self.r, 3), self.U3), 
                (smp.diff(self.U, self.r, 2), self.U2), 
                (smp.diff(self.U, self.r, 1), self.U1), 
                (self.U, self.U0)
            ]
        )
        self.dda = self.dda.subs(
            [
                (smp.diff(self.damage, self.r), self.da), 
                (smp.diff( self.B, self.r ), self.dB)
            ]
        )
        self.dda_f = smp.lambdify(
            (self.r, self.lbda, self.damage, self.da, self.ell, self.mu, self.Gc, self.Jm  ), 
            self.dda,
            )

    # ------------------------------------------------------------------
    # INCREMENTAL DAMAGE EQUILIBRIUM (dalphadr2_f)
    # ------------------------------------------------------------------
    def incremental_damage_equilibrium(self):
        """
        Derive the incremental damage equilibrium equation for the circumferential mode n.

        Starting point:
          - We have the incremental damage equilibrium for the perturbation β(r, θ)
            based on the pushed-forward reference functional (Eq. (79) in the SI).
          - We linearize around the base state and apply the same mode ansatz used
            in the elastic problem:
                v_r(r,θ)      =  U(r) sin(n θ)
                v_θ(r,θ)      =  V(r) cos(n θ)
                p¹(r,θ)       =  Q(r) sin(n θ)
                β(r,θ)        =  B(r) sin(n θ)

        Steps:
          1. Build the symbolic incremental damage equilibrium equation
             (δE/δβ = 0) using the geometric Laplacian and geometric coupling.
          2. Substitute the mode ansatz and eliminate V(r) using the
             incompressibility condition:
                 V(r) = (r U'(r) + U(r)) / n.
          3. Factor out the θ-dependence and rescale by r² λ² to obtain
             an ODE in r only.
          4. Solve algebraically for B''(r).
          5. Replace U and its derivatives by dummy variables U0, U1, U2, U3
             and B', α' by dummies dB, da.
          6. Lambdify B''(r) as dalphadr2_f.
        """
        
        # print("Setting incremental damage equilibrium (geometric Laplacian)...")

        # Geometric operator for the incremental damage, consistent with Eq. (79):
        #   0 = a''(α0) β W(F0)
        #       + a'(α0) (∂W/∂F(F0) : Γ F0)
        #       + (Gc/(c_w ℓ)) [ w''(α0) β
        #                        - 2 ℓ^2 div_x(F0 F0^T ∇_x β)
        #                        - 2 ℓ^2 div_x((Γ F0 F0^T + F0 F0^T Γ^T) ∇_x α0) ].
        #
        # We build the two geometric divergence terms symbolically in cylindrical
        # coordinates before substituting the mode ansatz.
        FFt0 = self.F * self.F.T

        # grad_x β = (∂β/∂r, (1/r) ∂β/∂θ)^T
        grad_beta = smp.Matrix(
            [
                smp.diff(self.damage_perturbation, self.r),
                1 / self.r * smp.diff(self.damage_perturbation, self.theta),
            ]
        )
        q_beta = FFt0 * grad_beta

        div_q_beta = (
            1 / self.r * smp.diff(self.r * q_beta[0], self.r)
            + 1 / self.r * smp.diff(q_beta[1], self.theta)
        )

        # grad_x α0 = (∂α0/∂r, 0)^T
        grad_alpha0 = smp.Matrix(
            [
                smp.diff(self.damage, self.r),
                0,
            ]
        )
        M_geo = self.Gamma * FFt0 + FFt0 * self.Gamma.T
        v_geo = M_geo * grad_alpha0

        div_geo = (
            1 / self.r * smp.diff(self.r * v_geo[0], self.r)
            + 1 / self.r * smp.diff(v_geo[1], self.theta)
        )

        self.Equilibrium_damage = (
            self.damage_perturbation
            * smp.diff(self.a, self.damage, 2)
            * self.strain_energy_density
            + smp.diff(self.a, self.damage) \
            * (self.dWdF.T * self.Gamma * self.F).trace()
            + self.Gc
            / (self.c_w * self.ell)
            * (
                smp.diff(self.w, self.damage, 2) * self.damage_perturbation
                - 2 * self.ell**2 * (div_q_beta + div_geo)
            )
        )

        # ------------------------------------------------------------
        # 2. Substitute mode ansatz into the damage equilibrium
        #
        #   v_r      = U(r) sin(n θ)
        #   v_θ      = V(r) cos(n θ)
        #   p¹       = Q(r) sin(n θ)
        #   β        = B(r) sin(n θ)
        # ------------------------------------------------------------
        self.ea = self.Equilibrium_damage.subs(
            [
                (self.vr, self.U * smp.sin(self.theta * self.n)),
                (self.vtheta, self.V * smp.cos(self.theta * self.n)),
                (self.p1, self.Q * smp.sin(self.theta * self.n)),
                (self.damage_perturbation, self.B * smp.sin(self.theta * self.n)),
            ]
        ).doit()


        # Enforce incompressibility on the incremental displacement:
        #   V(r) = (r U'(r) + U(r)) / n
        self.ea = self.ea.subs(
            [(self.V, (self.r * smp.diff(self.U, self.r) + self.U) / self.n)]
        ).doit()

        

        # ------------------------------------------------------------
        # 3. Factor out the θ-dependence and rescale by r² λ²
        #
        #   ea2 = ea / sin(nθ) * r² λ²
        #
        # This yields an ODE in r for B(r), with U and its derivatives
        # appearing as coefficients.
        # ------------------------------------------------------------
        self.ea2 = smp.simplify(
            self.ea / smp.sin(self.n * self.theta) * self.r**2 * self.lbda**2
        )
        if self.show_equations:
            print("Incremental damage equilibrium equation:")
            display(Math(r'\delta E / \delta \beta = ' + smp.latex(self.ea2) ))
        # ------------------------------------------------------------
        # 4. Solve algebraically for B''(r)
        # ------------------------------------------------------------
        self.sols = smp.solve(self.ea2, smp.diff(self.B, self.r, 2))


        # Replace α''(r) and λ'(r)
        self.dalphadr2 = self.sols[0].subs(
            [
                (smp.diff(self.damage, self.r, 2), self.dda),
                (smp.diff(self.lbda, self.r), self.lbda * (1 - self.lbda**2) / self.r),
            ]
        ).doit()
        if self.show_equations:
            print("B''(r) expression:") 
            display(Math(r"B''(r) = " + smp.latex(self.dalphadr2)))

        
        # ------------------------------------------------------------
        # 5. Replace U and B derivatives by dummy variables:
        #    U → U0, U' → U1, U'' → U2, U''' → U3,
        #    B' → dB, α' → da.
        # ------------------------------------------------------------

        self.dalphadr2 = self.dalphadr2.subs(
            [
                (smp.diff(self.U, self.r, 3), self.U3),
                (smp.diff(self.U, self.r, 2), self.U2),
                (smp.diff(self.U, self.r, 1), self.U1),
                (self.U, self.U0),
                (smp.diff(self.damage, self.r), self.da),
                (smp.diff(self.B, self.r), self.dB),
            ]
        )
        # print("variables:", self.dalphadr2.free_symbols )
        if self.show_equations:
            print("B''(r) with dummy variables:")       
            display(Math(r"B''(r) = " + smp.latex(self.dalphadr2)))

        # Lambdify B''(r)
        self.dalphadr2_f = smp.lambdify((self.r, self.lbda, self.n, self.U0, self.U1, self.U2, self.damage, self.da, self.B, self.dB, self.ell, self.mu, self.Gc , self.Jm), self.dalphadr2)


        # Example numeric evaluation (for debugging, commented out):
        # print("Example evaluation of dalphadr2_f:", self.dalphadr2_f(0.1, 1.0, 2, 0.1, 0.1, 1.0, 2.,1.0, 0.1, 0.1, 1.0, 1.0, 1.,100))
        
        # print("Incremental damage equilibrium set.")

    # ------------------------------------------------------------------
    # BOUNDARY CONDITIONS (Eij_f)
    # ------------------------------------------------------------------
    def boundary_conditions(self):
        """
        Derive and lambdify the boundary-condition coefficients E_ij.

        We consider two scalar boundary conditions at r = a, b:

          (1) Traction-free radial boundary condition:
                σ_rr¹ = 0  (incremental)
              which, in terms of the incremental fields, leads to
                BC1(U, U', U'', U''', B, B') = 0.

          (2) Shear-free / tangential boundary condition:
                σ_rθ¹ = 0
              leading to
                BC2(U, U', U'', U''', B, B') = 0.

        After substituting the mode ansatz and eliminating V and Q using the
        expressions derived in incremental_elastic_equilibrium, both BCs can be
        written in the generic form:

          BC1: E11 U + E12 U' + E13 U'' + E14 U''' + E15 B + E16 B' = 0
          BC2: E21 U + E22 U' + E23 U'' + E24 U''' + E25 B + E26 B' = 0

        This method:
          - Computes BC1 and BC2 symbolically.
          - Extracts the coefficients E_ij by taking the appropriate partial
            derivatives w.r.t. U, U', U'', U''', B, B'.
          - Lambdifies E_ij for later use in the numerical boundary matching.

        Side effects:
          - Sets symbolic attributes: E11, ..., E26.
          - Sets numeric functions:   E11_f, ..., E26_f.
          - Sets dummy "third-row" coefficients E3*_f used in the ODE system
            assembly (currently 0 or 1 as a placeholder).
        """

        # ------------------------------------------------------------
        # 1. First boundary condition: radial traction-free
        #
        # BC1_functional corresponds to σ_rr¹ = 0 in terms of v_r, p¹, etc.
        # ------------------------------------------------------------
        self.BC1_functional = -self.p1 + 2 * smp.diff(self.vr, self.r) / self.lbda**2

        # Substitute mode ansatz:
        #   v_r = U(r) sin(n θ)
        #   v_θ = V(r) cos(n θ)
        #   p¹  = Q(r) sin(n θ)
        self.BC1 = self.BC1_functional.subs(
            [
                (self.vr, self.U * smp.sin(self.theta * self.n)),
                (self.vtheta, self.V * smp.cos(self.theta * self.n)),
                (self.p1, self.Q * smp.sin(self.theta * self.n)),
            ]
        ).doit()

        # Incompressibility for the incremental displacement:
        #   V(r) = (r U'(r) + U(r)) / n
        self.BC1 = self.BC1.subs(
            [(self.V, (self.r * smp.diff(self.U, self.r) + self.U) / self.n)]
        ).doit()

        # Eliminate Q in favor of Q_sols from θ-equilibrium
        self.BC1 = self.BC1.subs(self.Q, self.Q_sols)

        # Replace λ'(r) by λ(r) (1 − λ(r)²) / r
        self.BC1 = self.BC1.subs(
            [(smp.diff(self.lbda, self.r), self.lbda * (1 - self.lbda**2) / self.r)]
        ).doit()

        # Normalize BC1 by factoring out sin(n θ) and a scaling factor involving
        # λ, n, (1 − α), r, etc. to obtain a clean polynomial in U, U', U'', U'''.
        self.BC1 = smp.collect(
            smp.expand(
                -self.lbda**2
                * self.n**2
                * (self.damage - 1)
                * self.BC1
                / (smp.sin(self.n * self.theta))
            ),
            smp.diff(self.U),
        )

        # ------------------------------------------------------------
        # 2. Second boundary condition: shear-free (σ_rθ¹ = 0)
        #
        # BC2_functional corresponds to continuity of tangential traction:
        #   ∂v_r/∂θ − v_θ + r ∂v_θ/∂r = 0  (in incremental form).
        # ------------------------------------------------------------
        self.BC2_functional = (
            smp.diff(self.vr, self.theta) - self.vtheta + self.r * smp.diff(self.vtheta, self.r)
        )

        # Use Q_sols_r already computed in incremental_elastic_equilibrium, if it
        # appears in BC2 (for completeness we keep this substitution here):
        self.BC2 = self.BC2_functional.subs(smp.diff(self.Q, self.r), self.Q_sols_r)

        # Substitute the mode ansatz:
        self.BC2 = self.BC2.subs(
            [
                (self.vr, self.U * smp.sin(self.theta * self.n)),
                (self.vtheta, self.V * smp.cos(self.theta * self.n)),
                (self.p1, self.Q * smp.sin(self.theta * self.n)),
            ]
        ).doit()

        # Again enforce V(r) = (r U'(r) + U(r)) / n
        self.BC2 = self.BC2.subs(
            [(self.V, (self.r * smp.diff(self.U, self.r) + self.U) / self.n)]
        ).doit()

        # Replace λ'(r) by λ(r) (1 − λ(r)²) / r
        self.BC2 = self.BC2.subs(
            [(smp.diff(self.lbda, self.r), self.lbda * (1 - self.lbda**2) / self.r)]
        ).doit()

        # Normalize by cos(n θ) and n to isolate the coefficients of U and its
        # derivatives in BC2
        self.BC2 = smp.collect(
            smp.expand((self.n * self.BC2 / smp.cos(self.n * self.theta)).simplify()),
            smp.diff(self.U),
        )

        # ------------------------------------------------------------
        # 3. Extract coefficients E_ij from BC1 and BC2
        # ------------------------------------------------------------
        # BC1: E11 U + E12 U' + E13 U'' + E14 U''' + E15 B + E16 B' = 0
        self.E11 = smp.factor(self.BC1.coeff(smp.diff(self.U, self.r, 0), 1))
        self.E12 = smp.factor(self.BC1.coeff(smp.diff(self.U, self.r, 1), 1))
        self.E13 = smp.factor(self.BC1.coeff(smp.diff(self.U, self.r, 2), 1))
        self.E14 = smp.factor(self.BC1.coeff(smp.diff(self.U, self.r, 3), 1))
        self.E15 = smp.factor(self.BC1.coeff(self.B, 1))
        self.E16 = smp.factor(self.BC1.coeff(smp.diff(self.B, self.r, 1), 1))

        # BC2: E21 U + E22 U' + E23 U'' + E24 U''' + E25 B + E26 B' = 0
        self.E21 = smp.factor(self.BC2.coeff(smp.diff(self.U, self.r, 0), 1))
        self.E22 = smp.factor(self.BC2.coeff(smp.diff(self.U, self.r, 1), 1))
        self.E23 = smp.factor(self.BC2.coeff(smp.diff(self.U, self.r, 2), 1))
        self.E24 = smp.factor(self.BC2.coeff(smp.diff(self.U, self.r, 3), 1))
        self.E25 = smp.factor(self.BC2.coeff(self.B, 1))
        self.E26 = smp.factor(self.BC2.coeff(smp.diff(self.B, self.r, 1), 1))

        # Optional: inspect the symbolic forms (commented out to avoid IPython dependency)
        # display(self.E11, self.E12, self.E13, self.E14, self.E15, self.E16)
        # display(self.E21, self.E22, self.E23, self.E24, self.E25, self.E26)

        # ------------------------------------------------------------
        # 4. Lambdify the E_ij coefficients for numerical use
        #
        # E1* depend on (n, r, λ, α, α', μ, Jm, [p0 for E15]).
        # E2* depend only on (n, r, λ) in the current formulation.
        # ------------------------------------------------------------

        # Row 1 coefficients: depend on base-state fields
        self.E11_f = smp.lambdify(
            (self.n, self.r, self.lbda, self.damage, smp.diff(self.damage, self.r), self.mu, self.Jm),
            self.E11,
        )
        self.E12_f = smp.lambdify(
            (self.n, self.r, self.lbda, self.damage, smp.diff(self.damage, self.r), self.mu, self.Jm),
            self.E12,
        )
        self.E13_f = smp.lambdify(
            (self.n, self.r, self.lbda, self.damage, smp.diff(self.damage, self.r), self.mu, self.Jm),
            self.E13,
        )
        self.E14_f = smp.lambdify(
            (self.n, self.r, self.lbda, self.damage, smp.diff(self.damage, self.r), self.mu, self.Jm),
            self.E14,
        )
        self.E15_f = smp.lambdify(
            (self.n, self.r, self.lbda, self.damage, smp.diff(self.damage, self.r), self.mu, self.Jm, self.p0),
            self.E15,
        )
        self.E16_f = smp.lambdify(
            (self.n, self.r, self.lbda, self.damage, smp.diff(self.damage, self.r), self.mu, self.Jm),
            self.E16,
        )

        # Row 2 coefficients: only depend on (n, r, λ) in this derivation
        self.E21_f = smp.lambdify((self.n, self.r, self.lbda), self.E21)
        self.E22_f = smp.lambdify((self.n, self.r, self.lbda), self.E22)
        self.E23_f = smp.lambdify((self.n, self.r, self.lbda), self.E23)
        self.E24_f = smp.lambdify((self.n, self.r, self.lbda), self.E24)
        self.E25_f = smp.lambdify((self.n, self.r, self.lbda), self.E25)
        self.E26_f = smp.lambdify((self.n, self.r, self.lbda), self.E26)

        # Row 3 coefficients: placeholders used when assembling the full 6×6
        # ODE system in the numerical solver (e.g., for compatibility conditions).
        self.E31_f = 0
        self.E32_f = 0
        self.E33_f = 0
        self.E34_f = 0
        self.E35_f = 0
        self.E36_f = 1
    
