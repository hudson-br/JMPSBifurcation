####################################################################

"""
Legacy FEniCS (dolfin) implementation used to generate the FEM datasets for
the JMPS revision package accompanying:

  H. Borja da Rocha & T. Cohen,
  "Damage localization as a bifurcation phenomenon and the resulting fracture patterns in soft materials".

What this script does
---------------------
- Builds a 2D plane-strain model of a soft solid with an inner cavity.
- Uses a stabilized mixed formulation (u, p) for (nearly) incompressible elasticity
  coupled to a phase-field / gradient-damage variable α.
- Solves quasi-statically via an *alternate minimization* loop at each load increment:
      (i)  solve elasticity for (u, p) with α fixed,
      (ii) solve damage for α with (u, p) fixed,
      (iii) iterate until convergence (or max iterations).

Outputs and what is included in the revision package
----------------------------------------------------
This script can write full-field visualization files (e.g. XDMF) *and* reduced
text outputs. For the JMPS revision package we only distribute the reduced
outputs (small `.txt` tables and a JSON parameter file) needed to reproduce
the figures, and we omit very large `.h5`/`.xdmf` time-series files.

See `FEM/output/` and the notebook `notebooks/data_analysis.ipynb` for how these
reduced files are post-processed into the plots shown in the manuscript.
"""

from dolfin import *
# from mshr import *
from ufl import rank
# Import python script
# import src.LogLoading as LogLoad
import math
import os
import shutil
import sympy
import time
import numpy as np
import matplotlib.pyplot as plt
import json

import meshio

from mesh import create_mesh, create_ellipsoid, create_perturbed_void
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# Parameters for DOLFIN and SOLVER
# ----------------------------------------------------------------------------
set_log_level(LogLevel.WARNING)  # 20 Information of general interest

# Set some dolfin specific parameters
parameters["form_compiler"]["representation"]="uflacs"
parameters["form_compiler"]["optimize"]=True
parameters["form_compiler"]["cpp_optimize"]=True
parameters["form_compiler"]["quadrature_degree"]=2
info(parameters,True)








# Parameters of the solvers for displacement and damage (alpha-problem)
# -----------------------------------------------------------------------------
# Parameters of the nonlinear newton solver used for the displacement u-problem
solver_up_parameters  = {"newton_solver": {
                                         "maximum_iterations": 200,
                                         "absolute_tolerance": 1e-6,
                                         "relative_tolerance": 1e-6,
                                         "report": True,
                                         "relaxation_parameter": 1.0,
                                         "error_on_nonconvergence": True}}

# Parameters of the PETSc/Tao solver used for the alpha-problem
tao_solver_parameters = {"maximum_iterations": 100,
                         "line_search": "more-thuente",
                         "linear_solver": "cg",
                         "preconditioner" : "hypre_amg",
                         "method": "tron",
                         "gradient_absolute_tol": 1e-4,
                         "gradient_relative_tol": 1e-4,
                         "report": False,
                         "error_on_nonconvergence": True}

# Set up the solvers
solver_alpha  = PETScTAOSolver()
solver_alpha.parameters.update(tao_solver_parameters)
# info(solver_alpha.parameters, True) # uncomment to see available parameters

# Define the minimisation problem by using OptimisationProblem class
class DamageProblem(OptimisationProblem):
    def __init__(self):
        OptimisationProblem.__init__(self)
        self.total_energy = damage_functional
        self.Dalpha_total_energy = E_alpha
        self.J_alpha = E_alpha_alpha
        self.alpha = alpha
        self.bc_alpha = bc_alpha
    def f(self, x):
        self.alpha.vector()[:] = x
        return assemble(self.total_energy)
    def F(self, b, x):
        self.alpha.vector()[:] = x
        assemble(self.Dalpha_total_energy, b)
        for bc in self.bc_alpha:
            bc.apply(b)
    def J(self, A, x):
        self.alpha.vector()[:] = x
        assemble(self.J_alpha, A)
        for bc in self.bc_alpha:
            bc.apply(A)



# Element-wise projection using LocalSolver
def local_project(v, V, u=None):
    dv = TrialFunction(V)
    v_ = TestFunction(V)
    a_proj = inner(dv, v_)*dx
    b_proj = inner(v, v_)*dx
    solver = LocalSolver(a_proj, b_proj)
    solver.factorize()
    if u is None:
        u = Function(V)
        solver.solve_local_rhs(u)
        return u
    else:
        solver.solve_local_rhs(u)
        return

# Initial condition (IC) class
class InitialConditions(UserExpression):
    def eval(self, values, x):
        # Displacement u0 = (values[0], values[1])
        values[0] = 0.0             # Displacement in x direction
        values[1] = 0.0             # Displacement in y direction
        values[2] = 0.0             # Pressure
        # values[3] = 1.0             # Deformation gradient component: F_{33}
    def value_shape(self):
         return (3,)


# Set the user parameters

parameters.parse()
# ----------------------------------------------------------------------------
# User parameters (edit here)
# ----------------------------------------------------------------------------
userpar = Parameters("user")
userpar.add("mu", 1.0)              # Shear modulus
userpar.add("kappa", 10000)         # Bulk modulus
userpar.add("Gc", 1.0 )             # Fracture toughness
userpar.add("k_ell", 5.e-3)         # Residual stiffness
userpar.add("load_max", 1)          # Maximum loading (fracture can occur earlier)
userpar.add("load_steps", 1001)     # Steps in which loading from 0 to load_max occurs
userpar.add("hsize", 0.01)          # Element size in the center of the domain
userpar.add("ell_multi", 5)         # For definition of phase-field width
# Parse command-line options
userpar.parse()

# Constants: some parsed from user parameters
# ----------------------------------------------------------------------------
# Geometry parameters
hsize = userpar["hsize"]

# Zero body force
body_force = Constant((0., 0.))

# Material model parameters
mu    = userpar["mu"]           # Shear modulus
kappa = userpar["kappa"]        # Bulk Modulus
Gc    = userpar["Gc"]           # Fracture toughness
k_ell = userpar["k_ell"]        # Residual stiffness
# Damage regularization parameter - internal length scale used for tuning Gc
ell_multi = userpar["ell_multi"]
ell = Constant(ell_multi*hsize) 




Radius = 1.                  # Radius of the domain
R0 = [0.0 ,0.0, 0.0]        # Center of the domain
R_void = 0.2                # Radius of the void
ell = 0.10 #25 * R_void
hsize = ell / ell_multi
# hsize = 0.02

x0 = [0.0, 0.0, 0.0]        # Center of the void
R_refinement = 4 * R_void   # Region of refinement
h = 0.1;                    # Refinement of the outer boundary
h2 = hsize                  # Refinement of the inner boundary
h_base = h

aspect_ratio  = 1.05
Rx_void = R_void
Ry_void = R_void / aspect_ratio


# Perturbed circle void
wave_length = 3
perturbation = 0.2

if MPI.rank(MPI.comm_world) == 0:
  print("The kappa/mu: {0:4e}".format(kappa/mu))
  print("The mu/Gc: {0:4e}".format(mu/Gc))

# Number of steps
load_min = 0.0
load_max = userpar["load_max"]
load_steps = userpar["load_steps"]

# Numerical parameters of the alternate minimization scheme
maxiteration = 200         # Sets a limit on number of iterations
AM_tolerance = 1e-3

# Naming parameters for saving output
modelname = "BifurcationPaper"
meshname  = modelname + "-mesh.xdmf"
simulation_params = "Gc_%.1f_mu_%.1f_ell_%.1f_k/mu_%.0f_h_%.2f_S_%.0f_dt_%.1f" \
                    % (Gc, mu, ell, kappa/mu, hsize, load_steps, load_max)

mesh_type    = "circular"     # "circular", "ellipse" or "perturbed-circle"
filename = mesh_type
if mesh_type == "perturbed-circle":
    modelname = "Appendix/" + mesh_type + "_wave-length_%.0f_perturbation_%.2f"\
                            %(wave_length, perturbation)
damage_model = 'AT2'                # AT1 w(alpha) = alpha, AT2, w(alpha) = alpha**2


simulation_params = mesh_type + "_" + damage_model+ "C1_%.2f_C2_%.2f_R_void_%.2f_mu_%.2f_ell_%.3f_Gc_%.1f_ld-stps_%.0f_max-ld_%.2f" \
                    % (Gc/(mu*ell), ell/R_void, R_void, mu, ell, Gc, load_steps, load_max)

savedir   = "output/" + modelname + "/" + simulation_params + "/"

input_parameters = {'damage_model':damage_model,'mu': mu, 'ell': ell, 'Gc': Gc, 'k_ell':k_ell, 
                    'load_min': load_min,'load_max':load_max, 'load_steps':load_steps, 
                    'Radius': 1, 'R_void': R_void,  'ell_multi': ell_multi, 'hsize': hsize, 'h_base': h_base, 'R_refinement':R_refinement,
                    'mesh_type': mesh_type, 'wave_length': wave_length, 'perturbation': perturbation
                    }

print(input_parameters)



# For parallel processing - write one directory
if MPI.rank(MPI.comm_world) == 0:
    if os.path.isdir(savedir):
        shutil.rmtree(savedir)


############################################################
############################################################
############################################################

# ----------------------------------------------------------------------------
# Mesh generation
# ----------------------------------------------------------------------------
# We generate the mesh through the gmsh Python API (see FEM/mesh.py).
# The mesh is written to XDMF and then imported into dolfin.
# Refinement is concentrated near the cavity where damage localizes.
#
# NOTE (revision package): the repository ships only *reduced outputs*;
# you can regenerate full-field visualization outputs by rerunning this
# script locally with a legacy FEniCS installation.
# ----------------------------------------------------------------------------


# Create mesh
if mesh_type == "circular":
    create_mesh(mesh_type, Radius, R_void, R_refinement, R0, x0, h_base, hsize )
elif mesh_type == "ellipse":  
    create_ellipsoid(mesh_type, Radius, R_refinement, Rx_void, Ry_void, R0, x0, h_base, hsize )
elif mesh_type == "perturbed-circle":  
    create_perturbed_void(mesh_type, Radius, R0, R_void, wave_length, perturbation, R_refinement, h_base, hsize )
elif mesh_type == "double-void":
    create_double_void(mesh_type, Radius, R0, x0, R_void0, x1, R_void1 , R_refinement, h_base, hsize )


xdmf_name = filename + ".xdmf"

# I'm not sure why I need to pass from .msh to xdmf then read the xdmf, but that's how it works ...

msh = meshio.read(filename+ ".msh")

meshio.write(
    xdmf_name,
    meshio.Mesh(points=msh.points[:,:2], cells={"triangle": msh.cells_dict["triangle"]}),
)
mmesh = meshio.read(xdmf_name)
mesh = Mesh()
with XDMFFile(xdmf_name) as mesh_file:
    mesh_file.read(mesh)

# Boundaries
exterior_circle = CompiledSubDomain("near(pow(x[0],2) + pow(x[1],2), pow(%f,2), 1.e-2)"%Radius)
hole1 = CompiledSubDomain("near(pow(x[0] - {R0},2) + pow(x[1],2), pow({R},2), 1.e-3)".format(R0 = x0[0], R = R_void) )

boundaries = MeshFunction("size_t", mesh,1)
boundaries.set_all(0)
exterior_circle.mark(boundaries, 1)
hole1.mark(boundaries, 2)

############################################################
############################################################
############################################################

# Stabilization parameters
h = CellDiameter(mesh)      # Characteristic element length
varpi_ = 1.0                # Non-dimension non-negative stability parameter

# Loading and initialization of vectors to store data of interest

load_multipliers = np.linspace(load_min, load_max, load_steps)

# Initialization of vectors to store data of interest
energies   = np.zeros((len(load_multipliers), 4))
iterations = np.zeros((len(load_multipliers), 2))

# Variational formulation
# ----------------------------------------------------------------------------

# Tensor space for projection of stress
T_DG0 = TensorFunctionSpace(mesh,'DG',0)
# Create equal order function space for elasticity + damage
V_CG1 = VectorFunctionSpace(mesh, "Lagrange", 1)
CG1 = FunctionSpace(mesh, "Lagrange", 1)
V_CG1elem = V_CG1.ufl_element()
CG1elem = CG1.ufl_element()
# Stabilized mixed FEM for incompressible elasticity
MixElem = MixedElement([V_CG1elem, CG1elem])
# Define function spaces for displacement, pressure, V_u
V_u = FunctionSpace(mesh, MixElem)
# Define function space for damage in V_alpha
V_alpha = FunctionSpace(mesh, "Lagrange", 1)

# Define the function, test and trial fields for elasticity problem
w_p    = Function(V_u)
u_p    = TrialFunction(V_u)
v_q    = TestFunction(V_u)
(u, p) = split(w_p)     # Functions for (u, p)
(v, q) = split(v_q)   # Test functions for u, p 
# Define the function, test and trial fields for damage problem
alpha  = Function(V_alpha, name = "Damage")
dalpha = TrialFunction(V_alpha)
beta   = TestFunction(V_alpha)

# Define functions to save
PTensor = Function(T_DG0, name="Nominal Stress")
STensor = Function(T_DG0, name="Cauchy Stress")
FTensor = Function(T_DG0, name="Deformation Gradient")
JScalar = Function(CG1, name="Volume Ratio")


# Initial Conditions (IC)
#------------------------------------------------------------------------------
# Initial conditions are created by using the class defined and then
# interpolating into a finite element space
init = InitialConditions(degree=1)          # Expression requires degree def.
w_p.interpolate(init)                       # Interpolate current solution

# Dirichlet boundary condition
# --------------------------------------------------------------------

# expression to move inner surface x component
xmove = Expression("t*(x[0] - {R0}) /(pow(pow(x[0] - {R0},2) + pow(x[1],2), 0.5))".format(R0 = 0, R1 = Radius), t = 0,  degree=2)
# expression to move inner surface y component
ymove = Expression("t*x[1]/(pow(pow(x[0] - {R0},2) + pow(x[1],2), 0.5))".format(R0 = 0, R1= Radius), t = 0,  degree=2)


bc_ic_x = DirichletBC(V_u.sub(0).sub(0), xmove, boundaries, 1) # u0 move 
bc_ic_y = DirichletBC(V_u.sub(0).sub(1), ymove, boundaries, 1) # u0 move 


# Dirichlet boundary condition for a traction test boundary
bc_u = [bc_ic_x, bc_ic_y]

bcalpha_1 = DirichletBC(V_alpha, 0.0, boundaries, 1)

bc_alpha = [bcalpha_1]
bc_alpha = [ ]




# Kinematics
d = len(u)
print("Dimension", d)
I = Identity(d)             # Identity tensor
F = I + grad(u)             # Deformation gradient
C = F.T*F                   # Right Cauchy-Green tensor

print(C.ufl_shape)
# Plane stress invariants
Ic = tr(C) #+ (F33)**2
J = det(F)#*(F33)

# Define the energy functional of the elasticity problem
# --------------------------------------------------------------------
def w(alpha):           # Specific energy dissipation per unit volume
    return alpha**2

def a(alpha):           # Modulation function
    return (1.0-alpha)**2

def b_sq(alpha):        # b(alpha) = (1-alpha)^6 therefore we define b squared
    return (1.0-alpha)**3

def P(u, alpha):        # Nominal stress tensor
    return a(alpha)*mu*(F - inv(F.T)) - b_sq(alpha)*p*J*inv(F.T)

def Sigma(u, alpha):        # Cauchy stress tensor
    return F * P(u,alpha)/J

# Stabilization term
varpi  = project(varpi_*h**2/(2.0*mu), FunctionSpace(mesh,'DG',0))
# Elastic energy, additional terms enforce material incompressibility and regularizes the Lagrange Multiplier
elastic_energy    = (a(alpha)+k_ell)*(mu/2.0)*(Ic - 2.0 - 2.0*ln(J))*dx \
                    - b_sq(alpha)*p*(J-1.)*dx - 1/(2*kappa)*p**2*dx
external_work     = dot(body_force, u)*dx
elastic_potential = elastic_energy - external_work

# Line 1: directional derivative about w_p in the direction of v (Gradient)
# Line 2: Plane stress term
# Line 3-5: Stabilization terms
F_u = derivative(elastic_potential, w_p, v_q) \
    - varpi*b_sq(alpha)*J*inner(inv(C),outer(b_sq(alpha)*grad(p),grad(q)))*dx \
    - varpi*b_sq(alpha)*J*inner(inv(C),outer(grad(b_sq(alpha))*p,grad(q)))*dx \
    + varpi*b_sq(alpha)*inner(mu*(F-inv(F.T))*grad(a(alpha)),inv(F.T)*grad(q))*dx

# Compute directional derivative about w_p in the direction of u_p (Hessian)
J_u = derivative(F_u, w_p, u_p)

# Variational problem to solve for displacement and pressure
problem_up = NonlinearVariationalProblem(F_u, w_p, bc_u, J=J_u)
# Set up the solver for displacement and pressure
solver_up  = NonlinearVariationalSolver(problem_up)
solver_up.parameters.update(solver_up_parameters)
info(solver_up.parameters, True) # uncomment to see available parameters
# prm = solver_up.parameters



# Define the energy functional of the damage problem
# --------------------------------------------------------------------
# Initial (known) damage is an undamaged state
alpha_0 = interpolate(Expression("0.", degree=0), V_alpha)


# Define the specific energy dissipation per unit volume
z = sympy.Symbol("z", positive=True)
c_w = float(4 * sympy.integrate(sympy.sqrt(w(z)), (z, 0, 1)))
print("c_w", c_w)
# Define the phase-field fracture term of the damage functional
dissipated_energy = Gc/float(c_w)*(w(alpha)/ell + ell*dot(grad(alpha), grad(alpha)))*dx
damage_functional = elastic_potential + dissipated_energy

# Compute directional derivative about alpha in the direction of beta (Gradient)
E_alpha = derivative(damage_functional, alpha, beta)
# Compute directional derivative about alpha in the direction of dalpha (Hessian)
E_alpha_alpha = derivative(E_alpha, alpha, dalpha)

# Set the lower and upper bound of the damage variable (0-1)
alpha_lb = interpolate(Expression("0.", degree=0), V_alpha)

alpha_ub = interpolate(Expression("1.", degree=0), V_alpha)
for bc in bc_alpha:
    bc.apply(alpha_lb.vector())
    bc.apply(alpha_ub.vector())

# Split into displacement, pressure
(u, p) = w_p.split()
# Data file name
file_tot = XDMFFile(MPI.comm_world, savedir + "/results.xdmf")
# Saves the file in case of interruption
file_tot.parameters["rewrite_function_mesh"] = False
file_tot.parameters["functions_share_mesh"]  = True
file_tot.parameters["flush_output"]          = True
# Write the parameters to file
File(savedir+"/parameters.xml") << userpar

# Timing
timer0 = time.process_time()




#  loading and initialization of vectors to store time datas
# ----------------------------------------------------------------------------

#####################################

# Leaf directory 
directory = savedir+"/last_iteration_damage_bd_damage_p"
    
# Parent Directories 
parent_dir = os.getcwd()
    
# Path 
path = os.path.join(parent_dir, directory) 
    
# Create the directory 

try:
    os.makedirs(path, exist_ok = True)
    print("Directory '%s' created successfully" % directory)
except OSError as error:
    print("Directory '%s' can not be created" % directory)


energies = np.zeros((len(load_multipliers),4))
iterations = np.zeros((len(load_multipliers),2))

forces = np.zeros((len(load_multipliers),3))
displacements = np.zeros((len(load_multipliers),3))


tol = 1e-6  # avoid hitting points outside the domain
size_damage_profile = 201
y = np.linspace(R_void + tol, 1 - tol, size_damage_profile)
points = [(y_, 0) for y_ in y]  # 2D points

x_inner_boundary = np.linspace(- R_void, R_void, size_damage_profile)
Thetas = np.linspace(0, 2*np.pi, size_damage_profile)
inner_boundary = [(R_void * np.cos(theta) , R_void * np.sin(theta) ) for theta in Thetas]  # 2D points
outer_boundary = [(0.99*Radius * np.cos(theta) , 0.99*Radius * np.sin(theta) ) for theta in Thetas]  # 2D points

# inner_boundary = [(x, np.sqrt( R_void**2 - x**2)) for x in x_inner_boundary]  # 2D points
# inner_boundary = np.append(inner_boundary, [(x, - np.sqrt( R_void**2 - x**2)) for x in x_inner_boundary])

damage_profile = np.zeros((len(load_multipliers) + 1,size_damage_profile))
damage_profile[-1] = y

damage_boundary = np.zeros((len(load_multipliers) + 1, size_damage_profile))
damage_boundary_out = np.zeros((len(load_multipliers) + 1, size_damage_profile))
pressure_boundary_in = np.zeros((len(load_multipliers) + 1, size_damage_profile))
pressure_boundary_out = np.zeros((len(load_multipliers) + 1, size_damage_profile))
pressure_profile = np.zeros((len(load_multipliers) + 1,size_damage_profile))
pressure_profile[-1] = y

def postprocessing():
    plt.figure(i_t)
    # plt.colorbar(plot(alpha, range_min=0., range_max=1., title = "Damage at loading %.4f"%(t*load0)))

    damage_profile_ = np.array([alpha(point) for point in points])

    plt.plot(y, damage_profile_, '-', linewidth=2)
    plt.grid(True)
    plt.xlabel('$R$')
    plt.ylabel('damage')
    plt.title("Damage at loading %.4f"%(t))
    plt.savefig(directory+'/damage' + str(i_t).zfill(5) + '.png')
    plt.close(i_t)

    plt.figure(i_t + 1)
    if mesh_type == "circular":
        damage_boundary_ = np.array([alpha(point) for point in inner_boundary])
    
        plt.plot(Thetas, damage_boundary_, '-', linewidth=2)
        plt.grid(True)
        plt.xlabel(r'angular coordinate $\theta \in (0, 2*\pi)$')
        plt.ylabel('damage at $R = R_0$ ')
        plt.title("Damage at loading %.4f"%(t))
        plt.ylim([0., 1.])
        plt.savefig(directory+'/bd_damage' + str(i_t).zfill(5) + '.png')
        plt.close(i_t + 1)
        damage_boundary[i_t] = np.array([alpha(point) for point in inner_boundary])

        damage_boundary_out[i_t] = np.array([alpha(point) for point in outer_boundary])
        pressure_boundary_in[i_t] = np.array([p(point) for point in inner_boundary])
        pressure_boundary_out[i_t] = np.array([p(point) for point in outer_boundary])

    # Save number of iterations for the time step
    iterations[i_t] = np.array([t,i_t])
    # Calculate the energies
    elastic_energy_value = assemble(elastic_energy)
    surface_energy_value = assemble(dissipated_energy)

    if MPI.comm_world.rank == 0:
        print("\nEnd of timestep %d with load multiplier %g"%(i_t, t))
        print("\nElastic and surface energies: (%g,%g)"%(elastic_energy_value,surface_energy_value))
        print("-----------------------------------------")

    
    
    energies[i_t] = np.array([t, elastic_energy_value, 
                                 surface_energy_value, 
                                 elastic_energy_value + surface_energy_value])
    forces[i_t] = np.array([t, PTensor[0,0]((1,0)),  STensor[0,0]((1,0))])
    displacements[i_t] = np.array([t,u[0]((R_void,0)) , u[0]((1,0))])
    damage_profile[i_t] = np.array([alpha(point) for point in points])
    pressure_profile[i_t] = np.array([p(point) for point in points])

    

    # Save some global quantities as a function of the time
    np.savetxt(savedir+'/energies.txt', energies)
    np.savetxt(savedir+'/forces.txt', forces)
    np.savetxt(savedir+'/iterations.txt', iterations)
    np.savetxt(savedir+'/displacements.txt', displacements)
    np.savetxt(savedir+'/damage.txt', damage_profile)
    np.savetxt(savedir+'/boundary_damage.txt', damage_boundary)
    np.savetxt(savedir+'/boundary_damage_out.txt', damage_boundary_out)
    np.savetxt(savedir+'/pressure_boundary_in.txt', pressure_boundary_in)
    np.savetxt(savedir+'/pressure_boundary_out.txt', pressure_boundary_out)
    np.savetxt(savedir+'/pressure.txt', pressure_profile)

with open(savedir + 'input_file.txt', "w") as fp:
            json.dump(input_parameters, fp)

# Solving at each timestep
# ----------------------------------------------------------------------------
for (i_t, t) in enumerate(load_multipliers):
    xmove.t = 1*t
    ymove.t = 1*t
    # Structure used for one printout of the statement
    if MPI.rank(MPI.comm_world) == 0:
        print("\033[1;32m--- Starting of Time step {0:2d}: t = {1:4f} ---\033[1;m".format(i_t, t))

    # ----------------------------------------------------------------------------
    # Alternate minimization scheme
    # ----------------------------------------------------------------------------
    # At each load step t:
    #   1) solve elasticity for (u,p) with damage α fixed
    #   2) solve damage for α with (u,p) fixed
    #   3) iterate until ||α^{k+1} - α^{k}|| < AM_tolerance
    # This produces a quasi-static evolution and the energy split reported in the paper.

    # Alternate Mininimization scheme
    # -------------------------------------------------------------------------
    # Solve for u holding alpha constant then solve for alpha holding u constant
    iteration = 1           # Initialization of iteration loop
    err_alpha = 1.0         # Initialization for condition for iteration

    # Conditions for iteration
    while err_alpha > AM_tolerance and iteration < maxiteration:
        # Solve elastic problem
        solver_up.solve()
        # Solve damage problem with box constraint
        solver_alpha.solve(DamageProblem(), alpha.vector(), alpha_lb.vector(), alpha_ub.vector())
        # Update the alpha condition for iteration by calculating the alpha error norm
        alpha_error = alpha.vector() - alpha_0.vector()
        err_alpha = alpha_error.norm('linf')    # Row-wise norm
        # Printouts to monitor the results and number of iterations
        if MPI.rank(MPI.comm_world) == 0:
            print ("AM Iteration: {:3d},  alpha_error: {:>14.8f}, alpha_max: {:.8f}".format(iteration, err_alpha, alpha.vector().max()))
        # Update variables for next iteration
        alpha_0.assign(alpha)
        iteration = iteration + 1

    # Updating the lower bound to account for the irreversibility of damage
    alpha_lb.vector()[:] = alpha.vector()


    # Project to the correct function space
    local_project(P(u, alpha), T_DG0, PTensor)
    local_project(Sigma(u, alpha), T_DG0, STensor)
    local_project(F, T_DG0, FTensor)
    local_project(J, CG1, JScalar)

    # Rename for paraview
    u.rename("Displacement", "u")
    p.rename("Pressure", "p")
    # F33.rename("F33", "F33")

    # Write solution to file
    file_tot.write(alpha, t)
    file_tot.write(u, t)
    file_tot.write(p, t)
    # file_tot.write(F33, t)
    file_tot.write(PTensor,t)
    file_tot.write(STensor,t)
    file_tot.write(JScalar,t)

    postprocessing()

    # Update the displacement with each iteration
    xmove.t = 1*t
    ymove.t = 1*t

    # Post-processing
    # --------------------------------------------------------------------------
    # Save number of iterations for the time step
    iterations[i_t] = np.array([t, iteration])

    # Calculate the energies
    elastic_energy_value = assemble(elastic_energy)
    surface_energy_value = assemble(dissipated_energy)
    # Save time and energies to data array
    energies[i_t] = np.array([t, elastic_energy_value, surface_energy_value, \
                              elastic_energy_value+surface_energy_value])

    if MPI.rank(MPI.comm_world) == 0:
        print("\nEnd of timestep {0:3d} with load multiplier {1:4f}".format(i_t, t))
        print("\nElastic and Surface Energies: [{0:6f},{1:6f}]".format(elastic_energy_value, surface_energy_value))
        print("\nElastic and Surface Energies: [{},{}]".format(elastic_energy_value, surface_energy_value))
        print("-----------------------------------------")
        # Save some global quantities as a function of the time
        np.savetxt(savedir + '/stabilized-energies.txt', energies)
        np.savetxt(savedir + '/stabilized-iterations.txt', iterations)

# ----------------------------------------------------------------------------
print("elapsed CPU time: ", (time.process_time() - timer0))

# Plot energy and stresses
if MPI.rank(MPI.comm_world) == 0:
    p1, = plt.plot(energies[slice(None), 0], energies[slice(None), 1])
    p2, = plt.plot(energies[slice(None), 0], energies[slice(None), 2])
    p3, = plt.plot(energies[slice(None), 0], energies[slice(None), 3])
    plt.legend([p1, p2, p3], ["Elastic", "Dissipated", "Total"], loc="best", frameon=False)
    plt.xlabel('Displacement')
    plt.ylabel('Energies')
    plt.title('stabilized FEM')
    plt.savefig(savedir + '/stabilized-energies.pdf', transparent=True)
    plt.close()
