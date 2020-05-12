from fenics import *
import numpy as np

T = 80.0	        # final time
num_steps = 80 	# number of time steps
dt = T / num_steps	# time step size
Nref = 100              # reference effective population size
N = 20                  # effective population size
nu = N/Nref             # relative effective population size
epsilon = 0.0           # distance from 0 and 1
p = 0.5                 # initial frequency

# Create mesh and define function space
nx = 100
mesh = IntervalMesh(nx, epsilon, 1-epsilon)
V = FunctionSpace(mesh, 'P', 1)

# Define boundary condition 
def boundary(x, on_boundary):
    return on_boundary

bc = DirichletBC(V, Constant(1.0), boundary)
#bc = []

# Define initial value
u_0 = Expression('100*exp(-pow(100*(x[0]-p),2))/sqrt(pi)', 
        N=N, p=p, degree=2)
#u_n = interpolate(u_D, V)
u_n = project(u_0, V)

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
xx = Expression('x[0]*(1-x[0])', degree = 2, domain=mesh)

a = u*v*dx + dt/(4*N) * grad(xx*u)[0]*grad(v)[0]*dx
L = u_n*v*dx

# Create VTK file for saving solution
vtkfile = File('solutions/1d_allele_freq_density_drift/solution.pvd')

# Time-stepping
u = Function(V)
t = 0
for _ in range(num_steps):
    
    # Update current time
    t += dt
    
    # Compute solution and integrate to 1
    solve(a == L, u, bc)
    u.vector()[:] = nx*u.vector()[:] / np.sum(u.vector()[:])

    # Save to file and plot solution
    vtkfile << (u, t)
    plot(u)

    # Update previous solution
    u_n.assign(u)

