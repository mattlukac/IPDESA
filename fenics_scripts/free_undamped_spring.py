from fenics import *
import numpy as np

omega = 3.0     # circular frequency


# Create mesh and define function space
nx = 100
mesh = IntervalMesh(nx, 0, 5)
V = FunctionSpace(mesh, 'P', 1)

# Define vanishing initial condition 
tol = 1e-14
def boundary(x):
    return near(x[0],0.0,tol)

bc = DirichletBC(V, Constant(0.0), boundary)

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
z = Constant(0.0) # initial velocity
omega2 = Constant(omega**2)

a = -grad(u)[0]*grad(v)[0]*dx + omega2*u*v*dx
L = z*v*dx
u = Function(V)
solve(a == L, u, bc)

# Create VTK file for saving solution
vtkfile = File('solutions/free_undamped_spring/solution.pvd')
vtkfile << u 
plot(u)

