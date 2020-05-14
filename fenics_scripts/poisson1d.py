"""
FEniCS tutorial demo program: Poisson equation with Dirichlet conditions.
Test problem is chosen to give an exact solution at all nodes of the mesh.
  -Laplace(u) = f    on [0,1]
         u(0) = u_0  
         u(1) = u_1
            f = constant
The above formulation gives an exact solution of
         u(x) = -(f/2)x^2 + (f/2 + u_1 - u_0)x + u_0
"""

from fenics import *
import matplotlib.pyplot as plt
import numpy as np

# Define parameters f, u0, u1
f = -2.0
u0 = 1.0
u1 = 2.0

# Create mesh and define function space
mesh = UnitIntervalMesh(99) # so len(u) = 100
V = FunctionSpace(mesh, 'P', 1)

# Define boundary condition
u_D = Expression('x[0]==0 ? u0 : u1', u0=u0, u1=u1, degree=2)

def boundary(x, on_boundary):
    return on_boundary

bc = DirichletBC(V, u_D, boundary)

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
f = Constant(f)
a = grad(u)[0]*grad(v)[0]*dx
L = f*v*dx

# Compute solution
u = Function(V)
solve(a == L, u, bc)

# Plot solution and mesh
plot(u)
plot(mesh)

# Save solution to file in VTK format
vtkfile = File('solutions/poisson1d/solution.pvd')
vtkfile << u

# Compute error in L2 norm
u_e = Expression('1 + x[0]*x[0]', degree=2) # exact solution
error_L2 = errornorm(u_e, u, 'L2')

# Compute maximum error at vertices
vertex_values_u_e = u_e.compute_vertex_values(mesh)
vertex_values_u = u.compute_vertex_values(mesh)
np.save('solutions/poisson1d/vertex_values_u', vertex_values_u)
error_max = np.max(np.abs(vertex_values_u_e - vertex_values_u))

# Print errors
print('error_L2  =', error_L2)
print('error_max =', error_max)

# Hold plot
#plt.show()
