"""
2-dimensional wave equation

  u_t = c^2 Laplacian(u) + f  in a square domain
  u_D = 0                     on the boundary 3 by 3 square
  u_0 = xy(3-x)(3-y)          initial condition
  du_0= 0                     initial velocity

The initial condition u_0 is chosen as a hump.
"""

from __future__ import print_function
from fenics import *
import time

T = 10.0            # final time
num_steps = 100     # number of time steps
dt = T / num_steps  # time step size
c = 3.0             # wave constant
#A = 1.0            # wave amplitude multiplier

# Create mesh and define function space
nx = ny = 30
mesh = RectangleMesh(Point(0, 0), Point(3, 3), nx, ny)
V = FunctionSpace(mesh, 'P', 1)

# Define boundary condition
#def boundary(x, on_boundary):
#    return x[0] == 0 or x[0] == 3 or x[1] == 0 or x[1] == 3

bc = DirichletBC(V, Constant(0.0), 'on_boundary')

# Define initial value
u_0 = Expression('0.5*x[0]*x[1]*(3-x[0])*(3-x[1])',
                 degree=2)
u_n0 = project(u_0, V)
u_n1 = project(u_0, V)

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
C = Constant(c)
f = Expression('c*c*cos(c*t)*( x[0]*(3-x[0]) + x[1]*(3-x[1]) - 0.5*x[0]*x[1]*(3-x[0])*(3-x[1]) )',
               degree = 2, t = 0, c = c)

a = u*v*dx + dt**2 * C**2 * inner(grad(u), grad(v))*dx
L = Constant(2.0)*u_n1*v*dx - u_n0*v*dx + dt**2 * f*v*dx

# Create VTK file for saving solution
vtkfile = File('solutions/2d_wave1/solution.pvd')

# Time-stepping
u = Function(V)
t = 0
for n in range(num_steps):

    # Compute solution
    solve(a == L, u, bc)

    # Save to file and plot solution
    vtkfile << (u, t)
    plot(u)

    # Update previous solutions
    u_n0.assign(u_n1)
    u_n1.assign(u)

    # Update current time
    t += dt
    f.t = t
