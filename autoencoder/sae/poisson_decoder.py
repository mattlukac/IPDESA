"""
FEniCS decoder: Poisson equation with Dirichlet conditions.
Test problem is chosen to give an exact solution at all nodes of the mesh.
  -Laplace(u) = f    on [0,1]
         u(0) = u_0  
         u(1) = u_1
            f = constant
The above formulation gives an exact solution of
         u(x) = -(f/2)x^2 + (f/2 + u_1 - u_0)x + u_0
"""

from fenics import *
import numpy as np

class Decoder():
    """Decoder class for poisson SAE.
    Contains attributes u0, u1, f (Pois eq parameters)
    and error_L2, error_max: the L2 norm error between the 
    exact solution and estimated solution, and the maxabs
    error between the two.
    """
    def __init__(self, u0, u1, f):
        self.u0 = u0
        self.u1 = u1
        self.f = f

    def solve(self, soln_dim):
        """Solves constant force 1D Poisson equation
        and returns the solution as a numpy array
        with length soln_dim.
        """
        # Create mesh and define function space
        mesh = UnitIntervalMesh(soln_dim-1) 
        V = FunctionSpace(mesh, 'P', 1)

        # Define boundary condition
        u_D = Expression('x[0]==0 ? u0 : u1', 
                u0=self.u0, 
                u1=self.u1, 
                degree=2)

        def boundary(x, on_boundary):
            return on_boundary

        bc = DirichletBC(V, u_D, boundary)

        # Define variational problem
        u = TrialFunction(V)
        v = TestFunction(V)
        f = Constant(self.f)
        a = grad(u)[0]*grad(v)[0]*dx
        L = f*v*dx

        # Compute solution
        u = Function(V)
        solve(a == L, u, bc)

        # Compute L2 error
        u_e = Expression('-f*x[0]*x[0]/2.0 + (f/2 + u1 - u0)*x[0] + u0', 
                u0=self.u0,
                u1= self.u1,
                f= self.f,
                degree=2)
        self.error_L2 = errornorm(u_e, u, 'L2')

        # Compute maximum error at vertices
        vertex_values_u_e = u_e.compute_vertex_values(mesh)
        vertex_values_u = u.compute_vertex_values(mesh)
        self.error_max = np.max(np.abs(vertex_values_u_e - vertex_values_u))

        return vertex_values_u

