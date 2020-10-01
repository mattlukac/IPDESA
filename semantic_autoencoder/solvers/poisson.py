from semantic_autoencoder.solvers.base import Solver
import numpy as np
from fenics import *
from fenics_adjoint import *


class Poisson(Solver):

    def __init__(self, nx):
        mesh = UnitIntervalMesh(nx)
        V = FunctionSpace(mesh, 'P', 1)
        u = TrialFunction(V)
        self.v = TestFunction(V)
        a = inner(grad(u), grad(self.v)) * dx
        super().__init__(mesh, V, a)

    def solve(self, theta):
        c, b0, b1 = theta
        c = Constant(c)
        L = c * self.v * dx

        u_D = Expression('x[0] == 0 ? b0: b1', b0=b0, b1=b1, degree=1)
        u_D = project(u_D, self.V)
        bc = DirichletBC(self.V, u_D, 'on_boundary')

        u = Function(self.V)
        solve(self.a == L, u, bc)
        controls = [Control(c), Control(u_D)]
        return u, controls

    def solve_adjoint(self, out, data, controls):
        Phi = Function(self.V)
        Phi.vector().set_local(data[self.dofs])
        J = assemble(0.5 * inner(out - Phi, out - Phi) * dx)

        dJdc, dJdb = compute_gradient(J, controls)
        dJdc = dJdc.values().item()

        dJdb = dJdb.compute_vertex_values(self.mesh)
        grads = [dJdc, dJdb[0], dJdb[-1]]
        return grads


if __name__ == '__main__':
    set_log_active(False)
    # test solver
    n = 500
    solver = Poisson(n-1)
    theta = [2., -1., 1.]
    u, controls = solver.solve(theta)
    u_np = u.compute_vertex_values(mesh)
 #   print(u_np)

    # test adjoint solver
    x = np.linspace(0., 1., n)
    theta_true = [4., 1., -1.]
    kappa, beta0, beta1 = theta_true
    Phi = -kappa / 2 * x ** 2
    Phi += (kappa / 2 + beta1 - beta0) * x 
    Phi += beta0
    grads = solver.solve_adjoint(u, Phi, controls)
    print('fenics grad\n', grads)

    # numerical gradient
    eps = 1e-12
    def numerical_grad(eps, idx):
        # perturb theta
        theta_eps = theta.copy()
        theta_eps[idx] += eps 

        # solve with perturbed theta
        u_eps, controls_eps = solver.solve(theta_eps)
        u_eps = u_eps.compute_vertex_values(mesh)

        # compute functionals
        coords = mesh.coordinates()
        delta_x = coords[1] - coords[0]
        J = 0.5 * np.sum((u_np - Phi) ** 2) * delta_x
        J_eps = 0.5 * np.sum((u_eps - Phi) ** 2) * delta_x
        grad = (J_eps - J) / eps
        return grad.item()

    num_grads = []
    for idx in range(3):
        num_grads.append(numerical_grad(eps, idx))
    print('numerical grad\n', num_grads)
    print('analytic grad\n', [-1./60, -5./12, 1./4])



