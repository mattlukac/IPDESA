from .base import Solver
import numpy as np
import multiprocessing as mp
from fenics import *
from fenics_adjoint import *


class Poisson(Solver):

    # TODO parallelize solving 
    # perhaps have mesh(comm) as init argument
    # and parallelize in base solver
    def __init__(self, nx):
        mesh = UnitIntervalMesh(MPI.comm_self, nx)
        V = FunctionSpace(mesh, 'P', 1)
        u = TrialFunction(V)
        self.v = TestFunction(V)
        a = inner(grad(u), grad(self.v)) * dx
        super().__init__(V, a)

    def solve(self, theta):
        """ 
        Returns list of tuples (u, controls)
        """
        c, b0, b1 = theta
        c = Constant(c)
        L = c * self.v * dx

        u_D = Expression('x[0] == 0 ? b0: b1', b0=b0, b1=b1, degree=1)
        u_D = interpolate(u_D, self.V) # projection results in inexact bd vals
        bc = DirichletBC(self.V, u_D, 'on_boundary')

        u = Function(self.V)
        solve(self.a == L, u, bc)
        controls = [Control(c), Control(u_D)]
        return u, controls

    def solve_adjoint(self, u_controls, data):
        def per_u(u_controls):
            Phi_W = Function(self.W)
            Phi_W.vector().set_local(data[self.v2d])
            Phi_V = project(Phi_W, self.V)

            Phi = Function(self.V)
            Phi.assign(Phi_V)
            J = 0.5 * assemble(inner(u - Phi, u - Phi) * dx)

            dJdc, dJdb = compute_gradient(J, controls)
            dJdc = dJdc.values().item()

            dJdb = fenics_to_numpy(dJdb)
            grads = [dJdc, dJdb[0], dJdb[-1]]
            return grads
        pool = mp.Pool()
        grads = pool.map(per_u, data)
        return grads


def fenics_to_numpy(f):
    idxs = np.argsort(f.function_space().tabulate_dof_coordinates(), axis=0)
    return f.vector().get_local()[idxs.flatten()]

if __name__ == '__main__':
    set_log_active(False)
    # test solver
    n = 500
    solver = Poisson(n-1)
    theta = [2., -1., 1.]
    u, controls = solver.solve(theta)
    u_np = fenics_to_numpy(u)

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
        u_eps = fenics_to_numpy(u_eps)

        # compute functionals
        coords = solver.mesh.coordinates().copy()
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




