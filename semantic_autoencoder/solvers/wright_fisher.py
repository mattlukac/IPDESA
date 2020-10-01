from semantic_autoencoder.solvers.base import Solver
import numpy as np
from fenics import *
from fenics_adjoint import *
from collections import OrderedDict


# TODO:
#   issue: solve_adjoint is returning the same gradient
#          for many observations, and sometimes returns 0
#   solution: let solve() take Phi as argument,
#             make J an attribute and integrate over time
#             first suppose we have full temporal observations of Phi


class WrightFisherOnePop(Solver):

    def __init__(self, nx, T):
        mesh = UnitIntervalMesh(nx)
        V = FunctionSpace(mesh, 'P', 1)
        self.dt = Constant(0.1)  # time step
        self.T = T     # terminal time
        self.u = TrialFunction(V)
        self.v = TestFunction(V)
        a = self.u * self.v * dx
        super().__init__(mesh, V, a)
        self.a_finalized = False

    def solve(self, theta):
        # set controls: eff pop size and selection coefficient
        N, gamma = theta
        N, gamma = Constant(N), Constant(gamma)

        # set up drift and selection terms
        xx = Expression('x[0] * (1 - x[0])', degree=1, domain=self.mesh)
        drift = xx / (4. * N) * self.u
        selection = gamma * xx * self.u

        # keep 'a' fixed after first call to solve()
        if not self.a_finalized:
            self.a += self.dt * inner(grad(drift), grad(self.v)) * dx 
            self.a -= self.dt * selection * grad(self.v)[0] * dx 
            self.a_finalized = True

        # Gaussian initial condition, frequencies centered at p
        u_0 = '100 * exp(-pow(100 * (x[0] - p), 2)) / sqrt(pi)'
        u_0 = Expression(u_0, p=0.5, degree=1)
        u_0 = interpolate(u_0, self.V)
        
        # weak form RHS 
        L = u_0 * self.v * dx

        # boundary conditions
        u = Function(self.V)
        bc = DirichletBC(self.V, Constant(0.5), 'on_boundary')

        # solve PDE
        delta_x = 1 / self.mesh.num_cells()
        t = float(self.dt)
        self.u_t = OrderedDict()
        self.u_t[0] = u_0.compute_vertex_values(self.mesh)

        while t <= self.T:
            # solve for u and normalize solution
            solve(self.a == L, u, bc)
            u.vector()[:] /= np.sum(u.vector()[:] * delta_x)
            self.u_t[t] = u.compute_vertex_values(self.mesh)
            
            # update u_0 and t
            u_0.assign(u)
            t += float(self.dt)

        controls = [Control(N), Control(gamma)]
        return u, controls

    def solve_adjoint(self, out, data, controls):
        # assemble loss functional
        Phi = Function(self.V)
        Phi.vector().set_local(data[self.dofs])
        J = assemble(0.5 * inner(out - Phi, out - Phi) * dx)

        # compute gradients
        dJdN, dJdgamma = compute_gradient(J, controls)
        dJdN = dJdN.values().item()
        dJdgamma = dJdgamma.values().item()
        grads = [dJdN, dJdgamma]
        return grads

if __name__ == '__main__':
    set_log_active(False)
    n = 100
    solver = WrightFisherOnePop(n-1, T=0.5)
    theta = [20., -0.01]
    u, controls = solver.solve(theta)
    u_np = u.compute_vertex_values(solver.mesh)
    print(solver.u_t)

    # test adjoint solver
#    Phi = np.random.randn(n)
#    grads = solver.solve_adjoint(u, Phi, controls)
#    print('fenics grads\n', grads)
#
#    # numerical gradient
#    eps = 1e-2
#    def numerical_grad(eps, idx):
#        # perturb theta
#        theta_eps = theta.copy()
#        theta_eps[idx] += eps
#
#        # solve with perturbed theta
#        u_eps, controls_eps = solver.solve(theta_eps)
#        u_eps = u_eps.compute_vertex_values(mesh)
#
#        # compute functionals
#        coords = mesh.coordinates()
#        delta_x = coords[1] - coords[0]
#        J = 0.5 * np.sum((u_np - Phi) ** 2) * delta_x 
#        J_eps = 0.5 * np.sum((u_eps - Phi) ** 2) * delta_x
#        grad = (J_eps - J) / eps 
#        return grad.item()
#
#    num_grads = []
#    grad_ratio = []
#    for idx in range(2):
#        num_grads.append(numerical_grad(eps, idx))
#        grad_ratio.append(grads[idx] / num_grads[idx])
#    print('numerical grads\n', num_grads)
#    print('grad ratio\n', grad_ratio)
#
