from .base import Solver
from .tools import *
import numpy as np
from scipy.special import binom
from fenics import *
from fenics_adjoint import *
from collections import OrderedDict
import matplotlib.pyplot as plt

# TODO:
#   issue: needs to work for observations other than terminal time
#   solution: let solve() take Phi as argument,
#             make J an attribute and integrate over time
#             first suppose we have full temporal observations of Phi

class WrightFisherOnePop(Solver):
    """ 
    WF 1pop diffusion approximation for AFD with reference eff pop size Nref
    Equation solution is u(x,t) where 

        du/dtau = 1/2 d^2/dx^2(nu * x(1-x) * u) - d/dx(gamma x(1-x) u)

    where tau = t/(2Nref)   time units: 2Nref generations
          nu = Nref/N       (inverse) relative effective population size
          gamma = 2Nref s   scaled (relative to Nref) selection coefficient
    """

    def __init__(self, nx=99, T=2, dt=0.1, deg=1, loss='l2', debug_mode=False, grad_test=False, nu_inv=True):
        self.loss = loss
        self.debug_mode = debug_mode
        self.grad_test = grad_test
        self.nu_inv = nu_inv
        # discretize temporal domain
        self.dt = Constant(dt)   # time step
        self.T = float(T)        # terminal time
        self.ts = self._time_steps() # time dict keys

        # discretize frequency domain
        mesh = UnitIntervalMesh(int(nx))
        x = mesh.coordinates()
        #x[:] = cosinify(x) # refine near boundaries
        x[:] = sigmoidish(x, k=-1.4) # refine near boundaries
        V = FunctionSpace(mesh, 'P', deg)

        # initial condition: concentration c, initial freq p
        u_0 = 'c * exp(-pow(c * (x[0] - p), 2) / 2.) / sqrt(2. * pi)'
        u_0 = Expression(u_0, c=100, p=0.5, degree=2)
        u_0 = project(u_0, V)
        normalise(u_0)
        self.u_0 = u_0

        # time dependent solutions dict
        self.u_t = OrderedDict()
        self.u_t[0] = u_0.copy(deepcopy=True)

        # time-dependent controls: relative eff pop size nu=N/Nref, gamma=2Nref s
        self.ctrls = OrderedDict()
        for t in self.ts:
            self.ctrls[t] = [Constant(10.), Constant(0.)]
        self.nu, self.gamma = ctrls[dt]
        
        # trial and test function, geometric term
        u = TrialFunction(V)
        v = TestFunction(V)
        self.u_n = Function(V)
        xx = Expression('0.5 * x[0] * (1 - x[0])', degree=2, domain=mesh)

        # variational forms
        a = u * v * dx
        if nu_inv:
            a += self.dt * inner(grad(xx * self.nu * u), grad(v)) * dx # drift
        else:
            a += self.dt * inner(grad(xx / self.nu * u), grad(v)) * dx # drift
        a -= self.dt * 2.0 * xx * self.gamma * u * grad(v)[0] * dx # selection
        self.L = self.u_n * v * dx

        super().__init__(V, a)

    def solve(self, theta):
        # initialize u
        self.u_n.assign(self.u_0)
        # update drift and selection controls
        nu, gamma = [float(x) for x in theta]
        self.nu.assign(nu)
        self.gamma.assign(gamma)

        # time index
        dt = float(self.dt) # time step
        digs = len(str(dt).split('.')[-1]) # sig digs for u_t keys
        t = dt

        # solution, boundary conds, solve
        u = Function(self.V)
        bc = DirichletBC(self.V, Constant(0.), 'on_boundary')
        for t in self.ts:
            # solve for density u and add to dict
            solve(self.a == self.L, u, bc)
            self.u_t[t] = u.copy(deepcopy=True)
            
            # update initial condition and controls
            self.u_n.assign(u)
            self.nu.assign(ctrls[t][0])
            self.gamma.assign(ctrls[t][1])
            
       # while t <= self.T:
       #     # solve for density u and add to dict
       #     solve(self.a == self.L, u, bc)
       #     self.u_t[t] = u.copy(deepcopy=True)
       #     
       #     # update initial condition u_0 and time t
       #     self.u_n.assign(u)
       #     t += dt
       #     t = round(t, digs)

        # time independent controls
        controls = [Control(self.nu), Control(self.gamma)]
        return u, controls

    def _time_steps(self):
        dt = float(self.dt) # initial timestep
        digs = len(str(dt).split('.')[-1]) # number of sig digits
        t = dt
        ts = [t]
        while t <= self.T:
            t += dt 
            t = round(t, digs)
            ts.append(t)
        return ts

    def solve_adjoint(self, u, data, controls):
        # data are assumed to be on vertices
        Phi = Function(self.V)
        Phi.vector().set_local(data)
        
        if self.loss == 'l2':
            J = 0.5 * assemble(inner(u - Phi, u - Phi) * dx)
        elif self.loss == 'l1':
            J = assemble(abs(u -  Phi) * dx)
        if self.debug_mode:
            print(f'J = {J}')
            # plot u and Phi
            self.plot_u_and_Phi(u, Phi)

        # compute gradients
        dJdnu, dJdgamma = compute_gradient(J, controls)

        if self.grad_test:
            h = [Constant(0.001), Constant(0.001)]
            Jhat = ReducedFunctional(J, controls)
            conv_rate = taylor_test(Jhat, [self.nu, self.gamma], h)
        dJdnu = dJdnu.values().item()
        dJdgamma = dJdgamma.values().item()
        grads = [dJdnu, dJdgamma]
        return grads

    def get_SFS(self, u, sample_size, derived_alleles):
        bino = 'pow(x[0], d) * pow(1-x[0], n-d)'
        binom_meas = Expression(bino, n=sample_size, d=derived_alleles, degree=2)
        SFS = binom(sample_size, derived_alleles) * assemble(binom_meas * u * dx)
        return SFS

    def plot_u_and_Phi(self, u, Phi):
        # get np arrays
        if not isinstance(u, np.ndarray):
            x = u.function_space().tabulate_dof_coordinates().copy()
            u = u.vector().get_local()
        if not isinstance(Phi, np.ndarray):
            x = Phi.function_space().tabulate_dof_coordinates().copy()
            Phi = Phi.vector().get_local()

        fig, ax = plt.subplots(dpi=150)
        ax.plot(x, Phi, label='truth')
        ax.plot(x, u, label='prediction')
        plt.legend()
        plt.show()
        plt.close()

    def loss_contour(self, fig, ax, theta_true):
        """ Contour plot of loss around true theta """
        nu_true, gamma_true = theta_true
        if not self.nu_inv:
            nu_min, nu_max = nu_true/3, 3*nu_true
        else:
            nu_min, nu_max = 0, 3*nu_true
        gamma_min, gamma_max = min(gamma_true-1, -1), max(gamma_true+1, 1)
        
        # make contour plot
        ranges = (nu_min, nu_max, gamma_min, gamma_max)
        self._get_tensors(theta_true, ranges, 15)
        lines = np.linspace(np.min(self.losses), np.max(self.losses), 20)
        cont = ax.contourf(self.nu_tensor, self.gamma_tensor, self.losses, lines)
        cbar = fig.colorbar(cont)
        cbar.set_label(r'loss $\mathcal{J}$', rotation=90)

        # make optimum point
        ax.scatter(nu_true, gamma_true, c='r', s=10**2)

        # other options
        ax.set_xlim([nu_min, nu_max])
        ax.set_ylim([gamma_min, gamma_max])
        ax.set_xlabel(r'$\nu$')
        ax.set_ylabel(r'$\gamma$')
        return fig, ax

    def _get_loss(self, theta_true, theta_hat):
        """ Compute mse loss between u_theta_hat and Phi = u_theta """
        Phi, _ = self.solve(theta_true)
        u, _ = self.solve(theta_hat)
        # l1 and l2 loss
        if self.loss == 'l1':
            J = assemble(abs(u - Phi) * dx)
        elif self.loss == 'l2':
            J = 0.5 * assemble(inner(u - Phi, u - Phi) * dx)
        return J

    def _get_tensors(self, theta_true, ranges, contour_res):
        """ Given ranges for nu and gamma and some resolution
        computes the loss tensor as a function of nu and gamma """
        # get ranges and make lattice
        nu_min, nu_max, gamma_min, gamma_max = ranges
        nu = np.linspace(nu_min, nu_max, contour_res)
        gamma = np.linspace(gamma_min, gamma_max, contour_res)
        nu_tensor, gamma_tensor = np.meshgrid(nu, gamma)

        # compute loss for each lattice point
        losses = np.zeros((contour_res, contour_res))
        for idx, _ in np.ndenumerate(nu_tensor):
            nu = nu_tensor[idx]
            gamma = gamma_tensor[idx]
            losses[idx] = self._get_loss(theta_true, [nu, gamma])

        # save as attributes
        self.nu_tensor = nu_tensor
        self.gamma_tensor = gamma_tensor
        self.losses = losses


if __name__ == '__main__':
    set_log_active(False)
    n = 100
    solver = WrightFisherOnePop(n-1, T=1, grad_test=True)
    theta = [20., -0.01]
    u, controls = solver.solve(theta)
    Phi = np.random.randn(n)
    grads = solver.solve_adjoint(u, Phi, controls)
    print(grads)

