""" Contains base Solver class for various PDE problems """
from abc import ABC, abstractmethod
import numpy as np
from fenics import *
from fenics_adjoint import *


class Solver(ABC):
    """ base solver class inherited by all solvers """

    def __init__(self, V, a):
        """ Creates mesh, function space V, and LHS of weak form a """
        self.mesh = V.mesh()
        self.V = V
        self.v2d = np.array(V.dofmap().dofs(self.mesh, 0))
        self.a = a

    @abstractmethod
    def solve(self, theta):
        """ Solve the PDE given parameters theta """

    def forward(self, theta_batch):
        """ 
        Computes batch solutions to the PDE
            arguments: theta batch array with shape (batch_size, theta_size)
            returns: solns array with shape (batch_size, soln_dim)
        """
        batch_solns = dict()
        self.controls = dict()
        self.solns = dict()

        # singleton batch case
        if theta_batch.ndim == 1:
            theta_batch = np.expand_dims(theta_batch, axis=0)
        batch_size = theta_batch.shape[0]

        # compute batch solutions
        for idx in range(batch_size):
            u, self.controls[idx] = self.solve(theta_batch[idx])
            self.solns[idx] = u
            batch_solns[idx] = u.compute_vertex_values(self.mesh)

        solns_array = np.array([u for u in batch_solns.values()])
        return solns_array

    @abstractmethod
    def solve_adjoint(out, data, controls):
        """ Compute loss J and loss gradients dJ/dtheta for a single theta """

    def backward(self, Phi_batch):
        """ 
        Compute batch gradients dJ/dtheta
            argument: data batch array with shape (batch_size, soln_dim)
            returns: loss grads dJ/dtheta with shape (batch_size, theta_size)
        """
        
        grads = dict()

        # if singleton batch, reshape to (1, Phi.shape)
        if Phi_batch.ndim == 1:
            Phi_batch = np.expand_dims(Phi_batch, axis=0)
        # solve adjoint batch
        batch_size = Phi_batch.shape[0]
        for idx in range(batch_size):
            grads[idx] = self.solve_adjoint(self.solns[idx], 
                                            Phi_batch[idx], 
                                            self.controls[idx])

        grads_array = np.array([g for g in grads.values()])
        return grads_array

