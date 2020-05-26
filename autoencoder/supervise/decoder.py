"""
FEniCS decoder: Poisson equation with Dirichlet conditions.
Test problem is chosen to give an exact solution at all nodes of the mesh.
  -Laplace(u) = c    on [0,1]
         u(0) = u_0  
         u(1) = u_1
            c = constant
The above formulation gives an exact solution of
         u(x) = -(c/2)x^2 + (c/2 + u_1 - u_0)x + u_0

This decoder takes (u_0, u_1, c) as input and uses FEniCS
to solve the above PDE to approximate u with u_hat.
The targets are the analytic solution u.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
from tensorflow.keras.layers import *
from sklearn import preprocessing
from fenics import *
import numpy as np
import matplotlib.pyplot as plt
from . import equation

## Layers: Dense input with num_params nodes and Id weight, bias=0
##         Custom Tensorflow layer that uses FEniCS solver
##         Output of FEniCS is a Dense final layer

class Decoder(Layer):
    """
    Custom Keras layer that uses FEniCS to solve the Poisson equation.
    Contains attributes u0, u1, c (Pois eq parameters)
    and error_L2, error_max: the L2 norm error between the 
    exact solution and estimated solution, and the maxabs
    error between the two.

    Input shape: (batch_size, num_params)
    Output shape: (batch_size, output_dim)
    """
    def __init__(self, output_dim):
        super(Decoder, self).__init__(name='decoder', dtype=tf.float64)
        self.output_dim = output_dim

    def call(self, inputs):
        batch_size = len(inputs)
        outputs = tf.Variable(initial_value=tf.zeros((batch_size, self.output_dim)),
                trainable=False)
        solns = []
        for theta in inputs:
            solns.append(self.solver(theta))
        outputs.assign(solns)
        return outputs

    def solver(self, theta):
        """
        Solves constant force 1D Poisson equation
        and returns the solution as a numpy array
        with length soln_dim.
        """
        u0 = theta[0].numpy().item()
        u1 = theta[1].numpy().item()
        c = theta[2].numpy().item()

        # Create mesh and define function space
        mesh = UnitIntervalMesh(self.output_dim - 1)
        V = FunctionSpace(mesh, 'P', 1)

        # Define boundary condition
        u_D = Expression('x[0] == 0 ? u0: u1',
                        u0 = u0,
                        u1 = u1,
                        degree = 2)
        def boundary(x, on_boundary):
            return on_boundary
        bc = DirichletBC(V, u_D, boundary)

        # Define variational problem
        u = TrialFunction(V)
        v = TestFunction(V)
        c = Constant(c)
        a = grad(u)[0] * grad(v)[0] * dx
        L = c * v * dx

        # Compute solution
        u = Function(V)
        solve(a == L, u, bc)

        # Compute L2 error
        u_e = Expression('-c * x[0] * x[0] / 2.0 + (c / 2 + u1 - u0) * x[0] + u0',
                        u0=u0,
                        u1=u1,
                        c=c,
                        degree=2)
        #self.error_L2 = errornorm(u_e, u, 'L2')

        # Compute maximum error at vertices
        vertex_values_u_e = u_e.compute_vertex_values(mesh)
        vertex_values_u = u.compute_vertex_values(mesh)
        #self.error_max = np.max(np.abs(vertex_values_u_e - vertex_values_u))
        return vertex_values_u

    def plot_solns(self):
        # get inputs
        dataset = equation.Dataset('poisson')
        Theta_names = dataset.Eqn.Theta_names 
        Theta_batch = dataset.Eqn.Theta(3)
        Theta_batch = tf.convert_to_tensor(Theta_batch)
        x = dataset.Eqn.domain()
        # compute solutions
        solns = self.call(Theta_batch)
        solns = solns.numpy()

        # plot solutions
        plt.rcParams.update({'font.size': 20})
        num_plots = len(Theta_batch)
        commas = [',  ', ',  ', '']
        fig, ax = plt.subplots(nrows=1, ncols=num_plots,
                               figsize=(20,6),
                               dpi=200)
        for i in range(num_plots):
            title = ''
            for j, name in enumerate(Theta_names):
                title += r'%s' % Theta_names[j]
                title += r'$= %.2f$' % Theta_batch[i,j]
                title += commas[j]
            ax[i].plot(x, solns[i], linewidth=3)
            ax[i].set_xlabel(r'$x$')
            ax[i].set_title(title, fontsize=18)
        ax[0].set_ylabel(r'$\hat u$')

        plt.show(fig)
        plt.close(fig)
