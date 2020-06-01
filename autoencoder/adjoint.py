import numpy as np 
from scipy import sparse
from scipy.sparse.linalg import spsolve 
import warnings
import matplotlib.pyplot as plt 
plt.rcParams.update({'font.size': 22})


class Poisson:
    """
    Class for computing the L2 loss gradient
    wrt PDE model parameters for the Poisson equation,
    via the adjoint method.
    """

    def __init__(self, theta, u0=None, u1=None, resolution=99):
        self.resolution = resolution # domain resolution
        self.dx = 1. / resolution    # step size dx
        self.U = resolution - 1      # interior soln dimension
        self.theta = theta
        self.Lap = self.Laplacian()
        # check if theta = (c, u0, u1)
        if isinstance(theta, list):
            self.T = len(theta)      # number of parameters
            self.bc = False          # bdry conds need learnin 
            self.c = theta[0]        # parameter c
            self.u0 = theta[1]       # left boundary value
            self.u1 = theta[2]       # right boundary value
        # theta = c, so boundary conditions are known
        elif u0 and u1 is not None:
            self.T = 1               # c is only parameter
            self.bc = True           # bdry conds known
            self.theta = theta       # parameter c
            self.u0 = u0             # left boundary value
            self.u1 = u1             # right boundary value

    def Laplacian(self):
        # discrete Laplacian
        diags = [1., -2., 1.] 
        diags = [x / self.dx ** 2 for x in diags]
        Delta = sparse.diags(diagonals=diags,
                             offsets=[-1, 0, 1], 
                             shape=(self.U, self.U),
                             format='csr')
        return Delta


    def get_gradient(self, theta_hat):
        """
        Computes and returns the L2 loss gradient
        This is done by first computing the adjoint of dL/du
        then solving Lap(v) = dL/du* for v
        and finally computing grad(L) = - v* dF/dtheta
        """
        # first check for consistency in thetas
        if not self.bc:
            assert isinstance(theta_hat, (list, np.ndarray))
            c, u0, u1 = theta_hat

        # compute observed u and u_hat
        u_obs = self.solver(self.theta)
        u_hat = self.solver(theta_hat)
        
        # solve linear adjoint system, compute gradient
        if self.bc:
            dLdu_adj = 2. * (u_hat - u_obs) * self.dx
            self.Lap = self.Laplacian()
            v = spsolve(self.Lap, dLdu_adj)
            dFdc = np.ones((self.U, 1))
            gradient = -v.T.dot(dFdc)
            return gradient.item()

        else:
            gradients = self.get_3d_gradient(theta_hat)
            return gradients

    def get_3d_gradient(self, theta_hat):

        assert isinstance(theta_hat, (list, np.ndarray))
        c, u0, u1 = theta_hat 

        u_hat = self.solver(theta_hat)
        u_obs = self.solver(self.theta)
        dLdu_adj = 2. * (u_hat - u_obs) * self.dx
        dFdc = np.ones((self.U, 1))
        dFdui = -1.0 
        
        # interior adjoint system solution
        dLdu_adj = dLdu_adj[1:-1]
        self.Lap = self.Laplacian()
        v = spsolve(self.Lap, dLdu_adj)
        gradient = -v.T.dot(dFdc)

        # left boundary adjoint system solution
        dLdu_adj0 = dLdu_adj[0]
        v0 = dLdu_adj0 / u0
        gradient0 = -v0 * dFdui
        
        # right boundary adjoint system solution
        dLdu_adj1 = dLdu_adj[-1]
        v1 = dLdu_adj0 / u1 
        gradient1 = -v1 * dFdui 

        gradients = np.concatenate((gradient, gradient0, gradient1))
        return gradients

    def solver(self, theta):
        """
        Computes the solution u given theta
        Acting as a proxy for the PDE solver
        """
        num_u = self.U + 2 * (not self.bc) # need bdry if not known
        u = np.zeros((num_u, 1))
        boundary = self.bc * self.dx  # dx if bc are known, 0 otherwise
        x = np.linspace(boundary, 1. - boundary, num_u)
        x = np.reshape(x, u.shape)
        if self.bc:
            assert isinstance(theta, float)
            u += -theta / 2 * x ** 2                 # quadratic term
            u += (theta / 2 + self.u1 - self.u0) * x # linear term
            u += self.u0                             # intercept
        else:
            assert isinstance(theta, (list, np.ndarray))
            c, u0, u1 = theta               # unpack parameter list
            u += -c / 2 * x ** 2            # quadratic term
            u += (c / 2 + u1 - u0) * x      # linear term
            u += u0                         # intercept
        return u 

    def gradient_descent(self, theta_init, gamma, tol, max_iter=500):
        """
        Performs gradient descent to converge to the true theta
        Inputs:
            theta_init - the initial guess at theta
            gamma - the learning rate
            tol - acceptable distance from true theta
            max_iter - bound for number of iterations
        """
        if self.bc:
            theta_new = theta_init 
            theta_true = self.theta
            theta_dist = abs(theta_new - theta_true)
        else:
            theta_new = np.array(theta_init)
            print('initial theta:', theta_new)
            theta_true = np.array(self.theta)
            print('true theta:', theta_true)
            theta_dist = np.sum(np.abs(theta_new - theta_true))
        thetas = [theta_new]
        i = 0
        while i < max_iter and tol < theta_dist:
            gradient = self.get_3d_gradient(theta_new)
            print(gradient)
            theta_new -= gamma * gradient 
            print(theta_new)
            thetas.append(theta_new)
            if self.bc:
                theta_dist = abs(theta_new - theta_true)
            else:
                theta_dist = np.sum(np.abs(theta_new - theta_true))
            i += 1
        if i == max_iter:
            warnings.warn('Tolerance not reached. Consider increasing max_iter')
        return thetas

    def plot_gradient_descent(self, theta_init, gamma, tol, max_iter=500):
        """
        Performs and plots gradient descent.
        Inputs:
            theta_init - initial guess at theta
            gamma - learning rate
            tol - acceptable distance from true theta
            max_iter - bound for number of iterations
        """
        thetas = self.gradient_descent(theta_init, gamma, tol, max_iter)
        indices = [idx + 1 for idx in range(len(thetas))]
        fig, ax = plt.subplots(1, 1, figsize=(18,10), dpi=200)
        ax.plot(indices, thetas, linewidth=3)
        ax.set_xlabel('Iterations')
        ax.set_ylabel(r'$\theta$')
        plt.axhline(self.theta, c='r', linestyle='dashed')
        plt.legend(['converging ' + r'$\theta$', 'true ' + r'$\theta$'])
        plt.title('Learning Rate ' + r'$\gamma = %.1f$' % gamma)
        plt.show()
        plt.close()
