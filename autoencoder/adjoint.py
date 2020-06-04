import numpy as np 
from scipy import sparse
from scipy.sparse.linalg import spsolve 
import warnings
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
from matplotlib.colors import ListedColormap
plt.rcParams.update({'font.size': 28,
                     'legend.handlelength': 3})


class Poisson:
    """
    Class for computing the L2 loss gradient
    wrt PDE model parameters for the Poisson equation,
    via the adjoint method.
    """

    def __init__(self, kappa, bc, resolution=99):
        self.dx = 1. / resolution    # step size dx
        self.U = resolution - 1      # interior soln dimension
        self.T = 1                   # parameter dimension
        self.kappa = kappa           # true constant force term
        self.b0 = bc[0]              # left boundary condition
        self.b1 = bc[1]              # right boundary condition
        self.Lap = self.Laplacian()  # discrete Laplacian

    def Laplacian(self):
        """
        Calculates and returns the discrete Laplacian
        stored as a scipy.sparse matrix
        """
        diags = [1., -2., 1.] 
        diags = [x / self.dx ** 2 for x in diags]
        Delta = sparse.diags(diagonals=diags,
                             offsets=[-1, 0, 1], 
                             shape=(self.U, self.U),
                             format='csr')
        return Delta

    def get_gradient(self, c):
        """
        Computes and returns the L2 loss gradient
        This is done by first computing the adjoint of dJ/du
        then solving Lap(lambda) = dJ/du* for lambda
        and finally computing grad(J) = -lambda* dF/dc
        """
        # compute observed u and u_hat
        Phi = self.solver(self.kappa)
        u = self.solver(c)
        
        # solve linear adjoint system, compute gradient
        dLdu = (u - Phi) * self.dx
        lamb = spsolve(self.Lap, dLdu)
        dFdc = np.ones((self.U, self.T))
        gradient = -lamb.T.dot(dFdc)
        return gradient.item()

    def solver(self, c):
        """
        Computes the solution u given force parameter c
        Acts as a proxy for the PDE solver
        """
        x = np.linspace(self.dx, 1. - self.dx, self.U)
        x = x.reshape((self.U, 1))
        soln = np.zeros((self.U, 1))
        # add quad, linear, intercept terms
        soln += -c / 2 * x ** 2
        soln += (c / 2 + self.b1 - self.b0) * x
        soln += self.b0 

        return soln

    def gradient_descent(self, c_init, gamma, tol, max_iter=500):
        """
        Performs gradient descent to converge to the true force param
        Inputs same as self.plot_gradient_descent():
            c_init - the initial guess at kappa
            gamma - the learning rate
            tol - acceptable distance from kappa
            max_iter - bound for number of iterations

        Returns values of c during convergence and number of iterations
        """
        c_new = c_init 
        dist = abs(c_new - self.kappa)
        cs = [c_new]
        iter_count = 0

        # perform the descent until close enough or reach max iterations
        while iter_count < max_iter and tol < dist:
            gradient = self.get_gradient(c_new)
            c_new -= gamma * gradient 
            cs.append(c_new)
            dist = abs(c_new - self.kappa)
            iter_count += 1
        if iter_count == max_iter:
            warnings.warn('Tolerance not reached. Consider changing lr')

        return cs, iter_count

    ################
    # PLOT METHODS #
    ################
    def plot_gradient_descent(self, c_init, gamma, tol, max_iter=500):
        """
        Performs and plots gradient descent.
        Inputs same as self.gradient_descent():
            c_init - initial guess at kappa
            gamma - learning rate
            tol - acceptable distance from kappa
            max_iter - bound for number of iterations
        """
        cs, _ = self.gradient_descent(c_init, gamma, tol, max_iter)
        indices = [i + 1 for i in range(len(cs))]
        fig, ax = plt.subplots(1, 1, figsize=(18,10), dpi=200)
        ax.plot(indices, cs, linewidth=3)
        ax.set_xlabel('Iterations')
        ax.set_ylabel(r'$c$')
        plt.axhline(self.kappa, c='r', linestyle='dashed')
        plt.legend(['converging ' + r'$c$', r'$\kappa$'])
        plt.title('Learning Rate ' + r'$\gamma = %.1f$' % gamma)
        plt.show()
        plt.close()

    def plot_learning_rates(self, c_init, lr_bounds, tol, max_iter=500):
        """
        Plots number of iterations to convergence
        as a function of learning rate given by lr_bounds
        """
        # collect data
        lr_min, lr_max = lr_bounds
        num_descents = 100    # number of times to perform grad descent
        learning_rates = np.linspace(lr_min, lr_max, num_descents)
        data = dict()
        for lr in learning_rates:
            _, i_count = self.gradient_descent(c_init, lr, tol, max_iter)
            data[lr] = i_count
        lr, counts = zip(*sorted(data.items()))
        lr_opt = lr[np.argmin(counts)]

        # now plot the data
        fig, ax = plt.subplots(1, 1, figsize=(18,10), dpi=200)
        ax.plot(lr, counts, linewidth=2)
        ax.set_xlabel('Learning Rate ' + r'$\gamma$')
        ax.set_ylabel('iterations to convergence (%d max)' % max_iter)
        ax.set_title(r'%d iterations at $\gamma = %.2f$' 
                     % (np.min(counts), lr_opt))
        plt.show()
        plt.close()


#############################
#     BOUNDARY CONTROL      #
############################# 
class PoissonBC(Poisson): 
    """ 
    Computes the gradient of L2 loss with respect to 
    force term c and boundary conditions b0 and b1 
    This is performed by solving the adjoint equation
    obtained by setting the first variation of the Lagrangian
    with respect to u equal to 0
    """

    def __init__(self, theta_true, resolution=99):
        self.theta_true = theta_true   # force term and boundary conds
        self.T = len(theta_true)       # number of parameters
        self.dx = 1. / resolution      # step size
        self.U = resolution - 1        # solution interior dimension
        Laplacian = Poisson.Laplacian  # inherit Laplacian method
        self.Lap = self.Laplacian()   # store Laplacian
        self.Phi = self.solver(self.theta_true) # observed data

    def solver(self, theta):
        """
        Computes the solution u given force parameter and boundary conds
        Acts as a proxy for the PDE solver
        """
        x = np.linspace(self.dx, 1. - self.dx, self.U)
        x = x.reshape((self.U, 1))
        soln = np.zeros((self.U, 1))
        c, b0, b1 = theta  # unpack theta
        soln += -c / 2 * x ** 2        # quadratic term
        soln += (c / 2 + b1 - b0) * x  # linear term
        soln += b0                     # intercept

        return soln

    def loss_contour(self, ranges):
        """
        Given ranges of theta=(c, b0, b1)
        compute a mesh grid for the loss J: R^3 -> R
        """

        # unpack theta ranges 
        x_min, x_max, y_min, y_max, z_min, z_max = ranges 
        num_x, num_y, num_z = [100, 100, 100]
        x = np.linspace(x_min, x_max, num_x)
        y = np.linspace(y_min, y_max, num_y)
        z = np.linspace(z_min, z_max, num_z)
        x_tensor, y_tensor, z_tensor = np.meshgrid(x, y, z)
        losses = self.get_loss(x_tensor, y_tensor, z_tensor)
        return x_tensor, y_tensor, z_tensor, losses

    def get_loss(self, c_tensor, b0_tensor, b1_tensor):
        """
        Given meshgrids X, Y, Z each of shape (num_x, num_y, num_z)
        compute the L2 loss J =  0.5 * sum( (u - Phi) ** 2 )
        For each combination of c, b compute u and then J
        The loss is a function of (c, b0, b1) so this function 
        returns a numpy array of dim = 4 with shape
               (num_c, num_b0, num_b1, loss)
        """
        Phi = self.solver(self.theta_true)     # true solution
        num_c, num_b0, num_b1 = c_tensor.shape # number of each parameter
        num_thetas = num_c * num_b0 * num_b1   # total number of thetas
        losses = np.zeros(c_tensor.shape)             # array to store thetas
        # compute losses
        for idx, _ in np.ndenumerate(losses):
            c = c_tensor[idx]
            b0 = b0_tensor[idx]
            b1 = b1_tensor[idx]
            u = self.solver([c, b0, b1])
            losses[idx] = 0.5 * np.sum( (u - Phi) ** 2 )
        return losses 

    def get_gradient(self, theta):
        """
        Computes gradient using Lagrangian adjoint equations
        """

        u = self.solver(theta)
        # dL/du* has shape U by 1
        dJdu = u - self.Phi

        # solve for the adjoint variable lambda
        lamb = spsolve(self.Lap, dJdu)

        # now that we have lambda, solve for mu0 and mu1
        mu0 = lamb[0] / self.dx
        mu1 = lamb[-1] / self.dx

        # now dJ/dtheta = -lambda* dF/dtheta - mu* dG/dtheta
        #               = (-lambda*, mu0*, mu1*)
        dFdtheta = np.zeros((self.U, self.T))
        dFdtheta[:,0] = 1.

        dGdtheta = np.zeros((1, self.T))
        dGdtheta[:,1] = mu0
        dGdtheta[:,2] = mu1

        dJdtheta = -lamb.T.dot(dFdtheta) - dGdtheta
        return dJdtheta

    def gradient_descent(self, theta_init, gamma, tol, max_iter=500):
        """
        Performs gradient descent to converge to the true theta
        Inputs same as self.gradient_descent():
            theta_init - initial guess at kappa, beta0, beta1
            gamma - learning rate
            tol - acceptable distance from kappa
            max_iter - bound for number of iterations
        """
        theta_init = np.array(theta_init)
        thetas = dict() # to store iterations
        err = np.sum(np.abs(theta_init - self.theta_true)) # L1 error
        iter_count = 0 
        thetas[iter_count] = np.array(theta_init.copy())
        theta_new = theta_init

        while iter_count < max_iter and tol < err:
            iter_count += 1
            grad = self.get_gradient(theta_new)
            theta_new -= gamma * grad.flatten()
            thetas[iter_count] = theta_new.copy() 
            err = np.sum(np.abs(theta_new - self.theta_true))

        if iter_count == max_iter:
            warnings.warn('Tolerance not reached. Consider increasing gamma')
        return thetas, iter_count

    ################
    # PLOT METHODS #
    ################
    def plot_gradient_descent(self, theta_init, gamma, tol, max_iter=500):
        """
        Performs and plots gradient descent.
        Inputs same as self.gradient_descent():
            theta_init - initial guess at kappa, beta0, beta1
            gamma - learning rate
            tol - acceptable distance from kappa
            max_iter - bound for number of iterations
        """
        # make names
        theta_names = [r'$c$', r'$b_0$', r'$b_1$']
        theta_true_names = [r'$\kappa$', r'$\beta_0$', r'$\beta_1$']
        
        # get thetas from gradient descent
        descent, _  = self.gradient_descent(theta_init, gamma, tol, max_iter)
        indices, theta = zip(*sorted(descent.items()))
        # stack thetas into array shape (num_iters, num_params)
        num_iters = len(theta)
        num_params = self.T 
        thetas = np.zeros((num_iters, num_params))
        for i in range(num_iters):
            thetas[i] = theta[i]

        fig, ax = plt.subplots(self.T, 1, 
                sharex=False, figsize=(18,15), dpi=200)
        for i in range(self.T):
            ax[i].plot(indices, thetas[:, i], linewidth=3)
            ax[i].hlines(self.theta_true[i], 
                         xmin=0, xmax=max(indices), 
                         linewidth=3, color='r', linestyle='dashed')
            #ax[i].set_title(theta_names[i])
            ax[i].legend([theta_names[i], theta_true_names[i]])
        fig.add_subplot(111, frame_on=False)
        plt.tick_params(labelcolor='none', bottom=False, left=False)
        plt.xlabel('Iterations')
        plt.title('Learning Rate ' + r'$\gamma = %.1f$' % gamma)
        plt.show()
        plt.close()

    def plot_learning_rates(self, theta_init, lr_bounds, tol, max_iter=500):
        """
        Plots number of iterations to convergence
        as a function of learning rate given by lr_bounds
        """
        # collect data
        lr_min, lr_max = lr_bounds
        num_descents = 100    # number of times to perform grad descent
        learning_rates = np.linspace(lr_min, lr_max, num_descents)
        data = dict()
        for lr in learning_rates:
            _, i_count = self.gradient_descent(theta_init, lr, tol, max_iter)
            data[lr] = i_count
        lr, counts = zip(*sorted(data.items()))
        lr_opt = lr[np.argmin(counts)]

        # now plot the data
        fig, ax = plt.subplots(1, 1, figsize=(18,10), dpi=200)
        ax.plot(lr, counts, linewidth=2)
        ax.set_xlabel('Learning Rate ' + r'$\gamma$')
        ax.set_ylabel('iterations to convergence (%d max)' % max_iter)
        ax.set_title(r'%d iterations at $\gamma = %.2f$' 
                     % (np.min(counts), lr_opt))
        plt.show()
        plt.close()

    def biplot_gradient_descent(self, theta_init, gamma, tol, max_iter=500):
        """
        Performs gradient descent and makes pairwise biplots
        Inputs same as self.gradient_descent():
            theta_init - initial guess at kappa, beta0, beta1
            gamma - learning rate
            tol - acceptable distance from kappa
            max_iter - bound for number of iterations
        """
        # make names
        theta_names = [r'$c$', r'$b_0$', r'$b_1$']
        theta_true_names = [r'$\kappa$', r'$\beta_0$', r'$\beta_1$']
        def ordered_pair(x, y):
            pair = '(' + x + ', ' + y + ')'
            return pair
        
        # get thetas from gradient descent
        descent, _  = self.gradient_descent(theta_init, gamma, tol, max_iter)
        indices, theta = zip(*sorted(descent.items()))
        # stack thetas into array shape (num_iters, num_params)
        num_iters = len(theta)
        num_params = self.T 
        thetas = np.zeros((num_iters, num_params))
        for i in range(num_iters):
            thetas[i] = theta[i]

        fig, ax = plt.subplots(self.T, 1, 
                sharex=False, figsize=(18,18), dpi=200)
        for i in range(self.T):
            j = (i + 1) % 3
            ax[i].plot(thetas[:, i] , thetas[:, j], linewidth=3)
            ax[i].scatter(self.theta_true[i], self.theta_true[j], 
                    s=15 ** 2, c='r')
            optim_name = ordered_pair(theta_true_names[i], 
                                      theta_true_names[j])
            curve_name = ordered_pair(theta_names[i], theta_names[j])
            ax[i].legend([curve_name, optim_name])
        fig.add_subplot(111, frame_on=False)
        plt.tick_params(labelcolor='none', bottom=False, left=False)
        title = '%d iterations to convergence, ' % num_iters
        title += r'$\gamma = %.1f$' % gamma
        plt.title(title)
        plt.show()
        plt.close()

    def plot_loss_contours(self, theta_init, gamma, tol, max_iter=500):
        """
        Performs gradient descent and makes pairwise contour biplots
        """
        pass

    def plot_curve_convergence(self, theta_init, gamma, tol, max_iter=500):
        """
        Performs and plots gradient descent.
        Inputs same as self.gradient_descent():
            theta_init - initial guess at kappa, beta0, beta1
            gamma - learning rate
            tol - acceptable distance from kappa
            max_iter - bound for number of iterations
        """
        # make names
        theta_names = [r'$c$', r'$b_0$', r'$b_1$']
        theta_true_names = [r'$\kappa$', r'$\beta_0$', r'$\beta_1$']
        
        # get thetas from gradient descent
        descent, _ = self.gradient_descent(theta_init, gamma, tol)
        indices, theta = zip(*sorted(descent.items()))

        # stack thetas into array shape (num_iters, num_params)
        num_iters = len(theta)
        num_curves = 10 
        step_size = int(num_iters/num_curves)
        num_params = self.T 
        thetas = np.zeros((num_curves, num_params))

        fig, ax = plt.subplots(1, 1, 
                sharex=False, figsize=(18,15), dpi=200)

        # make the plots
        x = np.linspace(self.dx, 1. - self.dx, self.U)
        Phi = self.solver(self.theta_true)
        Phi_plt = ax.plot(x, Phi, linewidth=3, linestyle='dashed', c='black')
        # plot the first n curves
        for i in range(num_iters):
            curve = self.solver(theta[i]).flatten() 
            alpha_i = 1./(i+2) + 0.01
            ax.plot(x, curve, 
                    linewidth=3, 
                    alpha=alpha_i, 
                    c='r')
        plt.legend([r'$\Phi$', r'$u$'], fontsize=30)
        plt.xlabel(r'$\Omega$', fontsize=33)
        plt.ylabel(r'$u$', fontsize=33)
        plt.title('Learning Rate ' + r'$\gamma = %.1f$' % gamma)
        return ax 
