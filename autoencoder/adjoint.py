import numpy as np 
from scipy import sparse
from scipy.sparse.linalg import spsolve 
import warnings
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
from matplotlib.colors import ListedColormap
plt.rcParams.update({'font.size': 28,
                     'legend.handlelength': 2})


class Poisson:
    """
    Class for computing the L2 loss gradient
    wrt PDE model parameters for the Poisson equation,
    via the adjoint method.
    """

    def __init__(self, kappa, bc, resolution=99):
        self.dx = 1. / resolution    # step size dx
        self.u_dim = resolution - 1  # interior soln dimension
        self.domain = np.linspace(
                self.dx, 
                1. - self.dx, 
                self.u_dim)          # Omega = (0, 1)
        self.num_params = 1          # parameter dimension
        self.kappa = kappa           # true constant force term
        self.b0 = bc[0]              # left boundary condition
        self.b1 = bc[1]              # right boundary condition
        self.Phi = self.solver(kappa)# observed data
        self.Lap = self.Laplacian()  # discrete Laplacian

    def solver(self, c):
        """
        Computes the solution u given force parameter c
        Acts as a proxy for the PDE solver
        """
        x = self.domain.reshape((self.u_dim, 1))
        soln = np.zeros((self.u_dim, 1))
        # add quad, linear, intercept terms
        soln += -c / 2 * x ** 2
        soln += (c / 2 + self.b1 - self.b0) * x
        soln += self.b0 

        return soln

    def Laplacian(self):
        """
        Calculates and returns the discrete Laplacian
        stored as a scipy.sparse matrix
        """
        diags = [1., -2., 1.] 
        diags = [x / self.dx ** 2 for x in diags]
        Delta = sparse.diags(diagonals=diags,
                             offsets=[-1, 0, 1], 
                             shape=(self.u_dim, self.u_dim),
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
        dFdc = np.ones((self.u_dim, self.num_params))
        gradient = -lamb.T.dot(dFdc)
        return gradient.item()

    def gradient_descent(self, c_init, gamma, tol, max_its=500):
        """
        Performs gradient descent to converge to the true force param
        Inputs same as self.plot_gradient_descent():
            c_init - the initial guess at kappa
            gamma - the learning rate
            tol - acceptable distance from kappa
            max_its - bound for number of iterations

        Returns dictionary of {num_iter: (c, u)} during convergence
        """
        dist = abs(c_init - self.kappa)
        u_init = self.solver(c_init).flatten()
        iter_count = 0
        data = dict()
        data[iter_count] = (c_init, u_init)

        # perform the descent until close enough or reach max iterations
        c_new = c_init
        u_new = u_init
        while iter_count < max_its and tol < dist:
            iter_count += 1
            gradient = self.get_gradient(c_new)
            c_new -= gamma * gradient 
            u_new = self.solver(c_new).flatten()
            data[iter_count] = (c_new, u_new)
            dist = abs(c_new - self.kappa)
        if iter_count == max_its:
            warnings.warn('Tolerance not reached. Consider changing lr')

        return data


    ################
    # PLOT METHODS #
    ################

    def insert_boundaries(self, u):
        x = self.domain
        # insert left boundary condition
        x = np.insert(x, 0, 0.)
        u = np.insert(u, 0, self.b0, axis=1)
        Phi = np.insert(self.Phi, 0, self.b0)
        # append right boundary condition
        x = np.append(x, 1.)
        Phi = np.append(Phi, self.b1)
        return x, u, Phi

    def plot_c(self, ax, iterations, c, gamma):
        max_its = max(iterations)
        ax.plot(iterations, c, color='g', linewidth=3)
        ax.set_xlabel('iterations')
        ax.set_ylabel(r'$c$')
        ax.hlines(self.kappa, xmin=0, xmax=max_its, 
                     color='k', linewidth=3, linestyle='dashed')
        ax.legend([r'$c$', r'$\kappa$'])
        ax.set_title('Learning Rate ' + r'$\gamma = %.1f$' % gamma)

    def plot_u(self, ax, u, max_its):
        x, u, Phi = self.insert_boundaries(u)

        # make plot
        u_init = np.append(u[0], self.b1)
        ax.plot(x, u_init, 
                label=r'$u$', color='C0', linewidth=3)   
        for i in range(max_its):
            alpha_i = 1./(i+2) + 0.01
            u_new = np.append(u[i+1], self.b1)
            ax.plot(x, u_new, 
                    linewidth=3, c='C0', alpha=alpha_i)
        ax.plot(x, Phi, 
                label=r'$\Phi$', linewidth=3, linestyle='dashed', c='k')
        ax.set_xlabel(r'$\Omega$')
        ax.set_ylabel(r'$u$')
        ax.legend()

    def plot_gradient_descent(self, c_init, gamma, tol, max_its=500):
        """
        Performs and plots gradient descent.
        Inputs same as self.gradient_descent():
            c_init - initial guess at kappa
            gamma - learning rate
            tol - acceptable distance from kappa
            max_its - bound for number of iterations
        """
        # do get cs and iterations from gradient descent
        data = self.gradient_descent(c_init, gamma, tol, max_its)
        # unzip dictionary
        iterations, c_and_u = zip(*sorted(data.items()))
        max_its = max(iterations)

        # build c and u for plotting
        c = np.zeros((max_its + 1,))
        u = np.zeros((max_its + 1, self.u_dim))
        for i in range(max_its + 1):
            c[i], u[i] = c_and_u[i]
        
        # make plots
        fig, ax = plt.subplots(2, 1, figsize=(15, 15), dpi=200)
        fig.subplots_adjust(hspace=0.25)
        self.plot_c(ax[0], iterations, c, gamma) # parameter space convergence
        self.plot_u(ax[1], u, max_its)           # solution space convergence
        plt.show()
        plt.close()

    def plot_learning_rates(self, c_init, lr_bounds, tol, max_its=500):
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
            iters = self.gradient_descent(c_init, lr, tol, max_its)
            data[lr] = len(iters)
        lr, counts = zip(*sorted(data.items()))
        lr_opt = lr[np.argmin(counts)]

        # now plot the data
        fig, ax = plt.subplots(1, 1, figsize=(18,10), dpi=200)
        ax.plot(lr, counts, linewidth=4, color='C1')
        ax.set_xlabel('Learning Rate ' + r'$\gamma$')
        ax.set_ylabel('iterations to convergence (%d max)' % max_its)
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
        self.dx = 1. / resolution          # step size
        self.u_dim = resolution - 1            # solution interior dimension
        self.num_params = len(theta_true)           # number of parameters
        self.theta_true = theta_true       # force term and boundary conds
        self.Phi = self.solver(theta_true) # observed data
        Laplacian = Poisson.Laplacian      # inherit Laplacian method
        self.Lap = self.Laplacian()        # store Laplacian

    def solver(self, theta, boundary=False):
        """
        Computes the solution u given force parameter and boundary conds
        Acts as a proxy for the PDE solver
        """
        if boundary:
            u_dim = self.u_dim + 2
        else:
            u_dim = self.u_dim 
        x = np.linspace(self.dx, 1. - self.dx, u_dim)
        x = x.reshape((u_dim, 1))
        soln = np.zeros((u_dim, 1))
        c, b0, b1 = theta              # unpack theta
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
        losses = self.get_loss_tensor(x_tensor, y_tensor, z_tensor)
        return x_tensor, y_tensor, z_tensor, losses

    def get_loss_tensor(self, c_tensor, b0_tensor, b1_tensor):
        """
        Given meshgrids X, Y, Z each of shape (num_x, num_y, num_z)
        compute the L2 loss J =  0.5 * sum( (u - Phi) ** 2 )
        For each combination of c, b compute u and then J
        The loss is a function of (c, b0, b1) so this function 
        returns a numpy array of dim = 4 with shape
               (num_c, num_b0, num_b1, loss)
        """
        num_c, num_b0, num_b1 = c_tensor.shape # number of each parameter
        num_thetas = num_c * num_b0 * num_b1   # total number of thetas
        losses = np.zeros(c_tensor.shape)             # array to store thetas
        # compute losses
        for idx, _ in np.ndenumerate(losses):
            c = c_tensor[idx]
            b0 = b0_tensor[idx]
            b1 = b1_tensor[idx]
            u = self.solver([c, b0, b1])
            losses[idx] = 0.5 * np.sum( (u - self.Phi) ** 2 )
        return losses 

    def get_loss(self, theta, Phi):
        u = self.solver(theta)
        return 0.5 * np.sum((u - Phi) ** 2) * self.dx

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
        dFdtheta = np.zeros((self.u_dim, self.num_params))
        dFdtheta[:,0] = 1.

        dGdtheta = np.zeros((1, self.num_params))
        dGdtheta[:,1] = mu0
        dGdtheta[:,2] = mu1

        dJdtheta = -lamb.T.dot(dFdtheta)*self.dx - dGdtheta
        return dJdtheta

    def gradient_descent(self, theta_init, gamma, tol, noise=False, max_its=500):
        """
        Performs gradient descent to converge to the true theta
        Inputs same as self.gradient_descent():
            theta_init - initial guess at kappa, beta0, beta1
            gamma - learning rate
            tol - acceptable distance from kappa
            max_its - bound for number of iterations
        """
        # add noise to Phi
        if noise:
            Phi = self.solver(self.theta_true, boundary=True)
            Phi += self.Phi_noise.reshape(self.u_dim + 2, 1)

        # learn the force term faster
        gamma = gamma*np.ones(3)
        gamma[0] = gamma[0] / self.dx  

        # convert theta to numpy array and get L2 error
        theta_new = np.array(theta_init.copy())
        err = self.get_loss(theta_new, self.Phi)

        # initialize descent data dictionary
        thetas = dict() # to store iterations
        iter_count = 0 
        thetas[iter_count] = theta_new.copy()

        while iter_count < max_its and tol < err:
            iter_count += 1
            grad = self.get_gradient(theta_new)
            theta_new -= gamma * grad.flatten()
            err = self.get_loss(theta_new, self.Phi)
            thetas[iter_count] = theta_new.copy()

        if iter_count == max_its:
            warnings.warn('Tolerance not reached. Consider increasing gamma')

        return thetas, iter_count

    ################
    # PLOT METHODS #
    ################
    def plot_param_converging(self, ax, param, true_param):
        iters = [i for i in range(len(param))]
        ax.plot(iters, param, color='g', linewidth=3)
        ax.hlines(true_param, 
                  xmin=0, xmax=max(iters), 
                  linewidth=3, color='k', linestyle='dashed')
        ax.set_xlabel('iterations')

    def plot_u_converging(self, ax, thetas, noise=False):
        # get domain and Phi (with noise)
        x = np.linspace(0., 1., self.u_dim + 2)
        Phi = self.solver(self.theta_true, boundary=True).flatten()
        
        # plot Phi
        ax.plot(x, Phi, label=r'$\Phi$', 
                linewidth=3, 
                color='k', 
                linestyle='dashed')
        ax.set_xlabel(r'$\Omega$')
        
        # add noise to Phi and plot
        if noise:
            Phi += self.Phi_noise
            ax.plot(x, Phi, label=r'$\Phi + $noise',
                    linewidth=3, 
                    color='0.5',
                    linestyle='dashed')

        # plot us
        u_lab = r'$u$'
        for i in range(len(thetas)):
            u = self.solver(thetas[i,:], boundary=True).flatten()
            alpha_i = 1./(i+1)
            ax.plot(x, u, 
                    label=u_lab, 
                    linewidth=3, 
                    color='C0', 
                    alpha=alpha_i)
            u_lab = ''
        ax.legend()

    def plot_grad_descent(self, theta_init, lr, tol, noise=False, max_its=500):
        """
        Performs and plots gradient descent.
        Inputs same as self.gradient_descent():
            theta_init - initial guess at kappa, beta0, beta1
            gamma - learning rate
            tol - acceptable distance from kappa
            max_its - bound for number of iterations
        """
        # make names
        theta_names = [r'$c$', r'$b_0$', r'$b_1$']
        theta_true_names = [r'$\kappa$', r'$\beta_0$', r'$\beta_1$']
        
        # get noise 
        np.random.seed(23)
        self.Phi_noise = np.random.randn(self.u_dim + 2) * 0.2

        # get thetas from gradient descent
        descent, num_iters = self.gradient_descent(theta_init, lr, tol, noise)
        num_its, thetas = zip(*sorted(descent.items()))
        thetas = np.concatenate([t.reshape(1, self.num_params) for t in thetas])

        # 2x2 grid (b0, b1, \\ c, u)
        fig, ax = plt.subplots(2, 2, figsize=(18,15), dpi=200)

        # top row: b0 and b1
        for i in range(2):
            self.plot_param_converging(ax[0,i], 
                                       thetas[:,i+1], 
                                       self.theta_true[i+1])
            ax[0,i].legend([theta_names[i+1], theta_true_names[i+1]])

        # bottom row: c and u
        self.plot_param_converging(ax[1,0], thetas[:,0], self.theta_true[0])
        ax[1,0].legend([theta_names[0], theta_true_names[0]])
        
        # u plot
        self.plot_u_converging(ax[1,1], thetas, noise)

        fig.add_subplot(111, frame_on=False)
        plt.tick_params(labelcolor='none', bottom=False, left=False)
        plt.title(r'%d iterations at learning rate $\gamma = %.1f$' % (max(num_its), lr))
        plt.show()
        plt.close()

    def plot_learning_rates(self, theta_init, lr_bounds, tol, max_its=500):
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
            _, i_count = self.gradient_descent(theta_init, 
                    lr, 
                    tol, 
                    noise=None,
                    max_its=max_its)
            data[lr] = i_count
        lr, counts = zip(*sorted(data.items()))
        lr_opt = lr[np.argmin(counts)]

        # now plot the data
        fig, ax = plt.subplots(1, 1, figsize=(18,10), dpi=200)
        ax.plot(lr, counts, linewidth=4, color='C1')
        ax.set_xlabel('Learning Rate ' + r'$\gamma$')
        ax.set_ylabel('iterations to convergence (%d max)' % max_its)
        ax.set_title(r'%d iterations at $\gamma = %.2f$' 
                     % (np.min(counts), lr_opt))
        plt.show()
        plt.close()

    def biplot_gradient_descent(self, theta_init, gamma, tol, max_its=500):
        """
        Performs gradient descent and makes pairwise biplots
        Inputs same as self.gradient_descent():
            theta_init - initial guess at kappa, beta0, beta1
            gamma - learning rate
            tol - acceptable distance from kappa
            max_its - bound for number of iterations
        """
        # make names
        theta_names = [r'$c$', r'$b_0$', r'$b_1$']
        theta_true_names = [r'$\kappa$', r'$\beta_0$', r'$\beta_1$']
        def ordered_pair(x, y):
            pair = '(' + x + ', ' + y + ')'
            return pair
        
        # get thetas from gradient descent
        descent, _  = self.gradient_descent(theta_init, 
                gamma, 
                tol, 
                noise=None,
                max_its=max_its)
        indices, theta = zip(*sorted(descent.items()))
        # stack thetas into array shape (num_iters, num_params)
        num_iters = len(theta)
        thetas = np.array([t for t in theta])

        fig, ax = plt.subplots(3, 1,
                sharex=False, figsize=(18,18), dpi=200)
        #flow = self.plot_curve_convergence(theta_init, gamma, tol, max_its)
        for i in range(self.num_params):
            j = (i + 1) % 3
            ax[i].plot(thetas[:, i] , thetas[:, j], linewidth=3, color='g')
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

    def plot_curve_convergence(self, theta_init, gamma, tol, max_its=500):
        """
        Performs and plots gradient descent.
        Inputs same as self.gradient_descent():
            theta_init - initial guess at kappa, beta0, beta1
            gamma - learning rate
            tol - acceptable distance from kappa
            max_its - bound for number of iterations
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
        num_params = self.num_params 
        thetas = np.zeros((num_curves, num_params))

        fig, ax = plt.subplots(1, 1, 
                sharex=False, figsize=(20,15), dpi=200)

        # make the plots
        x = np.linspace(0., 1., self.u_dim + 2)
        Phi = self.solver(self.theta_true, boundary=True)
        ax.plot(x, Phi, 
                linewidth=3, 
                linestyle='dashed', 
                c='black')
        # plot the first n curves
        for i in range(num_iters):
            curve = self.solver(theta[i], boundary=True).flatten() 
            alpha_i = 1./(i+1)
            ax.plot(x, curve, 
                    linewidth=3, 
                    alpha=alpha_i, 
                    c='C0')
        plt.legend([r'$\Phi$', r'$u$'], fontsize=30)
        plt.xlabel(r'$\Omega$', fontsize=33)
        plt.ylabel(r'$u$', fontsize=33)
        plt.title(r'%d iterations at learning rate $\gamma = %.1f$' % (num_iters, gamma))
        plt.show()
        plt.close()

