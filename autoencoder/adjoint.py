import numpy as np 
from scipy import sparse
from scipy.sparse.linalg import spsolve 
import warnings
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
from matplotlib.colors import ListedColormap
plt.rcParams.update({'font.size': 28,
                     'legend.handlelength': 2})


############################
#     BOUNDARY CONTROL     #
############################ 
class PoissonBC(): 
    """ 
    Computes the gradient of L2 loss with respect to 
    force term c and boundary conditions b0 and b1 
    This is performed by solving the adjoint equation
    obtained by setting the first variation of the Lagrangian
    with respect to u equal to 0

    To maintain mathematical consistency, shapes are 
        x, u, Phi - (U, 1)
        lambda - (1, U)
        Laplacian - (U, U)
        
    """

    def __init__(self, theta_true, resolution=99, sigma=0):
        """
        Initialized with 
            theta_true: [kappa, beta_0, beta_1]
            noise: standard deviation of Gaussian noise on data
            resolution: domain resolution N so that dx = 1/N
        """
        self.dx = 1. / resolution                   # step size
        self.u_dim = resolution + 1                 # solution dimension
        self.x = np.linspace(0., 1., self.u_dim)    # closure of Omega 
        self.Omega_dim = resolution - 1             # Omega dimension
        self.theta_dim = len(theta_true)            # number of parameters
        self.theta_true = np.array(theta_true)      # kappa and beta
        self.Phi = self.get_solution(theta_true)    # observed data
        self.Lap = self.get_Laplacian()             # Laplacian on Omega
        self.sigma = sigma                          # noise stdev
        noise = sigma * np.random.randn(self.u_dim) # Gaussian noise
        self.noise = noise.reshape(-1, 1)           # noise as vector
        self.Phi += self.noise                      # Phi with noise

    def get_Laplacian(self):
        """
        Calculates and returns the discrete Laplacian
        stored as a scipy.sparse matrix
        Has dimension u_dim - 2 since it is applied
        on the interior of the domain
        """
        diags = [1., -2., 1.]     # second central differences
        diags = [x / self.dx ** 2 for x in diags] # over dx^2
        Lap_shape = (self.Omega_dim, self.Omega_dim)

        # Discrete Laplacian
        Delta = sparse.diags(diagonals=diags,
                             offsets=[-1, 0, 1], 
                             shape=Lap_shape,
                             format='csr')
        return Delta

    def get_solution(self, theta):
        """
        Acts as a proxy for the PDE solver
        Computes the solution u given theta=(c, b0, b1)
        The domain is partitioned into U nodes with step size dx, i.e.
        0 = 0*dx, dx, 2*dx, ..., (U-1)*dx = 1
        Returns solution with shape (U, 1)
        """
        x = self.x.reshape(-1, 1)      # make column vector
        soln = np.zeros(x.shape)       # column vector solution
        c, b0, b1 = theta              # unpack theta
        soln += -c / 2 * x ** 2        # quadratic term
        soln += (c / 2 + b1 - b0) * x  # linear term
        soln += b0                     # intercept
        return soln

    def get_interior(self, vec):
        """
        Given column vector vec with shape (U, 1)
        (perhaps vec is domain, solution, or data)
        returns interior of vec, i.e. vec on (dx, .., 1-dx)
        """
        return vec[1:-1]

    def get_loss(self, theta):
        """
        Given current parameter guess theta=(c, b0, b1)
        and (possibly noisy) data Phi, compute L2 loss 
          J(theta) = 1/2 int_0^1 (u_theta - Phi)^2 dx
        """
        u = self.get_solution(theta)
        return 0.5 * np.sum((u - self.Phi) ** 2) * self.dx

    def get_grad(self, theta):
        """
        Given parameter guess theta 
          1. compute analytic PDE solution u,
          2. solve for adjoint variables lambda (interior) and mu (boundary),
          3. compute the constrained loss gradient dJ/dtheta using
              dJ/dtheta = -lambda^* dF/dtheta - mu^* dF/dtheta
        
        where lambda solves the adjoint PDE:
        
            Lap(lambda) = u - Phi   on Omega=(0,1)
            lambda = 0              on boundary {0, 1}
            mu = dlambda/dn         on boundary {0, 1}
        
        Here, n is outward unit normal vector, dlambda/dn = n*dlambda/dx
            and n=-1 for x=0 and n=1 for x=1

        Since dlambda(x) = lambda(x + dx) - lambda(x) and lambda=0 on {0,1}
            dlambda/dn = -lambda(dx) / dx        for x=0
            dlambda/dn = -lambda(1 - dx) / dx    for x=1
        """
        # compute adjoint force term
        u = self.get_solution(theta)   # analytic PDE solution
        dJdu = u - self.Phi            # adjoint PDE force term

        # solve adjoint PDE
        dJdu = self.get_interior(dJdu)
        lamb = spsolve(self.Lap, dJdu)
        mu0 = -lamb[0] / self.dx        # dlambda/dn on left boundary
        mu1 = -lamb[-1] / self.dx       # dlambda/dn on right boundary
        mu = np.array([mu0, mu1]).reshape(2, 1)

        # theta partial of interior constraint F 
        # has shape (Omega_dim, 3) with rows  1  0  0
        dF = np.zeros((self.Omega_dim, self.theta_dim))
        dF[:,0] = 1.
        
        # theta partial of boundary constraint G
        #    0 -1  0
        #    0  0 -1
        dG = np.zeros((2, self.theta_dim))
        dG[0,1] = -1.
        dG[1,2] = -1.

        # desired gradient, with shape (1, 3)
        grad_J = -lamb.T.dot(dF) * self.dx - mu.T.dot(dG)
        return grad_J

    def grad_descent(self, theta_init, lr, tol, max_its=500):
        """
        Performs gradient descent to converge to the true theta
        Inputs:
            theta_init - initial guess at [kappa, beta0, beta1]
            lr - learning rate gamma
            tol - when |grad_old - grad_new| < tol stop descending
            max_its - upper bound for number of iterations
        Returns:
            thetas - dictionary with key:value = num_iter:theta
        """
        # learn the force term faster, helps with convergence
        lr = lr * np.ones(3)
        lr[0] = lr[0] / self.dx  

        # convert theta to numpy array and get L2 error
        theta_new = np.array(theta_init)
        err = self.get_loss(theta_new)

        # initialize descent data dictionary
        thetas = dict() # to store iterations
        iter_count = 0 
        thetas[iter_count] = theta_new.copy()

        # the descent
        delta_grad = 1.0
        grad_old = self.get_grad(theta_new)
        while iter_count < max_its and tol < delta_grad:
            iter_count += 1                       # update iteration count
            theta_new -= lr * grad_old.flatten()  # updated theta
            grad_new = self.get_grad(theta_new)   # updated grad
            delta_grad = np.sum(np.abs(grad_old - grad_new))
            grad_old = grad_new                   # store old grad
            thetas[iter_count] = theta_new.copy() # record new theta

        # warn user about convergence
        if iter_count == max_its:
            message = 'Desired tolerance not reached. '
            message += 'Consider increasing learning rate or tolerance.'
            warnings.warn(message)

        return thetas

    def descend(self, theta_init, lr, tol, max_iters=500):
        """
        Does gradient descent starting at theta_init
        Assigns attributes thetas and solutions
        """
        # get descent data
        self.lr = lr
        data = self.grad_descent(theta_init, lr, tol, max_iters)
        iterations, thetas = zip(*data.items())

        # assign iterations
        self.iterations = iterations
        # assign thetas with shape (num_iters, theta_dim)
        thetas = np.concatenate([t.reshape(1, self.theta_dim) for t in thetas])
        self.thetas = thetas
        # assign solutions with shape (num_iters, u_dim)
        self.solutions = np.zeros((max(iterations) + 1, self.u_dim))
        for i, theta in enumerate(thetas):
            self.solutions[i] = self.get_solution(theta).flatten()

####################
# PLOTTING METHODS #
####################

    def plot_grad_descent(self, **kwargs):
        """
        Performs and plots gradient descent.
        Inputs same as self.gradient_descent():
            theta_init - initial guess at kappa, beta0, beta1
            gamma - learning rate
            tol - acceptable distance from kappa
            max_its - bound for number of iterations
        """
        # get parameters
        c, kappa = (self.thetas[:,0], self.theta_true[0])
        b0, beta0 = (self.thetas[:,1], self.theta_true[1])
        b1, beta1 = (self.thetas[:,2], self.theta_true[2])
        
        # 2x2 grid (b0, b1, \\ c, u)
        fig, ax = plt.subplots(2, 2, figsize=(18,15), dpi=200)

        # plot c, b0, b1
        self.plot_param(ax[1,0], c, 0)
        self.plot_param(ax[0,0], b0, 1)
        self.plot_param(ax[0,1], b1, 2)

        # plot solutions
        self.plot_solns(ax[1,1])

        # make title
        fig.add_subplot(111, frame_on=False)
        plt.tick_params(labelcolor='none', bottom=False, left=False)
        title = r'%d iterations ' % max(self.iterations)
        title += 'at learning rate $\gamma = %.1f$' % self.lr
        plt.title(title)

        plt.show()
        plt.close()

    def plot_curve_convergence(self):
        """
        Performs and plots gradient descent.
        Inputs same as self.gradient_descent():
            theta_init - initial guess at kappa, beta0, beta1
            gamma - learning rate
            tol - acceptable distance from kappa
            max_its - bound for number of iterations
        """
        fig, ax = plt.subplots(1, 1, figsize=(20,15), dpi=200)
        
        title = r'%d iterations ' % max(self.iterations)
        title += 'at learning rate $\gamma = %.1f$' % self.lr
        self.plot_solns(ax, title)
        plt.show()
        plt.close()

    def biplot_grad_descent(self):
        """
        Performs gradient descent and makes pairwise biplots
        Inputs same as self.gradient_descent():
            theta_init - initial guess at kappa, beta0, beta1
            gamma - learning rate
            tol - acceptable distance from kappa
            max_its - bound for number of iterations
        """
        fig, ax = plt.subplots(self.theta_dim, 1, figsize=(18,18), dpi=200)

        for i in range(self.theta_dim):
            j = (i + 1) % 3 
            params = [self.thetas[:,i], self.thetas[:,j]]
            idxs = [i, j]
            self.biplot_params(ax[i], params, idxs)
        
        fig.add_subplot(111, frame_on=False)
        plt.tick_params(labelcolor='none', bottom=False, left=False)
        title = '%d iterations to convergence, ' % max(self.iterations)
        title += r'$\gamma = %.1f$' % self.lr
        plt.title(title)
        plt.show()
        plt.close()

    def plot_learning_rates(self, theta_init, lr_bounds, tol, max_its=500):
        """
        Plots number of iterations to convergence
        as a function of learning rate given by lr_bounds
        """
        # make learning rates
        lr_min, lr_max = lr_bounds
        num_descents = 100    # number of times to perform grad descent
        learning_rates = np.linspace(lr_min, lr_max, num_descents)

        # collect data
        data = dict()
        for lr in learning_rates:
            thetas = self.grad_descent(theta_init, lr, tol, max_its)
            data[lr] = max(thetas.keys())  # keys are iterations
        lr, counts = zip(*data.items())
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

##################
# HELPER METHODS #
##################

    def plot_param(self, ax, param, idx):
        """
        Plots parameter convergence during gradient descent
        Inputs:
            ax - axis object for plotting
            param - parameter to plot vs iterations
            idx - index for true parameter (0:kappa, 1:beta0, 2:beta1)
        """
        # make names and get true parameter
        theta_names = [r'$c$', r'$b_0$', r'$b_1$']
        theta_true_names = [r'$\kappa$', r'$\beta_0$', r'$\beta_1$']
        true_param = self.theta_true[idx]

        ax.plot(self.iterations, param, color='g', linewidth=3)
        ax.hlines(true_param, 
                  xmin=0, xmax=max(self.iterations),
                  linewidth=3, 
                  color='k', 
                  linestyle='dashed')
        ax.set_xlabel('iterations')
        ax.legend([theta_names[idx], theta_true_names[idx]])

    def plot_solns(self, ax, title=''):

        # plot all the us
        u_lab = r'$u$'
        for i, u in enumerate(self.solutions):
            alpha_i = 1./(i+1)
            ax.plot(self.x, u, 
                    label=u_lab, 
                    linewidth=3, 
                    color='C0', 
                    alpha=alpha_i)
            u_lab = ''

        # plot Phi without noise
        Phi = self.Phi - self.noise
        ax.plot(self.x, Phi, 
                label=r'$\Phi$',
                linewidth=3, 
                color='k',
                linestyle='dashed')

        # plot Phi with noise if exists
        if self.sigma != 0:
            ax.plot(self.x, self.Phi, 
                    label=r'$\Phi + $noise',
                    linewidth=3, 
                    color='0.5', 
                    linestyle='dashed')
        
        ax.set_xlabel(r'$\Omega$')
        ax.set_title(title)
        ax.legend()

    def biplot_params(self, ax, params, idxs):

        # make names and get true parameter
        theta_names = [r'$c$', r'$b_0$', r'$b_1$']
        theta_true_names = [r'$\kappa$', r'$\beta_0$', r'$\beta_1$']
        param_x, param_y = params
        x, y = idxs

        def ordered_pair(x, y):
            pair = '(' + x + ', ' + y + ')'
            return pair
        curve = ordered_pair(theta_names[x], theta_names[y])
        optim = ordered_pair(theta_true_names[x], theta_true_names[y])

        ax.plot(param_x, param_y, linewidth=3, color='g')
        ax.scatter(self.theta_true[x], self.theta_true[y], s=15 ** 2, c='r')
        ax.legend([curve, optim])

###################
# TESTING METHODS #
###################

    def test_get_grad(self, theta):
        """
        Compares the analytic gradient to the numerical gradient
        The analytic gradient can be expressed as the covector-matrix product:

                                              1/120 5/120 5/120
        (c - kappa, b0 - beta0, b1 - beta1) * 1/24  8/24  4/24
                                              1/24  4/24  8/24
        """
        # first compute the analytic gradient
        theta_diff = np.array(theta) - self.theta_true
        theta_diff = theta_diff.reshape(1, -1)  # make covector
        coefs = np.array([[1./120, 5./120, 5./120],
                          [1./24, 8./24, 4./24],
                          [1./24, 4./24, 8./24]])
        analytic_grad_J = theta_diff.dot(coefs)

        # now the numerical gradient
        grad_J = self.get_grad(theta)

        # print results
        print('analytic gradient:', analytic_grad_J)
        print('numerical gradient:', grad_J)

    def test_grad_descent(self, theta_init, lr, tol):
        """
        Performs gradient descent and prints:
          initial theta
          number of iterations
          terminal theta
          true theta
        """
        print('initial theta:', theta_init)
        print('descending...')

        thetas = self.grad_descent(theta_init, lr, tol)
        num_iters = max(thetas.keys())
        theta_terminal = list(thetas.values())[-1]

        print('after %d iterations,' % num_iters)
        print('terminal theta:', theta_terminal)
        print('true theta:', self.theta_true)

########################
# CONTOUR PLOT TENSORS #
########################

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



#######################
# NO BOUNDARY CONTROL #
#######################
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

