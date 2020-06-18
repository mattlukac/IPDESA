import numpy as np 
from scipy import sparse
from scipy.sparse.linalg import spsolve 
import warnings
from . import plotter


############################
#     BOUNDARY CONTROL     #
############################ 
class PoissonBC: 
    """ 
    Computes the gradient of L2 loss with respect to 
    force term c and boundary conditions b0 and b1 
    This is performed by solving the adjoint equation
    obtained by setting the first variation of the Lagrangian
    with respect to u equal to 0

    Let N be discretization resolution so dx = 1/N
    To maintain mathematical consistency, shapes are 
        x, u, Phi:  (N+1, 1)    on closure of Omega
        lambda:     (1, N+1)    lambda is dual to u
        Laplacian:  (N-1, N-1)  on interior of Omega
        theta:      (3,)
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

        # theta partial of interior constraint F 
        # has shape (Omega_dim, 3) with rows  1  0  0
        self.dF = np.zeros((self.Omega_dim, self.theta_dim))
        self.dF[:,0] = 1.
        # theta partial of boundary constraint G
        #    0 -1  0
        #    0  0 -1
        self.dG = np.zeros((2, self.theta_dim))
        self.dG[0,1] = self.dG[1,2] = -1.

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
        mu = np.array([mu0, mu1]).reshape(-1, 1)

        # desired gradient, with shape (1, 3)
        grad_J = -lamb.T.dot(self.dF) * self.dx - mu.T.dot(self.dG)
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
        # initialize descent data dictionary
        theta = np.array(theta_init)
        thetas = dict() # to store iterations
        iter_count = 0 
        thetas[iter_count] = theta.copy()

        # are we learning (kappa, beta), (kappa, beta0), or just kappa?
        grad = self.get_grad(theta).flatten()
        grad_dim = len(grad)
        lr = lr * np.ones(grad_dim)
        if grad_dim == 3:
            lr[0] = lr[0] / self.dx   # lr = (lr/dx, lr) or (lr/dx, lr, lr)
        elif grad_dim == 2:
            lr[0] = lr[0] / np.sqrt(self.dx)

        # the descent
        grad_step = np.zeros(3)
        delta_grad = 1.0
        while iter_count < max_its and tol < delta_grad:
            # update iteration counter
            iter_count += 1
            # step with old gradient
            grad_step[:grad_dim] = lr * grad
            theta_new = theta - grad_step
            theta_new = theta_new.flatten()
            # solve new adjoint equation
            grad_new = self.get_grad(theta_new)
            # L1 distance between consecutive gradients
            delta_grad = np.sum(np.abs(grad - grad_new))
            # save new theta, update grad and theta for next iteration
            thetas[iter_count] = theta_new.copy() 
            grad = grad_new 
            theta = theta_new 

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
        print('descending...')
        data = self.grad_descent(theta_init, lr, tol, max_iters)
        iterations, thetas = zip(*data.items())
        print('completed with %d iterations' % max(iterations))

        # assign iterations
        self.iterations = iterations
        # assign thetas with shape (num_iters, theta_dim)
        thetas = np.concatenate([t.reshape(1, 3) for t in thetas])
        self.thetas = thetas
        # assign solutions with shape (num_iters, u_dim)
        self.solutions = np.zeros((max(iterations) + 1, self.u_dim))
        for i, theta in enumerate(thetas):
            self.solutions[i] = self.get_solution(theta).flatten()

        # pass grad descent data to plotter
        self.plt = plotter.AdjointDescent(self)

    ####################
    # PLOTTING METHODS #
    ####################

    def plot_grad_descent(self):
        """
        Performs and plots gradient descent.
        """
        self.plt.grad_descent_bc()

    def plot_curve_convergence(self):
        """
        Performs and plots gradient descent.
        Inputs same as self.gradient_descent():
            theta_init - initial guess at kappa, beta0, beta1
            gamma - learning rate
            tol - acceptable distance from kappa
            max_its - bound for number of iterations
        """
        self.plt.curve_convergence()

    def biplot_grad_descent(self):
        """
        Performs gradient descent and makes pairwise biplots
        Inputs same as self.gradient_descent():
            theta_init - initial guess at kappa, beta0, beta1
            gamma - learning rate
            tol - acceptable distance from kappa
            max_its - bound for number of iterations
        """
        self.plt.biplot_descent_bc()

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

        self.plt.learning_rates(data, max_its)

    ###################
    # TESTING METHODS #
    ###################

    def test_get_grad(self, theta):
        """
        Compares the numerical gradient to the analytic gradient 
        which can be expressed as the covector-matrix product:

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


###########################
#  LEFT BOUNDARY CONTROL  #
###########################

class PoissonLBC(PoissonBC):
    """
    Class for computing the L2 loss gradient
    wrt PDE model parameters for the Poisson equation,
    via the adjoint method.
    """
    def __init__(self, theta, bv, resolution=99, sigma=0):
        kappa, beta0 = theta  # true (optimal) theta
        self.b1 = bv  # given right boundary value
        theta_true = [kappa, beta0, self.b1]
        super().__init__(theta_true, resolution, sigma)
        self.theta_dim = len(theta)
        self.dF = np.zeros((self.Omega_dim, len(theta))) # force control
        self.dF[:,0] = 1.
        self.dG = np.zeros((2, len(theta)))      # left boundary control
        self.dG[0,1] = -1.

    def descend(self, theta_init, lr, tol):
        """
        Starting at c=c_init perform gradient descent until
        the change in gradient |grad_old - grad_new| < tol
        where c_new = c_old - lr * grad_old

        Data from the descent are saved as attributes
          iterations
          thetas
          solutions
        """
        self.lr = lr
        print('descending...')
        theta_init.append(self.b1)
        data = self.grad_descent(theta_init, lr, tol)
        print('completed with %d iterations' % len(data))

        # save iterations and thetas attributes
        iterations, thetas = zip(*data.items())
        self.iterations = iterations 
        thetas = np.concatenate([theta.reshape(1, 3) for theta in thetas])
        self.thetas = thetas 

        # save solutions attribute
        self.solutions = np.zeros((max(iterations) + 1, self.u_dim))
        for i, theta in enumerate(thetas):
            self.solutions[i] = self.get_solution(theta).flatten()

        # pass descent data to plotter
        self.plt = plotter.AdjointDescent(self)

    ################
    # PLOT METHODS #
    ################

    def plot_grad_descent(self):
        """
        Performs and plots gradient descent.
        Inputs same as self.gradient_descent():
            c_init - initial guess at kappa
            gamma - learning rate
            tol - acceptable distance from kappa
            max_its - bound for number of iterations
        """
        # get parameters
        c, kappa = (self.thetas[:,0], self.theta_true[0])
        b0, beta0 = (self.thetas[:,1], self.theta_true[1])

        self.plt.grad_descent_lbc(c, kappa, b0, beta0)

    def biplot_grad_descent(self):
        """ Makes pairwise biplots of theta convergence """
        self.plt.biplot_descent_lbc()

    def plot_learning_rates(self, theta_init, lr_bounds, tol, max_its=500):
        """
        Plots number of iterations to convergence
        as a function of learning rate given by lr_bounds
        """
        theta_init.append(self.b1)
        super().plot_learning_rates(theta_init, lr_bounds, tol, max_its)

    def save_contour_frame(self, frame_num):
        """
        Plots contours of loss surface where each frame
        is a value of the force parameter c
        """
        self.contour(frame_num + 1, self.ranges)
        frame_path = 'visuals/contour_lbc/frames/'
        self.plt.save_frame(frame_num, frame_path)

    def contour_mp4(self):
        """
        Save mp4 of parameters converging to optimum on contour plot
        """
        self.get_contour_slice()
        save_frame = self.save_contour_frame

        # make settings dictionary
        settings = dict()
        settings['mp4_path'] = 'visuals/contour_lbc/descending.mp4'
        settings['frame_path'] = 'visuals/contour_lbc/frames/'
        settings['num_frames'] = len(self.contour_c)
        settings['duration'] = 5 # seconds 

        self.plt.save_mp4(save_frame, settings)

    ################
    # LOSS CONTOUR #
    ################

    def get_tensors(self, ranges, contour_res):
        """
        Given ranges for c and b0 and some resolution
        computes the loss tensor as a function of c and b0
        """
        # get ranges and make lattice
        c_min, c_max, b0_min, b0_max = ranges
        c = np.linspace(c_min, c_max, contour_res)
        b0 = np.linspace(b0_min, b0_max, contour_res)
        c_tensor, b0_tensor = np.meshgrid(c, b0)

        # compute loss for each lattice point
        losses = np.zeros((contour_res, contour_res))
        for idx, _ in np.ndenumerate(c_tensor):
            c = c_tensor[idx]
            b0 = b0_tensor[idx]
            losses[idx] = self.get_loss([c, b0, self.b1])

        # save as attributes
        self.c_tensor = c_tensor
        self.b0_tensor = b0_tensor
        self.losses = losses

    def get_contour_slice(self):
        c_min, c_max, b0_min, b0_max = self.ranges
        c, b0 = (self.thetas[:,0], self.thetas[:,1])
        for i in range(len(c)):
            checker = (c[i] > c_min) & (b0[i] > b0_min)
            iter_start = i
            if checker:
                break
        c = c[(iter_start-1):]
        b0 = b0[(iter_start-1):]
        self.contour_c = c
        self.contour_b0 = b0

    def contour(self, max_iter, ranges):
        """
        Calculates loss tensor, if necessary,
        then plots the -log(loss) contour as a function of c and b0
        with the gradient descent path overlayed on top
        """
        self.ranges = ranges
        c_min, c_max, b0_min, b0_max = self.ranges
        contour_res = 80 
        # compute losses
        self.get_tensors(ranges, contour_res)

        # assign contour_c/b0 attributes, which have the first few 
        # iterations removed so video starts with params in frame
        self.get_contour_slice()
        c = self.contour_c[:max_iter]
        b0 = self.contour_b0[:max_iter]

        # get -log(loss) and level sets
        log_losses = -np.log(self.losses)
        lines = np.linspace(np.min(log_losses), np.max(log_losses), 20)

        self.plt.loss_contour(c, b0, log_losses)

    ###################
    # TESTING METHODS #
    ###################

    def test_get_grad(self, theta):
        """
        Compares the analytic gradient to the numerical gradient
            analytic grad = (c - kappa)/120
        """
        # first compute the analytic gradient
        theta = np.array(theta)
        theta_diffs = theta - self.theta_true[:2]
        theta_diffs = theta_diffs.reshape(1, -1)
        coefs = np.array([[1./120, 5./120], [1./24, 8./24]])
        analytic_grad_J = theta_diffs.dot(coefs)

        # now the numerical gradient
        c, b0 = theta
        grad_J = self.get_grad([c, b0, self.b1])

        # print results
        print('analytic gradient:', analytic_grad_J)
        print('numerical gradient:', grad_J)


#######################
# NO BOUNDARY CONTROL #
#######################

class Poisson(PoissonBC):
    """
    Class for computing the L2 loss gradient
    wrt PDE model parameters for the Poisson equation,
    via the adjoint method.
    """
    def __init__(self, kappa, bc, resolution=99, sigma=0):
        theta = [kappa, bc[0], bc[1]]
        self.kappa = kappa 
        self.b0 = bc[0]
        self.b1 = bc[1]
        super().__init__(theta, resolution, sigma)
        self.dF = np.ones((self.Omega_dim, 1)) # force term control
        self.dG = np.zeros((2,))               # no boundary control
        self.theta_dim = 1

    def descend(self, c_init, lr, tol):
        """
        Starting at c=c_init perform gradient descent until
        the change in gradient |grad_old - grad_new| < tol
        where c_new = c_old - lr * grad_old

        Data from the descent are saved as attributes
          iterations
          thetas
          solutions
        """
        self.lr = lr
        theta_init = [c_init, self.b0, self.b1]
        data = self.grad_descent(theta_init, lr, tol)

        # save iterations and thetas attributes
        iterations, thetas = zip(*data.items())
        self.iterations = iterations 
        thetas = np.concatenate([theta.reshape(1,3) for theta in thetas])
        self.thetas = thetas 

        # save solutions attribute
        self.solutions = np.zeros((max(iterations) + 1, self.u_dim))
        for i, theta in enumerate(thetas):
            self.solutions[i] = self.get_solution(theta).flatten()

        # pass descent data to plotter
        self.plt = plotter.AdjointDescent(self)

    ################
    # PLOT METHODS #
    ################

    def save_loss_frame(self, c_init, frame_num):
        """
        Plots contours of loss surface where each frame
        is a value of the force parameter c
        """
        self.plot_loss(c_init)
        frame_path = 'visuals/loss/frames/'
        self.plt.save_frame(frame_num, frame_path)

    def loss_mp4(self, c_init):
        """ Saves video of gradient descent starting at c_init """
        cs = self.thetas[:,0]
        save_frame = lambda frame : self.save_loss_frame(cs[frame], frame)

        # make settings dictionary
        settings = dict()
        settings['mp4_path'] = 'visuals/loss/descending.mp4'
        settings['frame_path'] = 'visuals/loss/frames/'
        settings['num_frames'] = len(self.thetas[:,0])
        settings['duration'] = 5 # seconds 
            
        self.plt.save_mp4(save_frame, settings)

    def plot_loss(self, c_init):
        """
        Plots loss J as a function of force parameter c
        """
        # establish domain bounds from c relative to kapp
        c_minus_kappa = c_init - self.kappa
        if c_minus_kappa > 0:
            lb = self.kappa - 1
            ub = self.kappa + c_minus_kappa + 1
        elif c_minus_kappa < 0:
            lb = self.kappa + c_minus_kappa - 1
            ub = self.kappa + 1
        c_range = np.linspace(lb, ub, 100)

        # get the point (c, J(c))
        theta_init = [c_init, self.b0, self.b1]
        Jc = self.get_loss(theta_init)

        # get slope dJ/dc and tangent line 
        dJdc = self.get_grad(theta_init).item()
        tangent = lambda dom, x, y, m : m * (dom - x) + y
        grad_line = tangent(c_range, c_init, Jc, dJdc)

        self.plt.loss(c_init, c_range, grad_line, Jc, dJdc)

    def plot_grad_descent(self):
        """ Plots gradient descent results """
        # get parameters
        c, kappa = (self.thetas[:,0], self.theta_true[0])

        self.plt.grad_descent(c, kappa)

    def plot_learning_rates(self, c_init, lr_bounds, tol, max_its=500):
        """
        Plots number of iterations to convergence
        as a function of learning rate given by lr_bounds
        """
        theta_init = [c_init, self.b0, self.b1]
        super().plot_learning_rates(theta_init, lr_bounds, tol, max_its)

    ###################
    # TESTING METHODS #
    ###################

    def test_get_grad(self, c):
        """
        Compares the analytic gradient to the numerical gradient
            analytic grad = (c - kappa)/120
        """
        # first compute the analytic gradient
        analytic_grad_J = (c - self.kappa) / 120.

        # now the numerical gradient
        grad_J = self.get_grad([c, self.b0, self.b1])

        # print results
        print('analytic gradient:', analytic_grad_J)
        print('numerical gradient:', grad_J.item())

