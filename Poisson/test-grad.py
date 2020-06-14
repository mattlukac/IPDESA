from adjoint import *

# set parameters
theta_true = [4., 1., -1.]
theta = [2., -1., 1.]
noise = 0.0
resolution = 99 
analytic_grad_J = [-1./60, -5./12, 1./4]

# compute gradients
pois = PoissonBC(theta_true, resolution, noise)
pois.test_get_grad(theta)
