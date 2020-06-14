from adjoint import *

theta_true = [4., 1., -1.]
theta_init = [2., -1., 1.]
noise = 0.0
resolution = 99 
lr = 1.0
tol = 1e-6

pois = PoissonBC(theta_true, resolution, noise)
pois.test_grad_descent(theta_init, lr, tol)
