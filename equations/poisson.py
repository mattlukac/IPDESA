"""
Defines the following Poisson equation problem:
  -Laplacian(u) = c    on Omega=(0,1)
           u(0) = b_0  
           u(1) = b_1
              c = constant
The above formulation gives an exact solution of
           u(x) = -(c/2)x^2 + (c/2 + b_1 - b_0)x + b_0

Here we have theta = (c, b0, b1).
"""

import numpy as np 

# list for plot titles
theta_names = [r'$c$', r'$b_0$', r'$b_1$']

# DOMAIN
def domain():
    """
    The unit interval as an array of length 100
    """
    Omega = np.linspace(0., 1., num=100)
    return Omega

# SOLUTION
def solution(theta):
    """
    Computes the solution u where theta=[c, b0, b1]
    """
    x = domain()
    c, b0, b1 = theta
    u = -c * x**2 / 2 + (c/2 + b1 - b0)*x + b0
    return u

# THETA
def theta(replicates):
    """
    Simulates replicates of theta
    Output has shape (replicates, numParams)
    """
    # create replicates of theta=[c, b0, b1]
    c = np.random.uniform(-100,100, size=replicates)
    b0 = np.random.uniform(-10,10, size=replicates)
    b1 = np.random.uniform(-10,10, size=replicates)
    theta = np.stack((c, b0, b1), axis=1)
    return theta
