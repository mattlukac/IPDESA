"""
Defines the following Poisson equation problem:
  -Laplace(u) = c    on [0,1]
         u(0) = u_0  
         u(1) = u_1
            c = constant
The above formulation gives an exact solution of
         u(x) = -(c/2)x^2 + (c/2 + u_1 - u_0)x + u_0

Here we have Theta = (u0, u1, c).
"""

import numpy as np 

# list for plot titles
Theta_names = [r'$u_0$', r'$u_1$', r'$c$']

# DOMAIN
def domain():
    """
    The unit interval as an array of length 100
    """
    Omega = np.linspace(0., 1., num=100)
    return Omega

# SOLUTION
def solution(Theta):
    """
    Computes the solution u where Theta=[u0, u1, c]
    """
    x = domain()
    u0 = Theta[0]
    u1 = Theta[1]
    c = Theta[2]
    u = -c * x**2 / 2 + (c/2 + u1 - u0)*x + u0
    return u

# THETA
def Theta(replicates):
    """
    Simulates replicates of Theta
    Output has shape (replicates, numParams)
    """
    # create replicates of Theta=[u0,u1,c]
    u0 = np.random.uniform(-10,10, size=replicates)
    u1 = np.random.uniform(-10,10, size=replicates)
    c = np.random.uniform(-100,100, size=replicates)
    Theta = np.stack((u0, u1, c), axis=1)
    return Theta
