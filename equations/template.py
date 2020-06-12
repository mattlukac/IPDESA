"""
Here is a template to configure a family of functions
parameterized by theta.
Leave theta_names empty if you do not wish to name the parameters.
"""

import numpy as np 

# NAME THE PARAMETERS
theta_names = []

# DOMAIN
def domain():
    """
    Create a lattice of the domain for the problem
    """

# THETA
def theta(replicates):
    """
    Simulates replicates of theta
    Output should have shape (replicates, num_params)
    """

# SOLUTION
def solution(theta):
    """
    Computes the solution u given a single replicate of theta
    """
    x = domain()
