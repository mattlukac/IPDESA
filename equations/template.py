"""
Here is a template to configure a family of functions
parameterized by Theta.
Leave Theta_names empty if you do not wish to name the parameters.
"""

import numpy as np 

# NAME THE PARAMETERS
Theta_names = []

# DOMAIN
def domain():
    """
    Create a lattice of the domain for the problem
    """

# THETA
def Theta(replicates):
    """
    Simulates replicates of Theta
    Output should have shape (replicates, num_params)
    """

# SOLUTION
def solution(Theta):
    """
    Computes the solution u given a single replicate of Theta
    """
    x = domain()
