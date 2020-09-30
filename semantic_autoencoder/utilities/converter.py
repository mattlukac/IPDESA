""" Functions that convert between torch, numpy, and fenics """

import numpy as np
import torch
from fenics import *


def torch_to_numpy(tensor):
    return tensor.detach().numpy()

def numpy_to_fenics(array, V, v2d):
    function = Function(V)
    return function.vector().set_local(array[v2d])

def fenics_to_numpy(function, mesh):
    return function.compute_vertex_values(mesh)

def numpy_to_torch(array):
    return torch.tensor(array, dtype=torch.float)
