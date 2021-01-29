""" Functions that convert between torch, numpy, and fenics """

import numpy as np
import torch
from fenics import *

def numpy_to_torch(array):
    return torch.tensor(array, dtype=torch.float)

def torch_to_numpy(tensor):
    return tensor.detach().numpy()

def numpy_to_fenics(array, V, v2d):
    function = Function(V)
    return function.vector().set_local(array[v2d])

def fenics_to_numpy(function, mesh):
    return function.compute_vertex_values(mesh)

def torch_to_fenics(tensor, V, v2d):
    array = torch_to_numpy(tensor)
    return numpy_to_fenics(array, V, v2d)

def fenics_to_torch(function, mesh):
    array = fenics_to_numpy(function, mesh)
    return numpy_to_torch(array)
