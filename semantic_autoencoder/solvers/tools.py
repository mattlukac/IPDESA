import numpy as np
from fenics import *
from fenics_adjoint import *


def normalise(func):
    func.vector()[:] /= assemble(func * dx)

def nonneg(func):
    func_array = func.vector()[:]
    func_array[func_array < 0.] = 0.
    func.vector().set_local(func_array)

def density(func):
    nonneg(func)
    normalise(func)

def cosinify(x):
    y = 2. * (x - 0.5)
    y = 0.5 * (np.cos(np.pi * (y - 1.) / 2.) + 1.)
    return y

def sigmoidish(x, k):
    def f(xx):
        return k * xx / (k - xx + 1.)
    def g(xx):
        y = f(xx)
        y[xx < 0.] = -f(-xx[xx < 0.])
        return y
    def h(xx):
        return 0.5 * g(2. * xx - 1.) + 0.5
    return h(x)

