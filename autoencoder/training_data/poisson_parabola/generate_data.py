"""
Generates training, validation, and test data
for the following Poisson equation problem:
  -Laplace(u) = f    on [0,1]
         u(0) = u_0  
         u(1) = u_1
            f = constant
The above formulation gives an exact solution of
         u(x) = -(f/2)x^2 + (f/2 + u_1 - u_0)x + u_0
"""

import numpy as np 

# training, validation, test set sizes
datatype = ['train','val','test']
N = dict(zip(datatype, [2000,500,1000]))

# sampled points in the domain
x = np.linspace(0., 1., num=100)

# numpy arrays for data
def u(x, u0, u1, f):
    u = -f * x**2 / 2 + (f/2 + u1 - u0)*x + u0
    return u

# generate and save data
u0 = N.copy()
u1 = N.copy()
f = N.copy()
solns = N.copy()
for datatype, setSize in N.items():
    # parameter priors
    u0[datatype] = np.random.uniform(-100,100, size=setSize)
    u1[datatype] = np.random.uniform(-10,10, size=setSize)
    f[datatype] = np.random.uniform(-10,10, size=setSize)
    # compute parabolas
    solns[datatype] = np.zeros((setSize, len(x)))
    for i in range(setSize):
        solns[datatype][i,:] = u(x, u0[datatype][i], u1[datatype][i], f[datatype][i])
    # save data 
    np.save('u_'+datatype, solns[datatype])
    np.save('u0_'+datatype, u0[datatype])
    np.save('u1_'+datatype, u1[datatype])
    np.save('f_'+datatype, f[datatype])
