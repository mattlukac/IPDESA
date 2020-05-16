"""
Generates training, validation, and test data
for the following Poisson equation problem:
  -Laplace(u) = f    on [0,1]
         u(0) = u_0  
         u(1) = u_1
            f = constant
The above formulation gives an exact solution of
         u(x) = -(f/2)x^2 + (f/2 + u_1 - u_0)x + u_0

Here we have Theta = (u0, u1, f).
Refer to ../save_data.py for how the data are saved
"""

import numpy as np 
import pickle

def pickle_data(domain, Theta, solution, dataSizes):
    # training, validation, test set sizes
    dataTypes = ['train','val','test']
    N = dict(zip(dataTypes, dataSizes))

    # generate and save data
    u = N.copy()
    for dataType, dataSize in N.items():
        # compute solutions
        u[dataType] = np.zeros((dataSize, len(domain)))
        for realization in range(dataSize):
            u[dataType][realization,:] = solution(domain, Theta[dataType][realization,:])

    # pickle the training data as the tuple (data, targets)
    theData = (u, Theta)
    thePickle = open('data.pkl', 'wb')
    pickle.dump(theData, thePickle)
    thePickle.close()


# training, validation, test set sizes
Ntrain, Nval, Ntest = 2000, 500, 1000
dataSizes = [Ntrain, Nval, Ntest]
N = dict(zip(['train', 'val', 'test'], dataSizes))

# define domain
domain = np.linspace(0., 1., num=100)

# define u_Theta(x)
def solution(x, Theta):
    """
    Computes the solution u where Theta=[u0, u1, f]
    """
    u0 = Theta[0]
    u1 = Theta[1]
    f = Theta[2]
    u = -f * x**2 / 2 + (f/2 + u1 - u0)*x + u0
    return u


# make Theta realizations
Theta = N.copy()
for dataLabel, dataSize in N.items():
    # create train/val/test replicates of Theta=[u0,u1,f]
    u0 = np.random.uniform(-10,10, size=dataSize)
    u1 = np.random.uniform(-10,10, size=dataSize)
    f = np.random.uniform(-100,100, size=dataSize)
    Theta[dataLabel] = np.stack((u0, u1, f), axis=1)

# make, pickle, and save the data
pickle_data(domain, Theta, solution, dataSizes)
