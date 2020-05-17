"""
Function that pickles training data for ../supervised_encoder.py
which learns the parameters Theta that determine u_Theta: Omega -> R^n
Inputs:
    domain - the domain Omega of u_Theta, as a numpy array
    Theta - dictionary with keys 'train', 'val', 'test'
            and values are numpy arrays containing parameter values
            with shape (dataSize, numParams)
    solution - function that takes as input (domain, Theta)
               where Theta is a single realization of parameter values
    dataSizes - list containing the sizes of the training, validation, test sets

The output will be the pickle data.pkl
which contains the tuple (data=u, targets=Theta) of dictionaries
with keys 'train', 'val', and 'test'. 
The values are numpy arrays for the data and targets.
The shapes of the data and target arrays are
(dataSize, len(x)) and (dataSize, numParams), respectively,
where numParams=3 for u0 u1 and f.
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
            u[dataType][realization,:] = solution(x, Theta[dataType][realization,:])

    # standardize data
    

    # pickle the training data as the tuple (data, targets)
    theData = (u, Theta)
    thePickle = open('data.pkl', 'wb')
    pickle.dump(theData, thePickle)
    thePickle.close()
