import os
import numpy as np
import pickle
import importlib


class Equation:

    def __init__(self, eqn_name):
        self.name = eqn_name

        # load domain, solution, Theta from config file
        spec = importlib.util.spec_from_file_location(
                self.name,
                'equations/{name}.py'.format(name=self.name))
        eqn = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(eqn)

        # assign config functions to Equation
        self.domain = eqn.domain
        self.solution = eqn.solution
        self.Theta = eqn.Theta

    def simulate(self, replicates):
        """
        Uses Equation() attributes to simulate a dataset
        with size replicates
        """
        domain = self.domain()
        uShape = (replicates, ) + domain.shape
        u = np.zeros(uShape)
        Thetas = self.Theta(replicates)
        for rep, Theta in enumerate(Thetas):
            u[rep, :] = self.solution(Theta)
        theData = (u, Thetas)
        thePickle = open('data/' + self.name + '.pkl', 'wb')
        pickle.dump(theData, thePickle)
        thePickle.close()


class Dataset:

    def __init__(self, eqn_name):
        self.eqn = Equation(eqn_name)

    def load(self, ratios=(0.6, 0.2, 0.2)):
        """
        Loads the (data, targets) tuple, then splits into
        training, validation, and test sets.
        Returns (train, val, test) 3-tuple of (data, targets) tuples
        """
        assert sum(ratios) == 1.0
        thePickle = open('data/' + self.eqn.name + '.pkl', 'rb')
        theData = pickle.load(thePickle)
        thePickle.close()
        
        self.split(theData, ratios)

    def split(self, data, ratios):
        """
        Splits the data into training, validation, and test sets
        whose sizes are specified by the 3-tuple ratios
        """
        replicates = data[0].shape[0]
        trainSize = int(ratios[0]*replicates)
        validateSize = int(ratios[1]*replicates)
        testSize = replicates - trainSize - validateSize

        # split to list [train, val, test]
        inputSplit = np.split(data[0], [trainSize, trainSize + validateSize])
        targetSplit = np.split(data[1], [trainSize, trainSize + validateSize])
        
        # assign train, validate, test attributes
        self.train = (inputSplit[0], targetSplit[0])
        self.validate = (inputSplit[1], targetSplit[1])
        self.test = (inputSplit[2], targetSplit[2])


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

