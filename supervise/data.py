import os
import numpy as np
import pickle
import importlib


class Equation():

    def __init__(self, name):
        self.name = name

        spec = importlib.util.spec_from_file_location(
                self.name,
                '../data/{name}.py'.format(name=self.name))
        eqn = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(eqn)

        self.domain = eqn.domain
        self.solution = eqn.solution
        self.Theta = eqn.Theta


    class Dataset():

        def __init__(self):
            self.domain = Equation.domain()
            self.solution = Equation.solution
            self.Theta = Equation.Theta

        def make(self, replicates):
            """
            Uses Equation() attributes to simulate a dataset
            with size replicates
            """
            Thetas = self.Theta(replicates)
            for rep in range(replicates):
                u[rep] = solution(Thetas[rep, :])

        def split(self, ratios=(0.6, 0.2, 0.2)):
            """
            Splits the data into training, validation, and test sets
            specified by the 3-tuple ratios
            """

        def load(self, directory, split=(0.6, 0.2, 0.2)):
            """
            Loads the (data, targets), both are dictionaries
            with keys 'train', 'val', 'test'.
            The input dirName is simply the name of the
            directory containing the pickled training data.
            """
            thePickle = open(dirName + '/data.pkl', 'rb')
            theData = pickle.load(thePickle)
            thePickle.close()
            return theData




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

