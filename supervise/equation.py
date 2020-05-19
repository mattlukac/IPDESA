import numpy as np
import pickle
import importlib
import os


class Equation:

    def __init__(self, eqn_name):
        self.name = eqn_name

        # load domain, solution, Theta from config file
        spec = importlib.util.spec_from_file_location(
                self.name,
                'equations/%s.py' % self.name)
        eqn = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(eqn)

        # assign config functions to Equation
        self.domain = eqn.domain
        self.solution = eqn.solution
        self.Theta = eqn.Theta

    def simulate(self, replicates):
        """
        Uses Equation attributes to simulate a dataset
        with size replicates and save data to a pickle
        """
        # simulate the data
        domain = self.domain()
        uShape = (replicates, ) + domain.shape
        u = np.zeros(uShape)
        Thetas = self.Theta(replicates)
        for rep, Theta in enumerate(Thetas):
            u[rep] = self.solution(Theta)

        # Thetas must have at least one column
        if Thetas.ndim == 1:
            Thetas = Thetas.reshape(-1, 1)

        # pickle the data
        theData = (u, Thetas)
        if not os.path.exists('data/'):
            os.mkdir('data')
        thePickle = open('data/%s.pkl' % self.name, 'wb')
        pickle.dump(theData, thePickle)
        thePickle.close()


class Dataset:

    def __init__(self, eqn_name):
        self.Eqn = Equation(eqn_name)

    def load(self, ratios=(0.6, 0.2, 0.2)):
        """
        Loads the (data, targets) tuple, then splits into
        training, validation, and test sets represented as
        Dataset attributes
        """
        assert sum(ratios) == 1.0

        # check pickled data exists; if not, simulate it
        if not os.path.exists('data/%s.pkl' % self.Eqn.name):
            replicates = 2000
            self.Eqn.simulate(replicates)
            print('Training data did not exist.')
            print('Simulated with %d replicates' % replicates)
            print('To change the number of replicates,',
                  'use Equation.simulate(replicates)')

        # load the pickled data
        thePickle = open('data/%s.pkl' % self.Eqn.name, 'rb')
        theData = pickle.load(thePickle)
        thePickle.close()
        
        # save normalizing constants then split
        self.target_min = np.min(theData[1], axis=0)
        self.target_range = np.ptp(theData[1], axis=0)
        self.split(theData, ratios)

    def split(self, data, ratios):
        """
        Splits the data into training, validation, and test sets
        whose sizes are specified by the 3-tuple ratios
        """
        # get train, val, test set sizes
        replicates = data[0].shape[0]
        trainSize = int(ratios[0]*replicates)
        validateSize = int(ratios[1]*replicates)

        # split to list [train, val, test]
        inputSplit = np.split(data[0], [trainSize, trainSize + validateSize])
        targetSplit = np.split(data[1], [trainSize, trainSize + validateSize])
        
        # assign train, val, test attributes
        splits = [(inputSplit[i], targetSplit[i]) for i in range(len(ratios))]
        self.train = splits[0]
        self.validate =  splits[1]
        self.test =  splits[2]

