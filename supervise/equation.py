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
        self.Theta_names = eqn.Theta_names

    def simulate(self, replicates):
        """
        Uses Equation attributes to simulate a dataset
        with size replicates and save data to a pickle
        """
        # simulate the data
        domain = self.domain()
        u_shape = (replicates, ) + domain.shape
        u = np.zeros(u_shape)
        Thetas = self.Theta(replicates)
        for rep, Theta in enumerate(Thetas):
            u[rep] = self.solution(Theta)

        # Thetas must have at least one column
        if Thetas.ndim == 1:
            Thetas = Thetas.reshape(-1, 1)

        # pickle the data
        the_data = (u, Thetas)
        if not os.path.exists('data/'):
            os.mkdir('data')
        the_pickle = open('data/%s.pkl' % self.name, 'wb')
        pickle.dump(the_data, the_pickle)
        the_pickle.close()


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
        the_pickle = open('data/%s.pkl' % self.Eqn.name, 'rb')
        the_data = pickle.load(the_pickle)
        the_pickle.close()
        
        # save normalizing constants then split
        self.target_min = np.min(the_data[1], axis=0)
        self.target_range = np.ptp(the_data[1], axis=0)
        self.split(the_data, ratios)

    def split(self, data, ratios):
        """
        Splits the data into training, validation, and test sets
        whose sizes are specified by the 3-tuple ratios
        """
        # get train, val, test set sizes
        replicates = data[0].shape[0]
        train_size = int(ratios[0]*replicates)
        val_size = int(ratios[1]*replicates)

        # split to list [train, val, test]
        input_split = np.split(data[0], [train_size, train_size + val_size])
        target_split = np.split(data[1], [train_size, train_size + val_size])
        
        # assign train, val, test attributes
        splits = [(input_split[i], target_split[i]) for i in range(len(ratios))]
        self.train = splits[0]
        self.validate =  splits[1]
        self.test =  splits[2]

