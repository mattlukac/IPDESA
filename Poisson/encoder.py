import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
from tensorflow.keras.layers import *
from tensorflow.keras import Model
from tensorflow.keras.callbacks import TensorBoard, LearningRateScheduler
from sklearn import preprocessing 
from copy import deepcopy
import numpy as np
import datetime
from . import equation, plotter
from .bootstrapper import bootstrap


class Encoder:

    def __init__(self, design):

        # attributes defined in design dictionary
        self.num_layers = len(design['unit_activations'])
        self.optim = design['optimizer']
        self.loss = design['loss']
        self.callbacks = design['callbacks']
        self.batch_size = design['batch_size']
        self.epochs = design['epochs']
        self.activations = design['unit_activations']
        self.drops = design['dropout']

        # attributes from Dataset class
        dataset = equation.Dataset('poisson')
        dataset.load()
        self.domain = dataset.domain()
        self.train_data = dataset.train
        self.val_data = dataset.validate 
        self.test_data = dataset.test
        self.theta_names = dataset.theta_names
        self.get_solution = dataset.vectorize_u
        
        # set log directory
        self.log_dir = "logs/fit/" 
        self.log_dir += datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    # build feed forward network
    def build_model(self):
        # input Phi
        Phi = Input(shape=self.train_data[0][0].shape, name='Phi')
        for layer in range(self.num_layers):
            # first hidden layer
            if layer == 0:
                x = Dense(self.activations[layer][0],
                        self.activations[layer][1])(Phi)
            # other hidden layers
            elif layer < self.num_layers - 1:
                x = Dropout(self.drops[layer-1])(x)
                x = Dense(self.activations[layer][0],
                        self.activations[layer][1])(x)
            # final layer
            else:
                theta = Dense(self.activations[layer][0],
                        self.activations[layer][1],
                        name='theta')(x)

        self.model = Model(Phi, theta)
        self.model.compile(self.optim, self.loss)

    ## TRAIN THE MODEL
    def train(self, verbose=0, sigma=0, transform=False):
        """
        Compiles the model, prints a summary, fits to data
        The boolean transform rescales the data if True (default),
        and uses raw data otherwise.

        The input sigma controls the noise for the train/val inputs
        """
        # load data and targets
        Phi_train, theta_Phi_train = deepcopy(self.train_data)
        Phi_val, theta_Phi_val = deepcopy(self.val_data)
        
        # add noise
        Phi_train, train_noise = plotter.add_noise(Phi_train, sigma, seed=2)
        Phi_val, val_noise = plotter.add_noise(Phi_val, sigma, seed=3)

        self.transformed = transform
        if transform:
            # transform train and val inputs
            Phi_train_tformer = preprocessing.MaxAbsScaler()
            Phi_val_tformer = preprocessing.MaxAbsScaler()
            Phi_train = Phi_train_tformer.fit_transform(Phi_train)
            Phi_val = Phi_val_tformer.fit_transform(Phi_val)

            # transform train and val targets
            theta_Phi_train_tformer = preprocessing.MaxAbsScaler()
            theta_Phi_val_tformer = preprocessing.MaxAbsScaler()
            theta_Phi_train = theta_Phi_train_tformer.fit_transform(theta_Phi_train)
            theta_Phi_val = theta_Phi_val_tformer.fit_transform(theta_Phi_val)

        # compile and print summary
        self.build_model()
        self.model.summary()

        # make callbacks and fit model
        callbacks = self.get_callbacks()
        self.model.fit(x=Phi_train, y=theta_Phi_train,
                 validation_data=(Phi_val, theta_Phi_val), 
                 batch_size = self.batch_size,
                 epochs = self.epochs,
                 callbacks = callbacks,
                 verbose=verbose)
        
####################
# PLOTTING METHODS #
####################
    def theta_from_Phi(self, Phi):
        return self.model.predict(Phi)

    def u_from_Phi(self, Phi):
        theta = self.model.predict(Phi)
        return self.get_solution(theta)

    def u_from_theta(self, theta):
        return self.get_solution(theta)

    def plot_theta_fit(self, sigma=0, seed=23, transform=True):
        Phi, theta_Phi = deepcopy(self.test_data)
        plotter.theta_fit(Phi, 
                          theta_Phi, 
                          self.theta_from_Phi, 
                          sigma, 
                          seed,
                          transform)

    def plot_solution_fit(self, sigma=0, seed=23):
        # get Phi, theta_Phi, theta
        Phi, theta_Phi = deepcopy(self.test_data)
        theta = self.model.predict(Phi)
        plotter.solution_fit(Phi, 
                             theta_Phi, 
                             self.theta_from_Phi, 
                             self.u_from_Phi, 
                             sigma, 
                             seed)

##########################
# BOOTSTRAP PLOT METHODS #
##########################
    def fitter(self, train):
        """
        Fits the model given some training data
        """
        self.build_model()
        self.model.fit(x=train[0], y=train[1],
                 batch_size=self.batch_size,
                 epochs=self.epochs, 
                 verbose=0)

    def evaluater(self, test):
        """
        Evaluates a trained model with test data
        """
        test_loss = self.model.evaluate(test[0], test[1], verbose=0)
        return test_loss

    def predicter(self, test):
        """
        Predicts with a trained model on test data
        """
        return self.model.predict(test[0])

    def bootstrap(self, num_boots, train_sigma=0, test_sigma=0, size=0.6):
        """
        Generates num_boots bootstrap samples,
        fits a model to each sample,
        evaluates the model fit on fixed test data,
        predicts with the fixed test data,
        saves evaluations and predictions as class attributes
        """
        # add noise to train inputs
        x_train, y_train = deepcopy(self.train_data)
        x_train, noise = plotter.add_noise(x_train, train_sigma, seed=2)
        train_data = (x_train, y_train)

        # add noise to test inputs
        x_test, y_test = deepcopy(self.test_data)
        x_test, noise = plotter.add_noise(x_test, test_sigma) 
        test_data = (x_test, y_test)

        # save noisy data and sigmas
        data = (train_data, test_data)
        self.train_sigma = train_sigma
        self.test_sigma = test_sigma

        samp_size = int(size * len(x_train))
        print('Bootstrapping with %d samples of size %d' % (num_boots, samp_size))
        results = bootstrap(num_boots, 
                data, 
                self.fitter, 
                self.evaluater, 
                self.predicter,
                size)
        print('done')
        self.boot_evals, self.boot_preds = results

    def plot_theta_boot(self):
        """
        Plots bootstrap means vs true theta values
        with 95% credible region errorbars
        """
        plotter.theta_boot(self.test_data, 
                self.boot_preds, 
                [self.train_sigma, self.test_sigma])

    def plot_solution_boot(self):
        """
        Plots mean bootstrap solution curve bounded above and below
        by the credible intervals.
        """
        plotter.solution_boot(self.test_data, 
                self.boot_preds, 
                self.u_from_theta,
                [self.train_sigma, self.test_sigma])


#############
# CALLBACKS #
#############
    def get_callbacks(self):
        callbacks = []
        for cb_code in self.callbacks:
            if cb_code == 'tensorboard':
                tb_callback = self.tensorboard_callback()
                callbacks.append(tb_callback)
            if cb_code == 'learning_rate':
                lr_callback = self.learning_rate_callback()
                callbacks.append(lr_callback)
        return callbacks

    # tensorboard logs callback
    def tensorboard_callback(self):
        tb_callback = TensorBoard(self.log_dir, histogram_freq=1)
        return tb_callback

    # lr scheduler callback
    def learning_rate_callback(self):
        """
        Defines a piecewise constant function to decrease
        the learning rate as training progresses.
        This is done to fine-tune the gradient descent
        so the optimizer doesn't hop over relative minimums.
        """
        def lr_sched(epoch):
            if epoch < 30:
                return 0.001
            elif epoch < 60:
                return 0.0001
            elif epoch < 90:
                return 0.00005
            elif epoch < 120:
                return 0.00005
            else:
                return 0.00005
        lr_callback = LearningRateScheduler(lr_sched)
        return lr_callback


##################
# ERROR CHECKING #
##################
    def print_errors(self, verbose=False):
        """
        Trained model is used to predict on test data,
        then computes the relative error (RE) to print
        statistics derived from RE.
        """
        # calculate relative error statistics
        print('min y test scale', np.min(theta_Phi_test_tform, axis=0))
        rel_err = np.abs(1.0 - theta_test_tform / theta_Phi_test_tform)
        cum_rel_err = np.sum(rel_err, axis=1) # row sums of rel_err
        max_err = np.argmax(cum_rel_err)
        min_err = np.argmin(cum_rel_err)

        # print diagnostics
        print('\nRELATIVE ERRORS (%):')
        print('\n max relative error:', 100*np.max(rel_err, axis=0))
        print('\n min relative error:', 100*np.min(rel_err, axis=0), '\n')
        print('max cumulative:', 100*np.max(cum_rel_err))
        print('min cumulative:', 100*np.min(cum_rel_err))
        print('mean relative error:', 100*np.mean(rel_err, axis=0))

        if verbose:
            print('\nVERBOSE OUTPUT:')
            print('\nTRUE AND PREDICTED TARGETS:')
            print('max relative error true target:', theta_Phi_test[max_err])
            print('max relative error predicted target:', theta_test[max_err])
            print('mean true target:', np.mean(theta_Phi_test, axis=0))
            print('mean predicted target:', np.mean(theta_test, axis=0))
            print('min relative error true target:', theta_Phi_test[min_err])
            print('min relative error predicted target:', theta_test[min_err])
            print('test input shape', Phi_test.shape)
            print('test targets shape', theta_Phi_test.shape)
            print('predicted normed target shape', theta_test_tform.shape)
            print('predicted normed target range', 
                    np.ptp(theta_test_tform, axis=0))
            print('target predicted range', np.ptp(theta_test, axis=0))
            print('true target range', np.ptp(theta_Phi_test, axis=0))
            print('rel_err shape', rel_err.shape)
            print('total rel_err shape', cum_rel_err.shape)
            print('max_err index', max_err)
            print('min_err index', min_err)
