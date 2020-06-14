import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import TensorBoard, LearningRateScheduler
from tensorflow.keras import Sequential
from sklearn import preprocessing 
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import datetime
from . import equation
from . import plotter


class Encoder(Sequential):

    def __init__(self, design, Dataset):
        super(Encoder, self).__init__()

        # attributes defined in design dictionary
        self.num_layers = len(design['unit_activations'])
        self.optim = design['optimizer']
        self.loss = design['loss']
        self.callbacks = design['callbacks']
        self.batch_size = design['batch_size']
        self.epochs = design['epochs']

        # attributes from Dataset class
        self.domain = Dataset.domain()
        self.train_data = Dataset.train
        self.val_data = Dataset.validate 
        self.test_data = Dataset.test
        self.theta_names = Dataset.theta_names
        self.get_solution = Dataset.vectorize_u
        
        # set log directory
        self.log_dir = "logs/fit/" 
        self.log_dir += datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        # build feed forward network
        unit_activ = design['unit_activations']
        drops = design['dropout']
        # construct network
        for layer in range(self.num_layers):
            if layer == 0:
                self.add(Dense(
                    units=unit_activ[layer][0],
                    activation=unit_activ[layer][1],
                    input_shape=self.train_data[0][0].shape))
            else:
                self.add(Dropout(drops[layer-1]))
                self.add(Dense(
                    unit_activ[layer][0], unit_activ[layer][1]))

    ## TRAIN THE MODEL
    def train(self, transform=False):
        """
        Compiles the model, prints a summary, fits to data
        The boolean transform rescales the data if True (default),
        and uses raw data otherwise.
        """
        # load data and targets
        Phi_train, theta_Phi_train = self.train_data 
        Phi_val, theta_Phi_val = self.val_data 
        
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
        self.compile(optimizer = self.optim,
                     loss = self.loss)
        self.summary()

        # make callbacks and fit model
        callbacks = self.get_callbacks()
        self.fit(x=Phi_train, y=theta_Phi_train,
                 validation_data=(Phi_val, theta_Phi_val), 
                 batch_size = self.batch_size,
                 epochs = self.epochs,
                 callbacks = callbacks,
                 verbose=2)
        
####################
# PLOTTING METHODS #
####################
    def theta_from_Phi(self, Phi):
        return self.predict(Phi)

    def u_from_Phi(self, Phi):
        theta = self.predict(Phi)
        return self.get_solution(theta)

    def plot_theta_fit(self, transform=True, sigma=0, seed=23):
        Phi, theta_Phi = deepcopy(self.test_data)
        plotter.theta_fit(Phi, theta_Phi, 
                          self.theta_from_Phi, 
                          transform, 
                          sigma, 
                          seed)

    def plot_solution_fit(self, sigma=0, seed=23):
        # get Phi, theta_Phi, u_theta, theta
        Phi, theta_Phi = deepcopy(self.test_data)
        theta = self.predict(Phi)
        plotter.solution_fit(Phi, 
                             theta_Phi, 
                             self.theta_from_Phi, 
                             self.u_from_Phi, 
                             sigma, 
                             seed)

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
