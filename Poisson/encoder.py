import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
from tensorflow.keras.layers import *
from tensorflow.keras import Model
from tensorflow.keras.callbacks import TensorBoard, LearningRateScheduler
from tensorflow.random import set_seed
from sklearn import preprocessing 
from copy import deepcopy
import numpy as np
import datetime
from . import equation, plotter, tools
from .bootstrapper import ensemble_bootstrap, parametric_bootstrap


class Encoder:

    def __init__(self, design=None):

        # attributes from Dataset class
        dataset = equation.Dataset('poisson')
        dataset.load()
        self.domain = dataset.domain()
        self.train_data = dataset.train
        self.val_data = dataset.validate 
        self.test_data = dataset.test
        self.theta_names = dataset.theta_names
        self.get_solution = dataset.vectorize_u
        
        # attributes defined in design dictionary
        if design is not None:
            self.num_layers = len(design['unit_activations'])
            self.optim = design['optimizer']
            self.loss = design['loss']
            self.callbacks = design['callbacks']
            self.batch_size = design['batch_size']
            self.epochs = design['epochs']
            self.activations = design['unit_activations']
            self.drops = design['dropout']
        # default design
        else:
            self.num_layers = 2
            self.optim = 'adam'
            self.loss = 'mse'
            self.callbacks = []
            self.batch_size = 25
            self.epochs = 25
            self.activations = [(20, 'linear'), (3, 'linear')]
            self.drops = [0.0]

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
#                x = LeakyReLU()(x)
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
    def train(self, verbose=0, sigma=0, seed=23, transform=False):
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
        Phi_train, train_noise = tools.add_noise(Phi_train, sigma, seed=2)
        Phi_val, val_noise = tools.add_noise(Phi_val, sigma, seed=3)

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
        set_seed(seed)
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
        print('test mse:', self.model.evaluate(self.test_data[0], self.test_data[1]))
        print('test thetas:', self.model.predict(self.test_data[0]))
        
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
        Phi, _ = tools.add_noise(Phi, sigma, seed)
        theta = self.theta_from_Phi(Phi)

        plotter.theta_fit(Phi, theta_Phi, 
                          theta, 
                          seed,
                          transform)
        plotter.show()

    def plot_solution_fit(self, sigma=0, seed=23):
        # get Phi, theta_Phi, theta
        _, theta_Phi = deepcopy(self.test_data)
        Phi, noise, sample = self.solution_fit_sample(sigma)
        theta_Phi = theta_Phi[sample]
        theta = self.theta_from_Phi(Phi)
        u_theta = self.u_from_Phi(Phi)

        plotter.solution_fit(Phi, noise, theta_Phi, 
                             theta, 
                             u_theta, 
                             sigma, 
                             seed)
        plotter.show()
        
    def solution_fit_sample(self, sigma=0):
        # get Phi, theta_Phi, theta
        Phi, theta_Phi = deepcopy(self.test_data)
        Phi, noise = tools.add_noise(Phi, sigma)

        num_plots = 9
        # get sample
        sample = np.random.randint(0, len(Phi)-1, num_plots)
        Phi, noise = Phi[sample], noise[sample]
        return Phi, noise, sample

##########################
# BOOTSTRAP PLOT METHODS #
##########################
    def fitter(self, train):
        """ Fits the model given some training data """
        self.build_model()
        self.model.fit(x=train[0], y=train[1],
                 batch_size=self.batch_size,
                 epochs=self.epochs, 
                 verbose=0)

    def evaluater(self, test):
        """ Evaluates a trained model with test data """
        test_loss = self.model.evaluate(test[0], test[1], verbose=0)
        return test_loss

    def predicter(self, test):
        """ Predicts with a trained model on test data """
        return self.model.predict(test[0])

    def ensemble_bootstrap(self, num_boots, train_sigma=0, test_sigma=0, size=0.6):
        """
        Generates num_boots bootstrap samples,
        fits a model to each sample,
        evaluates the model fit on fixed test data,
        predicts with the fixed test data,
        saves evaluations and predictions as class attributes
        """
        # add noise to train inputs
        x_train, y_train = deepcopy(self.train_data)
        x_train, noise = tools.add_noise(x_train, train_sigma, seed=2)
        train_data = (x_train, y_train)

        # add noise to test inputs
        x_test, y_test = deepcopy(self.test_data)
        x_test, noise = tools.add_noise(x_test, test_sigma) 
        test_data = (x_test, y_test)

        # save noisy data and sigmas
        data = (train_data, test_data)
        self.train_sigma = train_sigma
        self.test_sigma = test_sigma

        samp_size = int(size * len(x_train))
        print('Bootstrapping with %d samples of size %d' % (num_boots, samp_size))
        results = ensemble_bootstrap(
                num_boots, 
                data, 
                self.fitter, 
                self.evaluater, 
                self.predicter,
                size)
        print('done')
        self.boot_evals, self.boot_preds = results

    def parametric_bootstrap(self,
            num_boots, 
            train_sigma=0, 
            test_sigma=0,
            seed=23):
        """
        Using a single trained network, we
          1) predict parameters theta
          2) simulate data from these parameters
          3) predict with simulated data
          4) get confidence intervals from empirical distribution of these errors
        To simulate data we will (deterministically) map theta -> u
        and then throw noise on top of u many times.
        """
        # add noise to train inputs
        x_train, y_train = deepcopy(self.train_data)
        x_train, noise = tools.add_noise(x_train, train_sigma, seed=2)
        train_data = (x_train, y_train)

        # get test data
        test_data = deepcopy(self.test_data)

        # save noisy data and sigmas
        data = (train_data, test_data)
        self.train_sigma = train_sigma
        self.test_sigma = test_sigma
        print('train and test sigmas:', self.train_sigma, self.test_sigma)

        print('performing parametric bootstrapping...')
        theta_hats = parametric_bootstrap(
                num_boots, 
                data, 
                self.fitter, 
                self.evaluater,
                self.predicter,
                test_sigma,
                seed)
        print('done')
        self.parametric_boot_thetas = theta_hats

    def plot_ensemble_theta_boot(self, se='bootstrap', verbose=False):
        """
        Plots bootstrap means vs true theta values
        with 95% credible region errorbars
        """
        plotter.theta_boot(
                self.test_data, 
                self.boot_preds, 
                [self.train_sigma, self.test_sigma],
                se,
                verbose)
        plotter.show()

    def plot_parametric_theta_boot(self, se='quantile', verbose=False):
        plotter.theta_boot(
                self.test_data, 
                self.parametric_boot_thetas,
                [self.train_sigma, self.test_sigma],
                se,
                verbose)

    def plot_solution_boot(self, se='bootstrap'):
        """
        Plots mean bootstrap solution curve bounded above and below
        by the credible intervals.
        """
        plotter.solution_boot(self.test_data, 
                self.boot_preds, 
                self.u_from_theta,
                [self.train_sigma, self.test_sigma],
                se)
        plotter.show()


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



if __name__ == '__main__':
    import equation 
    
    eqn_name = 'poisson'

    ## LOAD DATA
    dataset = equation.Dataset(eqn_name)
    dataset.load()
    in_units = dataset.train[0].shape[1]
    out_units = dataset.train[1].shape[1]


    ## DEFINE MODEL PARAMETERS
    design = {'flavor':'ff',
              'unit_activations':[(in_units, 'tanh'),
                                  (50, 'tanh'),
                                  (out_units, 'linear')
                                 ],
              'dropout':[0.1, 0.],
              'optimizer':'adam',
              'loss':'mse',
              'callbacks':['learning_rate', 'tensorboard'],
              'batch_size':25,
              'epochs':100,
             }

    set_seed(23)

    # TRAIN MODEL
    model = encoder.Encoder(design, dataset)
    model.train()
    model.predict_plot()

    ## PRINT DIAGNOSTICS:
    #model.print_errors(verbose=True)
