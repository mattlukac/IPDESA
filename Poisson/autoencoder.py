import numpy as np 
from copy import deepcopy
import tensorflow as tf
from tensorflow.random import set_seed
from tensorflow.keras.layers import *
from sklearn import preprocessing
from . import equation, plotter
from .bootstrapper import bootstrap


class AnalyticAutoEncoder:

    def __init__(self, epochs=20, batch_size=25, lr=0.001):
        # model attributes
        self.epochs = epochs 
        self.batch_size = batch_size
        self.lr = lr
        self.optimizer = tf.keras.optimizers.Adam(lr)
        
        # load data
        data = equation.Dataset('poisson')
        data.load()
        self.domain = data.domain()
        self.train_data = data.train
        self.val_data = data.validate
        self.test_data = data.test 
        self.get_solution = data.vectorize_u

    def decoder(self, theta):
        """ Tensor operations to calculate analytic solution """
        c, b0, b1 = tf.split(theta, 3, axis=1)
        # x and x^2
        x = np.linspace(0., 1., 100)
        x2 = x ** 2
        # -c/2 x^2
        ux2 = tf.math.divide(c, -2.)
        ux2 = tf.math.multiply(ux2, x2)
        # (c/2 + b1 - b0) x
        ux = tf.math.divide(c, 2.)
        ux = tf.math.add(ux, b1)
        ux = tf.math.subtract(ux, b0)
        ux = tf.math.multiply(ux, x)
        # b0
        u = tf.math.add(ux2, ux)
        u = tf.math.add(u, b0)
        return u

    def build_model(self):
        """ Builds autoencoder and method to extract latent theta """
        # use eager tensors
        tf.config.experimental_run_functions_eagerly(True)

        # network parameters 
        input_shape = self.train_data[0].shape[1]
        latent_dim = self.train_data[1].shape[1]

        # encoder and decoder
        Phi = Input(shape=input_shape)
        x = Dense(20, 'linear', name='hidden')(Phi)
        theta = Dense(latent_dim, 'linear', name='theta')(x)
        u = Lambda(self.decoder, name='u')(theta)

        # build model
        self.model = tf.keras.Model(Phi, u)
        self.model.compile(self.optimizer, 'mse')

        # model that extracts latent theta
        self.get_theta = tf.keras.Model(self.model.input, 
                self.model.get_layer('theta').output)

    def train(self, sigma=0, seed=23, verbose=0):
        # add noise to data
        train_data = self.noisify(self.train_data, sigma, seed=2)
        val_data = self.noisify(self.val_data, sigma, seed=3)
        test_data = self.noisify(self.test_data, sigma)

        # make the network
        set_seed(seed)
        self.build_model()

        self.theta_history = LatentThetaHistory(self.test_data[0])

        # train and print losses
        print('training...')
        self.model.fit(x=train_data[0], y=train_data[0],
                       validation_data=(val_data[0], val_data[0]),
                       epochs=self.epochs,
                       batch_size=self.batch_size,
                       verbose=verbose, 
                       callbacks=[self.theta_history])
        print('--------')
        print('Reconstructed Phi MSE')
        print('Training:', 
                self.model.evaluate(train_data[0], train_data[0], verbose=0))
        print('Validation:',
                self.model.evaluate(val_data[0], val_data[0], verbose=0))
        print('Test:',
                self.model.evaluate(test_data[0], test_data[0], verbose=0))
        
        # extract latent space theta and compute MSE
        theta_test = self.theta_from_Phi(test_data[0])
        theta_test_mse = np.mean((theta_test - test_data[1]) ** 2, axis=0)
        print('--------')
        print('Latent theta MSE:', theta_test_mse)


################
# PLOT METHODS #
################

    def plot_theta_fit(self, sigma=0, seed=23, transform=True, history=None):
        Phi, theta_Phi = self.noisify(self.test_data, sigma)
        theta = self.theta_from_Phi(Phi)

        plotter.theta_fit(Phi, theta_Phi,
                          theta,
                          sigma,
                          seed,
                          transform)
        plotter.show()
        
    def plot_solution_fit(self, sigma=0, seed=23):
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
    
    def save_theta_epochs_frame(self, frame_num):
        theta_frame = self.theta_history.epochs[frame_num]
        Phi, theta_Phi = deepcopy(self.test_data)

        frame_path = 'visuals/theta_epochs/frames/'
        frame_plot = lambda : plotter.theta_fit(Phi, 
                                                theta_Phi, 
                                                theta_frame, 
                                                hold=True)

        plotter.save_frame(frame_num, frame_path, frame_plot)

    def save_theta_epochs_mp4(self):
        # function that saves frame given fram_num
        frame_saver = self.save_theta_epochs_frame
        # settings dictionary
        settings = dict()
        settings['mp4_path'] = 'visuals/theta_epochs/learn_theta.mp4'
        settings['frame_path'] = 'visuals/theta_epochs/frames/'
        settings['num_frames'] = len(self.theta_history.epochs)
        settings['duration'] = 3

        plotter.save_mp4(frame_saver, settings)

    def solution_fit_sample(self, sigma=0):
        # get Phi, theta_Phi, theta
        Phi, theta_Phi = deepcopy(self.test_data)
        Phi, noise = plotter.add_noise(Phi, sigma)
        
        num_plots = 9
        # get sample
        sample = np.random.randint(0, len(Phi)-1, num_plots)
        Phi, noise = Phi[sample], noise[sample]
        return Phi, noise, sample

    def save_solution_epochs_frame(self, epoch, sigma=0):
        num_eps = len(self.theta_history.epochs)
        theta_Phi = deepcopy(self.test_data[1])
        Phi, noise, sample = self.solution_fit_sample(sigma)
        theta_Phi = theta_Phi[sample]
        thetas = np.zeros((num_eps, len(sample), 3))
        u_thetas = np.zeros((num_eps,) + Phi.shape)
        for ep, theta in self.theta_history.epochs.items():
            thetas[ep] = theta[sample]
            u_thetas[ep] = self.u_from_theta(theta[sample])
        u_frame = u_thetas[epoch]

        # get common ylim over all epochs
        y_min = np.amin([Phi, np.amin(u_thetas, axis=0)], axis=(2,0))
        y_max = np.amax([Phi, np.amax(u_thetas, axis=0)], axis=(2,0))
        ylims = np.vstack((y_min, y_max)).T
        ylims *= 1.1

        plotter.solution_fit(Phi, noise, theta_Phi, 
                theta, 
                u_frame, 
                ylims=ylims,
                verbose=False)
        plotter.save_frame(epoch, 'visuals/solution_epochs/frames/')

    def save_solution_epochs_mp4(self):
        save_frame = self.save_solution_epochs_frame
        # settings dictionary
        settings = dict()
        settings['mp4_path'] = 'visuals/solution_epochs/learn_solution.mp4'
        settings['frame_path'] = 'visuals/solution_epochs/frames/'
        settings['num_frames'] = len(self.theta_history.epochs)
        settings['duration'] = 3

        plotter.save_mp4(save_frame, settings)
        
#####################
# BOOTSTRAP METHODS #
#####################

    def fitter(self, train):
        """ Fits the model given training data """
        self.build_model()
        self.model.fit(train[0], train[0],
                batch_size=self.batch_size, 
                epochs=self.batch_size, 
                verbose=0)

    def evaluater(self, test):
        """ Evaluates fitted model given test data """
        test_loss = self.model.evaluate(test[0], test[0], verbose=0)
        return test_loss
    
    def predicter(self, test):
        """ Predicts on fitted model given test data """
        Phi = test[0]
        return self.get_theta(Phi).numpy()

    def bootstrap(self, num_boots, train_sigma=0, test_sigma=0, size=0.6):
        """
        Performs num_boots bootstrap iterations 
        with sample 60% of original size
        Train on noisy data with stdev train_sigma
        Test on noisy data with stdev test_sigma
        """
        # add noise to train inputs
        train_data = self.noisify(self.train_data, train_sigma, seed=2)

        # add noise to test inputs
        test_data = self.noisify(self.test_data, test_sigma)

        # save noisy data and sigmas
        data = (train_data, test_data)
        self.train_sigma = train_sigma
        self.test_sigma = test_sigma

        print('Bootstrapping with %d boot samples' % num_boots)
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
        Plots predicted vs true thetas from bootstrapping results
        Inputs to theta_boot():
          self.test_data - used for ground truth thetas
          self.boot_preds - bootstrap results, has noise info built in
          sigmas - purely for the title of the plot
        """
        plotter.theta_boot(self.test_data, 
                self.boot_preds, 
                [self.train_sigma, self.test_sigma])
        plotter.show()

    def plot_solution_boot(self):
        """
        Plots true solution curves with bootstrap credible regions
        Inputs to solution_boot():
          self.test_data - ground truth information
          self.boot_preds - contains bootstrap results
          self.u_from_theta - reconstructed Phi given theta
          sigmas - used in plot title
        """
        plotter.solution_boot(self.test_data, 
                self.boot_preds, 
                self.u_from_theta,
                [self.train_sigma, self.test_sigma])
        plotter.show()

################
# MISC METHODS #
################

    def theta_from_Phi(self, Phi):
        """ Predict theta given Phi """
        return self.get_theta(Phi).numpy()

    def u_from_Phi(self, Phi):
        """ Predicts u_theta given Phi """
        return self.model.predict(Phi)

    def u_from_theta(self, theta):
        """ Reconstructs Phi given theta """
        return self.get_solution(theta)

    def noisify(self, data, sigma, seed=23):
        """ Add noise to input of data """
        x, y = deepcopy(data)
        x, noise = plotter.add_noise(x, sigma, seed)
        data_with_noise = (x, y)
        return data_with_noise

    def mse_from_Phi(self, sigma=0):
        """ Prints MSE from reconstructed Phi and latent theta """
        # compute u mse
        test_data = self.noisify(self.test_data, sigma)
        u_theta = self.u_from_Phi(test_data[0])
        u_mse = np.mean((u_theta - test_data[0]) ** 2)
        print('Reconstructed Noisy Phi MSE:', u_mse)

        # compute latent theta mse
        theta = self.theta_from_Phi(test_data[0])
        theta_mse = np.mean((test_data[1] - theta) ** 2, axis=0)
        print('Noisy theta MSE:', theta_mse)


class LatentThetaHistory(tf.keras.callbacks.Callback):
    """ Get latent theta from test set at each training batch """
    def __init__(self, test_Phi):
        self.test_Phi = test_Phi
#        self.batches = dict()
        self.epochs = dict()

    def set_model(self, model):
        self.model = model
        theta = self.model.get_layer(name='theta').output
        self.get_theta = tf.keras.Model(self.model.input, theta)

#    def on_train_batch_begin(self, batch, logs=None):
#        theta = self.get_theta.predict(self.test_Phi)
#        self.batches[batch] = theta

    def on_epoch_begin(self, epoch, logs=None):
        theta = self.get_theta.predict(self.test_Phi)
        self.epochs[epoch] = theta
