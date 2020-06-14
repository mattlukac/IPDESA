import tensorflow as tf
from tensorflow.random import set_seed
from tensorflow.keras.layers import *
from sklearn import preprocessing
from . import equation, plotter
from copy import deepcopy
import numpy as np 
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 26})


class AnalyticAutoEncoder:

    def __init__(self, epochs=20, batch_size=25, lr=0.001):
        self.epochs = epochs 
        self.batch_size = batch_size
        self.lr = lr
        
        # load data
        data = equation.Dataset('poisson')
        data.load()
        self.domain = data.domain()
        self.train_data = data.train
        self.val_data = data.validate
        self.test_data = data.test 

        Phi_train, theta_train = data.train
        Phi_val, theta_val = data.validate
        Phi_test, theta_test = data.test
        
        # data probably shouldn't be rescaled to interpret theta, but hey
        # maxabs scalers
        Phi_train_tformer = preprocessing.MaxAbsScaler()
        theta_train_tformer = preprocessing.MaxAbsScaler()
        Phi_val_tformer = preprocessing.MaxAbsScaler()
        theta_val_tformer = preprocessing.MaxAbsScaler()
        Phi_test_tformer = preprocessing.MaxAbsScaler()
        theta_test_tformer = preprocessing.MaxAbsScaler()

        # rescale data
        Phi_train_trans = Phi_train_tformer.fit_transform(Phi_train)
        theta_train_trans = theta_train_tformer.fit_transform(theta_train)
        Phi_val_trans = Phi_val_tformer.fit_transform(Phi_val)
        theta_val_trans = theta_val_tformer.fit_transform(theta_val)
        Phi_test_trans = Phi_test_tformer.fit_transform(Phi_test)
        theta_test_trans = theta_test_tformer.fit_transform(theta_test)

        # transformed data attributes
        self.train_data_tform = (Phi_train_trans, theta_train_trans)
        self.val_data_tform = (Phi_val_trans, theta_val_trans)
        self.test_data_tform = (Phi_test_trans, theta_test_trans)

    def decoder(self, theta):
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
        # use eager tensors
        tf.config.experimental_run_functions_eagerly(True)

        # network paramters 
        latent_dim = 3
        input_shape = (100,)

        # build the network
        hidden_activation = ['linear', 'tanh']
        Phi = Input(shape=input_shape)
        x = Dense(20, hidden_activation[self.transformed], name='hidden')(Phi)
        theta = Dense(latent_dim, 'linear', name='theta')(x)
        u = Lambda(self.decoder, name='u')(theta)
        self.model = tf.keras.Model(Phi, u)
        optimizer = tf.keras.optimizers.Adam(self.lr)
        self.model.compile('adam', 'mse')

        # model that extracts latent theta
        self.get_theta = tf.keras.Model(self.model.input, 
                self.model.get_layer('theta').output)

    def train(self, transform=False, verbose=0, seed=23):
        # assign training, validation, and test data
        self.transformed = transform
        if transform:
            Phi_train = self.train_data_tform[0]
            Phi_val = self.val_data_tform[0]
            Phi_test = self.test_data_tform[0]
        else:
            Phi_train = self.train_data[0]
            Phi_val = self.val_data[0]
            Phi_test = self.test_data[0]

        # make the network
        set_seed(seed)
        self.build_model()

        # train and print losses
        print('training...')
        self.model.fit(x=Phi_train, y=Phi_train,
                       validation_data=(Phi_val, Phi_val),
                       epochs=self.epochs,
                       batch_size=self.batch_size,
                       verbose=verbose)
        print('--------')
        print('Reconstructed Phi MSE')
        print('Training:', 
                self.model.evaluate(Phi_train, Phi_train, verbose=0))
        print('Validation:',
                self.model.evaluate(Phi_val, Phi_val, verbose=0))
        print('Test:',
                self.model.evaluate(Phi_test, Phi_test, verbose=0))
        
        # extract latent space theta and compute MSE
        theta_test = self.get_theta(Phi_test).numpy()
        theta_test_mse = np.mean((theta_test - self.test_data[1]) ** 2, axis=0)
        print('--------')
        print('Latent theta MSE:', theta_test_mse)

    def test_noise(self, sigma=0.0):
        Phi_test, theta_test  = self.test_data
        noise = np.random.randn(Phi_test.size).reshape(400, 100)
        noise = np.reshape(noise, Phi_test.shape)
        Phi_test_noisy = Phi_test + sigma * noise
        print('Reconstructed Noisy Phi MSE:', 
                self.model.evaluate(Phi_test_noisy, Phi_test_noisy, 
                    verbose=0))
        u_test_noisy = self.model.predict(Phi_test_noisy)
        noisy_theta_test = self.get_theta(Phi_test_noisy).numpy()
        noisy_theta_mse = np.mean((theta_test - noisy_theta_test))
        print('Noisy theta MSE:', noisy_theta_mse)


################
# PLOT METHODS #
################

    def plot_theta_fit(self, transform=True, sigma=0, seed=23):
        Phi, theta_Phi = deepcopy(self.test_data)
        plotter.theta_fit(Phi, theta_Phi,
                          self.theta_from_Phi,
                          transform,
                          sigma,
                          seed)

    def plot_solution_fit(self, sigma=0, seed=23):
        # get Phi, theta_Phi, theta
        Phi, theta_Phi = deepcopy(self.test_data)
        plotter.solution_fit(Phi, theta_Phi,
                             self.theta_from_Phi,
                             self.u_from_Phi,
                             sigma, seed)
        
    def theta_from_Phi(self, Phi):
        return self.get_theta(Phi).numpy()

    def u_from_Phi(self, Phi):
        return self.model.predict(Phi)

