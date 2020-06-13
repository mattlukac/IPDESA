import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import TensorBoard, LearningRateScheduler
from tensorflow.keras import Sequential
from sklearn import preprocessing 
import numpy as np
import matplotlib.pyplot as plt
import datetime
from . import equation


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
        self.train_data = Dataset.train
        self.val_data = Dataset.validate 
        self.test_data = Dataset.test
        self.theta_names = Dataset.Eqn.theta_names
        self.get_solution = Dataset.Eqn.vectorize_u
        
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
    def train(self, transform=True):
        """
        Compiles the model, prints a summary, fits to data
        The boolean transform rescales the data if True (default),
        and uses raw data otherwise.
        """
        # load data and targets
        x_train, y_train = self.train_data 
        x_val, y_val = self.val_data 
        
        self.transformed = transform
        if transform:
            # transform train and val inputs
            x_train_transformer = preprocessing.MaxAbsScaler()
            x_val_transformer = preprocessing.MaxAbsScaler()
            x_train = x_train_transformer.fit_transform(x_train)
            x_val = x_val_transformer.fit_transform(x_val)

            # transform train and val targets
            y_train_transformer = preprocessing.MaxAbsScaler()
            y_val_transformer = preprocessing.MaxAbsScaler()
            y_train = y_train_transformer.fit_transform(y_train)
            y_val = y_val_transformer.fit_transform(y_val)

        # compile and print summary
        self.compile(optimizer = self.optim,
                     loss = self.loss)
        self.summary()

        # make callbacks and fit model
        callbacks = self.get_callbacks()
        self.fit(x=x_train, y=y_train,
                 validation_data=(x_val, y_val), 
                 batch_size = self.batch_size,
                 epochs = self.epochs,
                 callbacks = callbacks,
                 verbose=2)
        
####################
# PLOTTING METHODS #
####################
    def predict_plot(self, here=False):
        """
        Evaluates trained model on test data
        and compares predicted to true theta
        """
        # get input and targets
        x_test, y_test = self.test_data

        # initialize axes
        num_plots = y_test.shape[1]
        fig, ax = plt.subplots(nrows=1, ncols=num_plots, 
                sharey=self.transformed,
                figsize=(20,8), 
                dpi=200)

        # plot transformed thetas
        if self.transformed:
            filename = '/predict_vs_true_transform.png'

            # transform test data
            x_test_transformer = preprocessing.MaxAbsScaler()
            y_test_transformer = preprocessing.MaxAbsScaler()
            x_test_trans = x_test_transformer.fit_transform(x_test)
            y_test_trans = y_test_transformer.fit_transform(y_test)

            # evaluate, predict, and plot with trained model
            results = self.evaluate(x_test_trans, y_test_trans, verbose=0)
            print('transformed test loss:', results)
            y_test_hat_trans = self.predict(x_test_trans)
            y_test_hat = y_test_transformer.inverse_transform(y_test_hat_trans)
            self.plot_theta_fit(fig, ax, y_test_trans, y_test_hat_trans)

        # plot untransformed thetas
        else:
            filename = '/predict_vs_true.png'

            # evaluate, predict, and plot with trained model
            results = self.evaluate(x_test, y_test, verbose=0)
            print('test loss:', results)
            y_test_hat = self.predict(x_test)
            self.plot_theta_fit(fig, ax, y_test, y_test_hat)

        # show (if requested) and save plot
        if here:
            plt.show()
        if not os.path.exists(self.log_dir):
            os.mkdir(self.log_dir)
        fig.savefig(self.log_dir + filename)
        plt.close()

        # return untransformed thetas
        return y_test, y_test_hat

    def plot_theta_fit(self, fig, ax, theta_Phi, theta):
        """
        Plots theta vs theta_Phi
        Agnostic to data transformation
        """
        num_plots = len(ax)
        for i in range(num_plots):
            ax[i].scatter(theta_Phi[:,i], theta[:,i], 
                    alpha=0.7)
            ax[i].set_title(self.theta_names[i], fontsize=22)
            ax[i].plot([0,1], [0,1], 
                    transform=ax[i].transAxes, 
                    c='r',
                    linewidth=3)
        fig.add_subplot(111, frame_on=False)
        plt.tick_params(labelcolor='none', bottom=False, left=False)
        plt.xlabel('Truth', fontsize=22)
        plt.ylabel('Prediction', fontsize=22, labelpad=50.0)
        

    ## CALCULATE RELATIVE ERRORS
    def print_errors(self, verbose=False):
        """
        Trained model is used to predict on test data,
        then computes the relative error (RE) to print
        statistics derived from RE.
        """
        # calculate relative error statistics
        print('min y test scale', np.min(y_test_trans, axis=0))
        rel_err = np.abs(1.0 - y_test_trans_pred / y_test_trans)
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
            print('max relative error true target:', y_test[max_err])
            print('max relative error predicted target:', y_test_pred[max_err])
            print('mean true target:', np.mean(y_test, axis=0))
            print('mean predicted target:', np.mean(y_test_pred, axis=0))
            print('min relative error true target:', y_test[min_err])
            print('min relative error predicted target:', y_test_pred[min_err])
            print('test input shape', x_test.shape)
            print('test targets shape', y_test.shape)
            print('predicted normed target shape', y_test_trans_pred.shape)
            print('predicted normed target range', 
                    np.ptp(y_test_trans_pred, axis=0))
            print('target predicted range', np.ptp(y_test_pred, axis=0))
            print('true target range', np.ptp(y_test, axis=0))
            print('rel_err shape', rel_err.shape)
            print('total rel_err shape', cum_rel_err.shape)
            print('max_err index', max_err)
            print('min_err index', min_err)

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

