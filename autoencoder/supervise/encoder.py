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
        super().__init__()

        # attributes defined in design dictionary
        self.flavor = design['flavor']
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
        self.target_min = Dataset.target_min
        self.target_range = Dataset.target_range
        self.Theta_names = Dataset.Eqn.Theta_names
        
        # set log directory
        self.log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        # check target ranges
        print('INITIALIZED ENCODER CLASS')
        print('train y range', np.ptp(self.train_data[1], axis=0))
        print('val y range', np.ptp(self.val_data[1], axis=0))
        print('test y range', np.ptp(self.test_data[1], axis=0))

        # build feed forward network
        if self.flavor == 'ff':
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
                    #self.add(Dropout(drops[layer-1]))
                    self.add(Dense(unit_activ[layer][0], unit_activ[layer][1]))


    ## TRAIN THE MODEL
    def train(self):
        """
        Compiles the model, prints a summary, fits to data
        """
        # preprocess features and targets
        x_train, y_train = self.train_data 
        x_val, y_val = self.val_data 
        
        ## normalize inputs
        train_transformer = preprocessing.MaxAbsScaler()
        val_transformer = preprocessing.MaxAbsScaler()
        x_train_trans = train_transformer.fit_transform(x_train)
        x_val_trans = val_transformer.fit_transform(x_val)

        ## rescale targets to unit interval
        train_transformer = preprocessing.MaxAbsScaler()
        val_transformer = preprocessing.MaxAbsScaler()
        y_train_trans = train_transformer.fit_transform(y_train)
        y_val_trans = val_transformer.fit_transform(y_val)

        # compile and print summary
        self.compile(optimizer = self.optim,
                     loss = self.loss)
        #self.summary()

        # make callbacks and fit model
        callbacks = self.get_callbacks()

        # train model
        self.fit(x=x_train_trans, y=y_train_trans,
                 validation_data=(x_val_trans, y_val_trans), 
                 batch_size = self.batch_size,
                 epochs = self.epochs,
                 callbacks = callbacks,
                 verbose=2)
        
    def predict_plot(self):
        # get input and targets
        x_test, y_test = self.test_data
        test_transformer = preprocessing.MaxAbsScaler()
        test_transformer = preprocessing.MaxAbsScaler()
        x_test_trans = test_transformer.fit_transform(x_test)
        y_test_trans = test_transformer.fit_transform(y_test)

        # predict with trained model
        results = self.evaluate(x_test_trans, y_test_trans)
        print('test loss:', results)
        y_test_trans_pred = self.predict(x_test_trans)
        y_test_pred = test_transformer.inverse_transform(y_test_trans_pred)

        # plot transformed predicted vs true
        ncols = y_test.shape[1]
        fig, ax = plt.subplots(nrows=1, ncols=ncols, 
                               sharex=True, sharey=True,
                               figsize=(20,10))
        for col in range(ncols):
            ax[col].scatter(y_test_trans[:,col], y_test_trans_pred[:,col])
            ax[col].set_title(self.Theta_names[col], fontsize=18)
        fig.add_subplot(111, frame_on=False)
        plt.tick_params(labelcolor='none', bottom=False, left=False)
        plt.xlabel('Truth', fontsize=18)
        plt.ylabel('Prediction', fontsize=18)
        fig.savefig(self.log_dir + '/transformed_predict_vs_true.png')
        plt.close(fig)

        # plot inverse transformed predicted vs true
        fig, ax = plt.subplots(nrows=1, ncols=ncols, figsize=(20,10))
        for col in range(ncols):
            ax[col].scatter(y_test[:,col], y_test_pred[:,col])
            ax[col].set_title(self.Theta_names[col], fontsize=18)
        fig.add_subplot(111, frame_on=False)
        plt.tick_params(labelcolor='none', bottom=False, left=False)
        plt.xlabel('Truth', fontsize=18)
        plt.ylabel('Prediction', fontsize=18)
        fig.savefig(self.log_dir + '/predict_vs_true.png')
        plt.close(fig)

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
            print('range and min shapes', self.target_range.shape, self.target_min.shape)
            print('test input shape', x_test.shape)
            print('test targets shape', y_test.shape)
            print('predicted normed target shape', y_test_trans_pred.shape)
            print('predicted normed target range', np.ptp(y_test_trans_pred, axis=0))
            print('target predicted range', np.ptp(y_test_pred, axis=0))
            print('true target range', np.ptp(y_test, axis=0))
            print('rel_err shape', rel_err.shape)
            print('total rel_err shape', cum_rel_err.shape)
            print('max_err index', max_err)
            print('min_err index', min_err)

    # NORMALIZING METHODS
    def normalize_targets(self, targets):
        """
        Normalizes the dataset's targets
        sets so they are bounded below by 0 and above by 1
        This is the map x -> (x-a)/(b-a) for all x in (a,b)
        """
        print('NORMALIZING TARGETS...')
        print('og targs have range', np.ptp(targets, axis=0))
        targets -= self.target_min 
        targets /= self.target_range
        print('normed targs have range', np.ptp(targets, axis=0))
        return targets

    def invert_normalize_targets(self, targets):
        """
        Inverse map of normalize_targets()
        """
        assert targets.shape[1] == len(self.target_range)
        print('INVERTING NORMALIZED TARGETS...')
        print('normed targs have range', np.ptp(targets, axis=0))
        targets *= self.target_range
        targets += self.target_min
        print('og targs have range', np.ptp(targets, axis=0))
        return targets


    ## CALLBACKS
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

