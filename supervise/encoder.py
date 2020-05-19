import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import TensorBoard, LearningRateScheduler
from tensorflow.keras import Sequential
import numpy as np
import datetime
from . import equation


class Encoder(Sequential):

    def __init__(self, design, Dataset):
        super().__init__()

        # attributes defined in design dictionary
        self.flavor = design['flavor']
        self.num_layers = len(design['unit_activations'])
        self.optimizer = design['optimizer']
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

        # build feed forward network
        if self.flavor == 'ff':
            unit_activ = design['unit_activations']
            # construct network
            for layer in range(self.num_layers):
                if layer == 0:
                    self.add(Dense(
                        units=unit_activ[layer][0],
                        activation=unit_activ[layer][1],
                        input_shape=self.train_data[0][0].shape))
                self.add(Dense(unit_activ[layer][0], unit_activ[layer][1]))


    ## TRAIN THE MODEL
    def train(self):
        """
        Compiles the model, prints a summary, fits to data
        """
        # compile and print summary
        self.compile(optimizer = self.optimizer,
                     loss = self.loss)
        self.summary()

        # make callbacks and fit model
        callbacks = self.get_callbacks()

        # normalize train and validation targets
        normed_train_targets = self.normalize_targets(self.train_data[1])
        normed_val_targets = self.normalize_targets(self.val_data[1])
#        val_in, val_out = self.val_data 
#        normed_val_targets = self.normalize_targets(val_out)

        # train model
        self.fit(x=self.train_data[0], y=normed_train_targets,
                 validation_data = (self.val_data[0], normed_val_targets),
                 batch_size = self.batch_size,
                 epochs = self.epochs,
                 callbacks = callbacks,
                 verbose=2)

    def get_callbacks(self):
        callbacks = []
        for cb_code in self.callbacks:
            if cb_code == 'tensorboard':
                tb_callback = tensorboard_callback()
                callbacks.append(tb_callback)
            if cb_code == 'learning_rate':
                lr_callback = learning_rate_callback()
                callbacks.append(lr_callback)
        return callbacks

    # NORMALIZING METHODS
    def normalize_targets(self, targets):
        """
        Normalizes the dataset's targets
        sets so they are bounded below by 0 and above by 1

        This is the map x -> (x-a)/(b-a) for all x in (a,b)
        """
        normalized_targets = (targets - self.target_min) / self.target_range
        return normalized_targets

    def invert_normalize_targets(self, targets):
        """
        Inverse map of normalize_targets()
        """
        assert targets.shape[1] == len(self.target_range)
        return targets * self.target_range + self.target_min

    ## CALCULATE RELATIVE ERRORS
    def print_errors(self, verbose=False):
        """
        Trained model is used to predict on test data,
        then computes the relative error (RE) to print
        statistics derived from RE.
        """
        # get input and targets
        test_input = self.test_data[0]
        target_true = self.test_data[1]
        print('training targets:\n', self.train_data[1])
        print('min train target:', np.min(self.train_data[1]))
        norm_target_predict = self.predict(test_input)

        # check test targets have same shape as predicted targets
        assert target_true.shape == norm_target_predict.shape

        # invert the normalization to avoid dividing by 0
        target_predict = self.invert_normalize_targets(norm_target_predict)
        print('predicted targets:\n', target_predict[0:7])

        # calculate relative error statistics
        relErr = np.abs((target_true - target_predict) / target_true)
        cumRelErr = np.sum(relErr, axis=1) # row sums of relErr
        maxErr = np.argmax(cumRelErr)
        minErr = np.argmin(cumRelErr)

        # print diagnostics
        print('max cumulative relative error:', np.max(cumRelErr))
        print('min cumulative relative error:', np.min(cumRelErr))
        print('mean relative error:', np.mean(relErr, axis=0))
        print('mean true target:', np.mean(target_true, axis=0))
        print('mean predicted target:', np.mean(target_predict, axis=0))
        print('\n max relative error (%):', 100*np.max(relErr, axis=0), '\n')
        print('max relative error true target:', target_true[maxErr])
        print('max relative error predicted target:', target_predict[maxErr])
        print('min relative error (%):', 100*np.min(relErr, axis=0))
        print('min relative error true target:', target_true[minErr])
        print('min relative error predicted target:', target_predict[minErr])

        if verbose:
            print('test input shape', test_input.shape)
            print('test targets shape', target_true.shape)
            print('predicted normed target shape', norm_target_predict.shape)
            print('predicted normed target range', 
                    np.ptp(norm_target_predict, axis=0) )
            print('target predicted range', np.ptp(target_predict, axis=0) )
            print('true target range', np.ptp(target_true, axis=0) )
            print('relErr shape', relErr.shape)
            print('total relErr shape', cumRelErr.shape)
            print('maxErr index', maxErr)
            print('minErr index', minErr)


## CALLBACKS
# tensorboard logs callback
def tensorboard_callback():
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tb_callback = TensorBoard(log_dir, histogram_freq=1)
    return tb_callback

# lr scheduler callback
def learning_rate_callback():
    """
    Defines a piecewise constant function to decrease
    the learning rate as training progresses.
    This is done to fine-tune the gradient descent
    so the optimizer doesn't hop over relative minimums.
    """
    def lr_sched(epoch):
        if epoch < 100:
            return 0.001
        elif epoch < 200:

            return 0.0001
        elif epoch < 300:
            return 0.00005
        elif epoch < 400:
            return 0.00005
        else:
            return 0.00005
    lr_callback = LearningRateScheduler(lr_sched)
    return lr_callback

