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

        # check target ranges
        print('INITIALIZED ENCODER CLASS')
        print('train targs have range', np.ptp(self.train_data[1], axis=0))
        print('val targs have range', np.ptp(self.val_data[1], axis=0))
        print('test targs have range', np.ptp(self.test_data[1], axis=0))

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
                    self.add(Dropout(drops[layer-1]))
                    self.add(Dense(unit_activ[layer][0], unit_activ[layer][1]))


    ## TRAIN THE MODEL
    def train(self):
        """
        Compiles the model, prints a summary, fits to data
        """
        # compile and print summary
        self.compile(optimizer = self.optim,
                     loss = self.loss)
        #self.summary()

        # make callbacks and fit model
        callbacks = self.get_callbacks()

        # normalize train and validation targets
        train_targets = self.train_data[1]
        val_targets = self.val_data[1]
        normed_train_targets = self.normalize_targets(train_targets.copy())
        normed_val_targets = self.normalize_targets(val_targets.copy())
        normed_val_data = (self.val_data[0], normed_val_targets)

        # test normalization procedure 
        print('train_targets range:', np.ptp(train_targets, axis=0))
        print('normed_train_targets range:', np.ptp(normed_train_targets, axis=0))

        # train model
        self.fit(x=self.train_data[0], y=normed_train_targets,
                 validation_data=normed_val_data, 
                 batch_size = self.batch_size,
                 epochs = self.epochs,
                 callbacks = callbacks,
                 verbose=2)
        
        # evaluate model
        test_targets = self.test_data[1]
        normed_test_targets = self.normalize_targets(test_targets.copy())
        results = self.evaluate(self.test_data[0], normed_test_targets)
        print('test loss', results)

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
        normed_target_pred = self.predict(test_input)

        # check test targets have same shape as predicted targets
        assert target_true.shape == normed_target_pred.shape

        # invert the normalization
        target_pred = self.invert_normalize_targets(normed_target_pred.copy())
        print('normed pred targ\n', normed_target_pred[0:10])
        print('pred targ\n', target_pred[0:10])
        print('true targ\n', target_true[0:10])

        # calculate relative error statistics
        rel_err = np.abs((target_true - target_pred) / target_true)
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
        
        print('\nTRUE AND PREDICTED TARGETS:')
        print('max relative error true target:', target_true[max_err])
        print('max relative error predicted target:', target_pred[max_err])
        print('mean true target:', np.mean(target_true, axis=0))
        print('mean predicted target:', np.mean(target_pred, axis=0))
        print('min relative error true target:', target_true[min_err])
        print('min relative error predicted target:', target_pred[min_err])

        if verbose:
            print('\nVERBOSE OUTPUT:')
            print('range and min shapes', self.target_range.shape, self.target_min.shape)
            print('test input shape', test_input.shape)
            print('test targets shape', target_true.shape)
            print('predicted normed target shape', normed_target_pred.shape)
            print('predicted normed target range', np.ptp(normed_target_pred, axis=0))
            print('target predicted range', np.ptp(target_pred, axis=0))
            print('true target range', np.ptp(target_true, axis=0))
            print('rel_err shape', rel_err.shape)
            print('total rel_err shape', cum_rel_err.shape)
            print('max_err index', max_err)
            print('min_err index', min_err)


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

