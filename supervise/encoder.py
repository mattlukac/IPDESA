import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import TensorBoard, LearningRateScheduler
from tensorflow.keras import Sequential
import numpy as np
import datetime
import equation


class Encoder(Sequential):

    # flavor is either 'ff' or 'cnn'
    def __init__(self, design, Dataset):
        super(Encoder, self).__init__()

        # attributes defined in design dictionary
        self.flavor = design['flavor']
        self.num_layers = len(design['denseUnits'])
        self.optimizer = design['optimizer']
        self.loss = design['loss']
        self.callbacks = design['callbacks']
        self.batch_size = design['batch_size']
        self.epochs = design['epochs']

        # attributes from Dataset class
        self.input = Dataset.train[0]
        self.input_shape = self.input.shape 
        self.targets = Dataset.train[1]
        self.val_data = Dataset.validate 
        self.test_data = Dataset.test

        # feed forward network
        if self.flavor == 'ff':
            units = design['denseUnits']
            activs = design['denseActivations']
            # construct network
            self.add(Dense(units=units[0],
                           activation=activs[0],
                           input_shape=self.input_shape))
            for layer in range(1, self.num_layers):
                self.add(Dense(units[layer], activs[layer]))

    ## TRAIN THE MODEL
    def train(self):
        """
        Compiles the model, prints a summary, fits to data
        """
        model.compile(optimizer = self.optimizer,
                      loss = self.loss)
        model.summary()

        # make callbacks and fit model
        callbacks = get_callbacks()

        model.fit(x=self.input, y=self.targets,
                  validation_data = self.val_dat,
                  batch_size = self.batch_size,
                  epochs = self.epochs,
                  callbacks = self.callbacks)

    def get_callbacks(self):
        callbacks = []
        for cb in self.callbacks:
            if cb == 'tensorboard':
                callbacks.append(tensorboard_callback())
            if cb == 'learning_rate':
                callbacks.append(learning_rate_callback())
        return callbacks

    ## CALCULATE RELATIVE ERRORS
    def print_errors(self):
        """
        Trained model is used to predict on test data,
        then computes the relative error (RE) to print
        statistics derived from RE.
        """
        test_input = self.test_data[0]
        test_targets = self.test_data[1]
        Theta_predict = model.predict(test_input)
        
        # two cases: Theta is one dimensional or more
        if Theta_predict.shape[1] == 1:
            Theta_predict = np.squeeze(Theta_predict)
            relError = np.abs(test_targets - Theta_predict)/test_targets
            relErrMax = np.argmax(relError)
        else:
            relError = np.abs((test_targets - Theta_predict)/test_targets)
            totalRelError = np.sum(relError, axis=0)
            relErrMax = np.argmax(totalRelError)
            relErrMin = np.argmin(totalRelError)
        print('max relative error:', np.max(relError[relErrMax, :]))
        print('min relative error:', np.min(relError[relErrMin, :]))
        print('mean relative error:', np.mean(relError, axis=0))
        print('max relative error true and predicted Theta:',
                test_targets[relErrMax, :],
                Theta_predict[relErrMax, :])
        print('mean predicted Theta:', np.mean(Theta_predict, axis=0))
        print('mean true Theta:', np.mean(test_targets, axis=0))
        print('min predicted Theta:', np.min(Theta_predict, axis=0))


## CALLBACKS
# tensorboard logs callback
def tensorboard_callback():
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tb_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
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

