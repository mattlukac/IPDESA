import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import TensorBoard, LearningRateScheduler
from tensorflow.keras import Sequential
import numpy as np
import datetime
import pickle

## Encoder class
class Encoder(Sequential):

    # flavor is either 'ff' or 'cnn'
    def __init__(self, params):
        super(Encoder, self).__init__()
        self.flavor = params['flavor']
        numLayers = len(params['denseUnits'])
        in_shape = params['inputShape']
        if self.flavor == 'ff':
            units = params['denseUnits']
            activs = params['denseActivations']
            # construct network
            self.add(Dense(units=units[0], 
                           activation=activs[0], 
                           input_shape=in_shape))
            for l in range(1, numLayers):
                self.add(Dense(units[l], activs[l]))


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

## LOAD DATA
def load_data(dirName):
    """
    Loads the (data, targets), both are dictionaries
    with keys 'train', 'val', 'test'.
    The input dirName is simply the name of the 
    directory containing the pickled training data.
    """
    thePickle = open(dirName + '/data.pkl', 'rb')
    theData = pickle.load(thePickle)
    thePickle.close()
    return theData

## TRAIN MODEL
def train_model(model, training):
    """
    Compiles the model using the optimizer and loss function
    specified in the training dictionary.
    Then print a summary of the model.
    Finally, fits the model using training data, validation data,
    batch_size, epochs, and callbacks 
    specified in the training dictionary.
    """
    model.compile(optimizer = training['optimizer'], loss = training['loss'])
    model.summary()
    model.fit(x=training['x'], y=training['y'], 
              validation_data = training['val_dat'], 
              batch_size = training['batch_size'], 
              epochs = training['epochs'],
              callbacks = training['callbacks'])

## CALCULATE RELATIVE ERRORS
def print_relative_errors(model, test_dat):
    """
    Trained model is used to predict on test data tuple,
    then computes the relative error (RE) to print the 
    maximum, minimum, mean RE, as well as 
    the true and predicted Theta values associated with the max RE.
    """
    Theta_predict = model.predict(test_dat[0])
    # two cases: Theta is one dimensional or more
    if Theta_predict.shape[1] == 1:
        Theta_predict = np.squeeze(Theta_predict)
        relError = np.abs(test_dat[1] - Theta_predict)/test_dat[1]
        relErrMax = np.argmax(relError)
    else:
        relError = np.abs((test_dat[1] - Theta_predict)/test_dat[1])
        totalRelError = np.sum(relError, axis=0)
        relErrMax = np.argmax(totalRelError)
        relErrMin = np.argmin(totalRelError)
    print('max relative error:', np.max(relError[relErrMax, :]))
    print('min relative error:', np.min(relError[relErrMin, :]))
    print('mean relative error:', np.mean(relError, axis=0))
    print('max relative error true and predicted Theta:', 
            test_dat[1][relErrMax, :], 
            Theta_predict[relErrMax, :]) 
    print('mean predicted Theta:', np.mean(Theta_predict, axis=0))
    print('mean true Theta:', np.mean(test_dat[1], axis=0))
    print('min predicted Theta:', np.min(Theta_predict, axis=0))
