import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import TensorBoard, LearningRateScheduler
from tensorflow.keras import Sequential
import numpy as np
import datetime

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
def load_data(DIR, dclass):
    """
    Loads the (data, targets) and returns both.
    The solution function file must be named u_*
    and the latent parameters must be named Theta_*
    """
    u_ = np.load(DIR+'u_'+dclass+'.npy')
    Theta_ = np.load(DIR+'Theta_'+dclass+'.npy')
    return u_, Theta_

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
    Theta_predict = np.squeeze(model.predict(test_dat[0]))
    relError = np.abs(test_dat[1] - Theta_predict)/test_dat[1]
    relErrMax = np.argmax(relError)
    print('max relative error:', np.max(relError))
    print('min relative error:', np.min(relError))
    print('mean relative error:', np.mean(relError))
    print('max relative error test and predicted Theta:', 
          test_dat[1][relErrMax], 
          Theta_predict[relErrMax]) 
