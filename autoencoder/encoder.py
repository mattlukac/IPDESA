import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
import numpy as np
import datetime

# load data
u_train = np.load('u_train.npy')
D_train = np.load('D_train.npy')
u_val = np.load('u_val.npy')
D_val = np.load('D_val.npy')
u_test = np.load('u_test.npy')
D_test = np.load('D_test.npy')

# custom rescaled sigmoid for final activation
#def scaled_sig(x):
#    return(2*tf.keras.activations.sigmoid(x))

# network layers
layers = [tf.keras.layers.InputLayer(input_shape=(200,)),
          tf.keras.layers.Dense(100, 'sigmoid'),
          tf.keras.layers.Dense(80, 'sigmoid'),
          tf.keras.layers.Dense(60, 'sigmoid'),
          tf.keras.layers.Dense(30, 'sigmoid'),
          tf.keras.layers.Dense(1, 'sigmoid')]
#          tf.keras.layers.Activation(scaled_sig)]
model = tf.keras.Sequential(layers)

model.compile(optimizer = tf.keras.optimizers.Adam(lr=0.0002),
              loss = 'mae')
print(model.summary())

# tensorboard logs
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# lr scheduler callback
def lr_sched(epoch):
    if epoch < 100:
        return 0.001
    elif epoch < 200:
        return 0.0005
    elif epoch < 300:
        return 0.0001
    elif epoch < 400:
        return 0.00001
    else:
        return 0.000001
lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_sched)
# fit model
model.fit(u_train, D_train, 
          validation_data = (u_val, D_val), 
          epochs=500, 
          batch_size = 15, 
          callbacks = [tensorboard_callback, lr_callback])

# relative errors on test set
D_predict = np.squeeze(model.predict(u_test))
relError = np.abs(D_test - D_predict)/D_test
print('max relError:', np.max(relError))
print('min relError:', np.min(relError))
print('mean relError:', np.mean(relError))
print('max relError D and predicted D:', D_test[np.argmax(relError)], D_predict[np.argmax(relError)]) 
