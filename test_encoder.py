from supervise import equation, encoder
#import os 
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import argparse
from tensorflow.random import set_seed
#from tensorflow.keras import Sequential
#from tensorflow.keras.layers import Dense
#import numpy as np 

set_seed(23)

# get the equation name
parser = argparse.ArgumentParser()
parser.add_argument('--train_on', 
        help='name of file containing the training data')
args = parser.parse_args()

eqn_name = args.train_on

## LOAD DATA
dataset = equation.Dataset(eqn_name)
dataset.load()
inUnits = dataset.train[0].shape[1]
outUnits = dataset.train[1].shape[1]


## DEFINE MODEL PARAMETERS
design = {'flavor':'ff',
          'unit_activations':[(inUnits, 'relu'),
                              (70, 'relu'),
                              (40, 'relu'),
                              (10, 'relu'),
                              (outUnits, 'relu')
                             ],
          'optimizer':'adam',
          'loss':'mse',
          'callbacks':['learning_rate', 'tensorboard'],
          'batch_size':15,
          'epochs':20,
         }

##### emulate Encoder class #####
#def normalize_targets(targets, target_min, target_range):
#    return (targets - target_min) / target_range
#
#def train_encoder(design, Dataset):
#    # attributes defined in design dictionary
#    flavor = design['flavor']
#    num_layers = len(design['unit_activations'])
#    optim = design['optimizer']
#    loss = design['loss']
#    batch_size = design['batch_size']
#    epochs = design['epochs']
#
#    # attributes from Dataset class
#    train_data = Dataset.train
#    val_data = Dataset.validate
#    test_data = Dataset.test
#    target_min = Dataset.target_min
#    target_range = Dataset.target_range
#
#    # build feed forward network
#    if flavor == 'ff':
#        model = Sequential()
#        unit_activ = design['unit_activations']
#        # construct network
#        for layer in range(num_layers):
#            if layer == 0:
#                model.add(Dense(
#                    units=unit_activ[layer][0],
#                    activation=unit_activ[layer][1],
#                    input_shape=train_data[0][0].shape))
#            else:
#                model.add(Dense(unit_activ[layer][0], unit_activ[layer][1]))
#
#    # compile and print summary
#    model.compile(optimizer = optim, loss = loss)
#    model.summary()
#
#    # normalize train and validation targets
#    normed_train_targets = normalize_targets(train_data[1], target_min, target_range)
#    normed_val_targets = normalize_targets(val_data[1], target_min, target_range)
#
#    # train model
#    model.fit(x=train_data[0], y=train_data[1],
#              validation_data=val_data,
#              batch_size=batch_size,
#              epochs=epochs,
#              verbose=2)
#
#train_encoder(design, dataset)


##### using Encoder class ######
# TRAIN MODEL
model = encoder.Encoder(design, dataset)
model.train()

## PRINT DIAGNOSTICS:
#model.print_errors()
