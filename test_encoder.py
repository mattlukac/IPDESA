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
in_units = dataset.train[0].shape[1]
out_units = dataset.train[1].shape[1]


## DEFINE MODEL PARAMETERS
design = {'flavor':'ff',
          'unit_activations':[(in_units, 'relu'),
                              (70, 'relu'),
                              (40, 'relu'),
                              (10, 'relu'),
                              (out_units, 'relu')
                             ],
          'optimizer':'adam',
          'loss':'mse',
          'callbacks':['learning_rate', 'tensorboard'],
          'batch_size':15,
          'epochs':20,
         }

# TRAIN MODEL
model = encoder.Encoder(design, dataset)
model.train()

## PRINT DIAGNOSTICS:
#model.print_errors()
