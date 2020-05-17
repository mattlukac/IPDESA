import supervise.equation as equation
import supervise.encoder as encoder
from os import path
import argparse
import importlib

# get the equation name
parser = argparse.ArgumentParser()
parser.add_argument('--train_on', 
        help='name of directory containing the training data')
args = parser.parse_args()

eqn_name = args.train_on

## LOAD DATA
data = equation.Dataset(eqn_name)
data.load()

## DEFINE MODEL PARAMETERS
design = {'flavor':'ff',
          'denseUnits':(100, 70, 40, 10, 3),
          'denseActivations':('relu', 'relu', 'relu', 'relu', 'relu'),
          'optimizer':'adam',
          'loss':'mae',
          'callbacks':['learning_rate', 'tensorboard'],
          'batch_size':15,
          'epochs':50,
         }
assert len(design['denseUnits']) == len(design['denseActivations'])

## TRAIN MODEL
model = encoder.Encoder(design, data)
model.train()

## PRINT DIAGNOSTICS:
model.print_errors()
