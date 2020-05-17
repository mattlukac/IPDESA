import supervise
from os import path
import argparse
import importlib

# get the equation type
parser = argparse.ArgumentParser()
parser.add_argument('--train_on', 
        help='name of directory containing the training data')
args = parser.parse_args()

problem = importlib.import_module()

eqn_name = args.train_on
eqn_path = 'data/' + eqn_name + '.pkl'
eqn = supervise.equation.Equation(eqn_name)

## LOAD DATA
if not path.exists(eqn_path):
    # run generate_data.py
    dataset = supervise.equation.Dataset(eqn_name)
    train, val, test = dataset.load()

## LOAD CALLBACKS
tb_callback = supervised.tensorboard_callback()
lr_callback = supervised.learning_rate_callback()

## DEFINE MODEL PARAMETERS
design = {'flavor':'ff',
          'inputShape':u['train'][0,:].shape,
          'denseUnits':(100, 70, 40, 10, 3),
          'denseActivations':('relu', 'relu', 'relu', 'relu', 'relu'),
          'optimizer':'adam',
          'loss':'mae',
          'callbacks':[lr_callback, tb_callback],
          'batch_size':15,
          'epochs':150,
          'data':u['train'],
          'targets':Theta['train'],
          'val_data':(u['val'], Theta['val']),
          'test_data':(u['test'], Theta['test'])
         }
assert len(design['denseUnits']) == len(design['denseActivations'])


## TRAIN MODEL
model = supervised.Encoder(design)
model.train()

## PRINT DIAGNOSTICS:
model.print_errors()
