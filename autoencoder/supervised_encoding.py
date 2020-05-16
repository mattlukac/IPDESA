from sae.supervised_encoder import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--train_on', 
        help='name of directory containing the training data')
args = parser.parse_args()

## LOAD DATA
dirName = 'training_data/' + args.train_on
u, Theta = load_data(dirName)

## LOAD CALLBACKS
tb_callback = tensorboard_callback()
lr_callback = learning_rate_callback()

## DEFINE MODEL PARAMETERS
design = {'flavor':'ff',
          'inputShape':u['train'][0,:].shape,
          'denseUnits':(100, 70, 40, 10, 3),
          'denseActivations':('relu', 'relu', 'relu', 'relu', 'relu')
         }
assert len(design['denseUnits']) == len(design['denseActivations'])

training = {'optimizer':'adam',
            'loss':'mae',
            'callbacks':[lr_callback, tb_callback],
            'batch_size':15, 
            'epochs':150,
            'x':u['train'],
            'y':Theta['train'],
            'val_dat':(u['val'], Theta['val']),
            'test_dat':(u['test'], Theta['test'])
           }


## TRAIN MODEL
model = Encoder(design)
train_model(model, training)

## PRINT DIAGNOSTICS:
print_relative_errors(model, test_dat = training['test_dat'])
