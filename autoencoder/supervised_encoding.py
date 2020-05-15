from sae.supervised_encoder import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', help='directory to the training data')
args = parser.parse_args()

## LOAD DATA
DIR = args.data_dir
u_train, Theta_train = load_data(DIR, 'train')
u_val, Theta_val = load_data(DIR, 'val')
u_test, Theta_test = load_data(DIR, 'test')

## LOAD CALLBACKS
tb_callback = tensorboard_callback()
lr_callback = learning_rate_callback()

## DEFINE MODEL PARAMETERS
design = {'flavor':'ff',
          'inputShape':u_train[0,:].shape,
          'denseUnits':(70, 30, 1),
          'denseActivations':('relu','sigmoid','sigmoid')
         }
assert len(design['denseUnits']) == len(design['denseActivations'])

training = {'optimizer':'adam',
            'loss':'mae',
            'callbacks':[lr_callback, tb_callback],
            'batch_size':15, 
            'epochs':500,
            'x':u_train,
            'y':Theta_train,
            'val_dat':(u_val, Theta_val),
            'test_dat':(u_test, Theta_test)
           }


## TRAIN MODEL
model = Encoder(design)
train_model(model, training)

## PRINT DIAGNOSTICS
print_relative_errors(model, test_dat = training['test_dat'])
