from sae.supervised_encoder import *


## LOAD DATA
DIR = 'training_data/gaussian_family/'
u_train, D_train = load_data(DIR, 'train')
u_val, D_val = load_data(DIR, 'val')
u_test, D_test = load_data(DIR, 'test')

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
            'y':D_train,
            'val_dat':(u_val, D_val),
            'test_dat':(u_test, D_test)
           }


## TRAIN MODEL
model = Encoder(design)
train_model(model, training)

## PRINT DIAGNOSTICS
print_relative_errors(model, test_dat = training['test_dat'])
