from autoencoder.supervise import equation, encoder
import argparse
from tensorflow.random import set_seed

# get the equation name
parser = argparse.ArgumentParser()
parser.add_argument('--trainon', 
        help='name of file containing the configuration data')
args = parser.parse_args()

eqn_name = args.trainon

## LOAD DATA
dataset = equation.Dataset(eqn_name)
dataset.load()
in_units = dataset.train[0].shape[1]
out_units = dataset.train[1].shape[1]


## DEFINE MODEL PARAMETERS
design = {'flavor':'ff',
          'unit_activations':[(in_units, 'tanh'),
                              (100, 'tanh'),
                              (50, 'tanh'),
                              (out_units, 'linear')
                             ],
          'dropout':[0.1, 0.1, 0.1],
          'optimizer':'adam',
          'loss':'mse',
          'callbacks':['learning_rate', 'tensorboard'],
          'batch_size':25,
          'epochs':100,
         }

set_seed(23)

# TRAIN MODEL
model = encoder.Encoder(design, dataset)
model.train()
model.predict_plot()

## PRINT DIAGNOSTICS:
#model.print_errors(verbose=True)
