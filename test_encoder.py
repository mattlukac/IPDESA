from supervise import equation, encoder
import argparse

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
          'unit_activations':[(in_units, 'linear'),
                              (10, 'relu'),
                              (10, 'relu'),
                              (out_units, 'sigmoid')
                             ],
          'dropout':[0.1, 0.1, 0.1],
          'optimizer':'adam',
          'loss':'mse',
          'callbacks':['learning_rate', 'tensorboard'],
          'batch_size':25,
          'epochs':100,
         }

# TRAIN MODEL
model = encoder.Encoder(design, dataset)
model.train()

## PRINT DIAGNOSTICS:
model.print_errors(verbose=True)
