from supervise import equation, encoder
import argparse

# get the equation name
parser = argparse.ArgumentParser()
parser.add_argument('--train_on', 
        help='name of directory containing the training data')
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
          'loss':'mae',
          'callbacks':['learning_rate', 'tensorboard'],
          'batch_size':15,
          'epochs':20,
         }

## TRAIN MODEL
model = encoder.Encoder(design, dataset)
model.train()

## PRINT DIAGNOSTICS:
model.print_errors()
