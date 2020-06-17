"""
Performs bootstrapping to assess uncertainty
when inferring Poisson's equation theta
Here the test set is fixed, and bootstrapping
is done on the training set.
For each fitted model we predict on the test set,
then plot predicted vs true with error bounds.
"""

import numpy as np
from .plotter import add_noise

# in the bag sampling
def sample(train_data, size=0.6):
    inputs, outputs = train_data
    num_inputs = len(inputs)
    train_size = int(size * num_inputs)
    train_idx = np.random.choice(num_inputs, train_size) 
    train_samp = (train_data[0][train_idx], train_data[1][train_idx])

    return train_samp

def fit_eval_predict(data, fit, evaluate, predict):
    """
    data is a tuple containing train and test boot samples
    fit is a function that trains the model given train samples
    evaluate is a function that evaluates the trained model on test samples
    predict is a function that predicts with the trained model on test samples
    """
    train, test = data 
    fit(train)
    test_loss = evaluate(test) 
    test_predictions = predict(test)
    return test_loss, test_predictions 

def bootstrap(boots, data, fit, evaluate, predict, size=0.6, verbose=False):
    """
    for each boot: sample data, fit, evaluate, predict with model
    the input data is a tuple (train, test) where train and test
    are tuples (input, output)
    """
    train, test = data  

    evals = np.zeros((boots,))
    preds = dict()
    for b in range(boots):
        train_boot = sample(train, size)
        if verbose:
            samp_size = train_boot[0].shape[0]
            message = 'iteration %d out of %d' % (b+1, boots)
            message += ' with %d in the bag samples' % samp_size
            print(message)
        boot_data = (train_boot, test)
        boot_eval, boot_pred = fit_eval_predict(boot_data, 
                fit, 
                evaluate, 
                predict)
        evals[b] = boot_eval
        preds[b] = boot_pred
    return evals, preds
