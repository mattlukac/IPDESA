# Semantic Autoencoder for PDE models
This is a semantic autoencoder to solve the following inverse problem:
Given (perhaps noisy) observations of some phenomenon which is modeled as 
a solution to a known partial differential equation (PDE) 
with unknown parameters `Theta`, can we infer `Theta`?

The semantic autoencoder merges Tensorflow 2 and FEniCS, 
a finite element method package.
Tensorflow encodes the observations to a latent space that represents `Theta`,
then FEniCS decodes `Theta` to reproduce the observations according to the PDE model.
Unsupervised training is performed by comparing the observations with those 
produced by the network, thereby training the network to learn `Theta`.

## Supervised Encoding Usage
**Requirements:** Tensorflow 2.2 and scikit-learn 0.23.1

The `equations/` directory contains some example config files.
There are four requirements for the config files:
 1. `Theta_names` is a list of strings representing names for each parameter in `Theta`
 2. `domain()` is a function that returns the domain for the PDE
 3. `solution(Theta)` is a function that calculates the analytic solution to the PDE
 as it depends on `Theta`.
 4. `Theta(replicates)` is a function that simulates `Theta` from a prior distribution.

With a config file containing these, such as `equations/poisson.py`,
one simply needs to run

`python train_encoder.py --trainon poisson`

and observe the encoder will be trained.
You will find plots of the predicted vs true `Theta` values in
`logs/fit/{datetime}/` where `datetime` is the date and time 
at which the encoder was trained.

A more detailed example of training using `poisson.py` is
worked through in `poisson_encoder_example.ipynb`.
