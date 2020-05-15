# Semantic Autoencoder for PDE models
This is a semantic autoencoder to solve the following inverse problem:
Given (perhaps noisy) observations of some phenomenon which is modeled as 
a solution to a known partial differential equation (PDE) with unknown parameters `Theta`,
can we infer `Theta`?

The  semantic autoencoder merges Tensorflow 2 and FEniCS, a finite element method package.
Tensorflow encodes the observations to some latent space that represents `Theta`,
then FEniCS decodes `Theta` to reproduce the observations according to the PDE model.
Unsupervised training is performed by comparing the observations with those 
produced by the network, thereby training the network to learn `Theta`.
