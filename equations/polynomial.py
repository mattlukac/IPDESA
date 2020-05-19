## generates training and test data and labels
## for the basic ff network to infer gaussian variance
## D in u(x) = 1/sqrt(2pi D) exp(-x^2/(2D))

import numpy as np 

## DOMAIN 
def domain():
    Omega = np.linspace(-1., 1., num=200)
    return Omega

## SOLUTION
def solution(Theta):
    x = domain()
    u = Theta * x**2 
    return(u)

## THETA
def Theta(replicates):
    Theta = np.random.uniform(0.1,1, size=replicates)
    return Theta
