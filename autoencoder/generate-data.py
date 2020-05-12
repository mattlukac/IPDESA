## generates training and test data and labels
## for the basic ff network to infer diffusion 
## coefficients D in the one-dimension equation
## \grad\dot( D\grad u ) + f = 0 where
## f(x) = D\sin(\pi D x), thus
## u(x) = \sin(\pi D x)/(\pi D)^2 on \Omega = [-5, 5]
## and D ~ Uniform(0.1, 1) with free boundary conditions


import numpy as np 

# training, validation, test set sizes
sets = ['train','val','test']
N = dict(zip(sets, [900,100,1000]))

# sampled points in the domain
x = np.linspace(-1., 1., num=200)

# numpy arrays for data
def u(x, D):
    y = np.exp(-x**2 / (2*D))/np.sqrt(2 * np.pi * D)
    return(y)

# make and save data and labels
D = N.copy()
us = N.copy()
for k, setSize in N.items():
    D[k] = np.random.uniform(0.1,1, size=setSize)
    us[k] = np.zeros((setSize, len(x)))
    for i in range(setSize):
        us[k][i,:] = u(x,D[k][i])
    np.save('u_'+k, us[k])
    np.save('D_'+k, D[k])
