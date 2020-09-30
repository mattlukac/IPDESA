import numpy as np
import wright_fisher as wf
import pickle
from fenics import *
from fenics_adjoint import *

# generate parameters 
size = 200
N = np.random.uniform(low=5, high=1000, size=size)
gamma = np.random.uniform(low=-1, high=1, size=size)
theta = np.stack((N, gamma), axis=1)
assert theta.shape == (size, 2)

# generate solutions in batches
batch_size = 25
nx = 19
assert size % batch_size == 0
num_batches = int(size / batch_size)

solns = np.zeros((size, nx+1))
solver = wf.WrightFisherOnePop(nx)
for i in range(num_batches):
    # get batch
    start = i * batch_size
    end = start + batch_size
    theta_batch = theta[start:end]
    assert len(theta_batch) == batch_size

    # get batch solutions
    solns[start:end] = solver.forward(theta_batch)
    
    # reset grad tape
    tape = get_working_tape()
    tape.__init__()

# pickle data
data = (solns, theta)
name = f'wf_onepop_nx={nx}'
pkl = open('../../data/' + name + '.pkl', 'wb')
pickle.dump(data, pkl)
pkl.close()
