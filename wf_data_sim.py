from semantic_autoencoder.solvers import wright_fisher as wf
import numpy as np
import pickle
from fenics import *
from fenics_adjoint import *


set_log_active(False)

# simulation size
size = 2000

# simulation parameters 
nu_lo, nu_hi = 1, 5000
gamma_lo, gamma_hi = -1, 1
nu = np.random.uniform(low=nu_lo, high=nu_hi, size=size)
# invert nu
lo, hi = 1 / nu_hi, 1 / nu_lo
nu_lo, nu_hi = lo, hi
nu = 1 / nu
gamma = np.random.uniform(low=gamma_lo, high=gamma_hi, size=size)
theta = np.stack((nu, gamma), axis=1)
assert theta.shape == (size, 2)

# solver parameters
nx = 99
T = 2
dt = 0.2 
deg = 1

# generate solutions
solns = dict()
solver = wf.WrightFisherOnePop(nx, T, dt, deg)
for i in range(size):
    if i % 10 == 0:
        print(f'simulating {i} out of {size}')
    theta_i = theta[i]
    u_i, _ = solver.solve(theta_i)
    u_t = np.array([u.vector().get_local() for u in solver.u_t.values()])
    solns[i] = u_t

    if (i+1) % 25 == 0:
        tape = get_working_tape()
        tape.__init__()

time = [t for t in solver.u_t.keys()]

# pickle data
#coords = solver.mesh.coordinates().copy()
domain = solver.V.tabulate_dof_coordinates().copy()
idxs = np.argsort(domain, axis=0).flatten()
# domain == coords[idxs]
data = {'domain': domain, 
        'idxs': idxs, 
        'time': time,
        'solns': solns, 
        'theta': theta}
filename = 'data/wf_1pop_' # sim type
filename += f'nx={nx}_T={T}_dt={dt}_deg={deg}_' # mesh parameters
filename += f'nu={nu_lo}-{nu_hi}_gamma={gamma_lo}-{gamma_hi}_' # evol parameters
filename += f'{size}_samples.pkl' # number of samples
with open(filename, 'wb') as f:
    pickle.dump(data, f)
