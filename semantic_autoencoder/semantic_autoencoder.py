from .utilities.converter import *
from .solvers.poisson import Poisson
from .solvers.wright_fisher import WrightFisherOnePop
import numpy as np
import glob
import pickle
import os

# to reset gradient tape
from fenics import *
from fenics_adjoint import get_working_tape

# torch imports
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split


class TorchFenicsSolver(torch.autograd.Function):
    """
    PDE solver layer that takes theta as input,
    the forward pass solves the Poisson equation
    and computes L2 loss J as output.

    The backward pass computes dJ/dtheta
    """

    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(ctx, theta, Phi, solver):
        ctx.Phi = Phi
        ctx.solver = solver
        theta_array = theta.clone().numpy()
        out = solver.forward(theta_array)
        return numpy_to_torch(out)

    @staticmethod
    def backward(ctx, dout):
        Phi = torch_to_numpy(ctx.Phi)
        solver = ctx.solver
        dtheta = torch.tensor(solver.backward(Phi))
        return dtheta, None, None


class SemanticAutoEncoder(pl.LightningModule):
    """ Dense network connected to FEniCS solver """
    
    def __init__(self, solver, Phi_dim, theta_dim, debug_mode=False):
        self.debug_mode = debug_mode
        super().__init__()
        self.solver = solver
        self.linear1 = torch.nn.Linear(Phi_dim, 50)
        self.nu_linear = torch.nn.Linear(50, 1)
        self.gamma_linear = torch.nn.Linear(50, 1)
        self.tanh = torch.nn.Tanh()
        self.sigmoid = torch.nn.Sigmoid()
        self.softplus = torch.nn.Softplus()
        self.fenics_solver = TorchFenicsSolver()
        self.example_input_array = torch.zeros(1, Phi_dim)
        self.batch_loss = None # used in progress bar
        self.train_history = dict() # used to plot 
        self.valid_history = dict() # used to plot 
        
    # nu has range (1, 1000) so we rescale nu
    def forward(self, inputs):
        inputs = inputs[0] if isinstance(inputs, list) else inputs
        batch_size = len(inputs)
        x = self.linear1(inputs)
        self.nu = self.softplus(self.nu_linear(x))
#        self.nu = torch.mul(self.sigmoid(self.nu_linear(x)), 1-0.001)
#        self.nu = torch.add(self.nu, 0.001)
        self.gamma = self.tanh(self.gamma_linear(x))
        self.theta = torch.cat((self.nu, self.gamma), dim=1)

        if self.debug_mode: print('theta\n', self.theta)
        u = self.fenics_solver.apply(self.theta, inputs, self.solver)
        return u
    
    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=0.01)
        scheduler = StepLR(optimizer, step_size=10, gamma=1.0)
        return [optimizer], [scheduler]
    
    def training_step(self, batch, batch_idx):
        u = self(batch)
        Phi, = batch
        L1 = torch.nn.L1Loss()
        L2 = torch.nn.MSELoss()
        loss = L1(u, Phi) + L2(u, Phi)
        self.batch_loss = loss.detach().item()
        result = pl.TrainResult(minimize=loss)
        result.log('train_loss', loss)
        return result

    def training_epoch_end(self, outputs):
        """ Record training loss history """
        loss = outputs['train_loss'].detach().mean()
        self.train_history[self.current_epoch] = loss
        return outputs

    def validation_step(self, batch, batch_idx):
        u = self(batch)
        Phi, = batch
        loss = torch.nn.MSELoss()(u, Phi)
        result = pl.EvalResult(checkpoint_on=loss)
        result.log('val_loss', loss)
        return result

    def validation_epoch_end(self, outputs):
        """ Record validation loss history """
        loss = outputs['val_loss'].detach().mean()
        self.valid_history[self.current_epoch] = loss
        outputs['val_loss'] = loss
        return outputs

    def get_theta(self, Phi):
        if isinstance(Phi, np.ndarray):
            Phi = numpy_to_torch(Phi)
        activation = {}
        def get_activation(name):
            def hook(self, inputs, outputs):
                activation[name] = outputs.detach()
            return hook

        self.linear2.register_forward_hook(get_activation('linear2'))
        out = self(Phi)
        return activation['linear2']

#    def get_progress_bar_dict(self):
#        """ This prints a more accurate loss in the progress bar """
#        tqdm_dict = super().get_progress_bar_dict()
#        tqdm_dict['loss'] = self.batch_loss
#        return tqdm_dict

class ResetTape(pl.Callback):
    """ Resets gradient tape for fenics adjoint to prevent memory leak """
    def on_batch_end(self, trainer, pl_module):
        tape = get_working_tape()
        tape.__init__()


# train network
def fit(train_data, valid_data, solver, max_epochs=40, debug_mode=False):
    """ Creates solver, model, trainer, data loader, then fits the model """
    if not isinstance(train_data, torch.Tensor):
        train_data = torch.tensor(train_data, dtype=torch.float)
        train_data = TensorDataset(train_data)
    if not isinstance(valid_data, torch.Tensor):
        valid_data = torch.tensor(valid_data, dtype=torch.float)
        valid_data = TensorDataset(valid_data)

    model = SemanticAutoEncoder(solver, Phi_dim=100, theta_dim=2, debug_mode=debug_mode)

    tb_logger = pl.loggers.TensorBoardLogger('lightning_logs/')
    ckpt_filepath = 'lightning_logs/{epoch:02d}|{val_loss:.2e}'
    checkpointer = ModelCheckpoint(filepath=ckpt_filepath, verbose=True)
    trainer = pl.Trainer(max_epochs=max_epochs,
                         callbacks=[ResetTape()],
                         checkpoint_callback=checkpointer)

    train_loader = DataLoader(train_data, batch_size=25, num_workers=8)
    valid_loader = DataLoader(valid_data, batch_size=25, num_workers=8)
    trainer.fit(model, train_loader, valid_loader)

    # move checkpoint file
    vnum = trainer.logger.version
    model.logdir = f'lightning_logs/version_{vnum}/'
    os.system('mv lightning_logs/*.ckpt ' + model.logdir)

    return model

# train with Poisson data
def train_Poisson():
    from Poisson import equation
    dataset = equation.Dataset('poisson')
    dataset.load()
    Phi_train, theta_train = dataset.train
    Phi_val, theta_val = dataset.validate
    solver = Poisson()
    fit(Phi_train, Phi_val, solver, max_epochs=5)

# train with WF1P data
def train_WF1P(debug_mode=False):
    wf_file = glob.glob('data/wf*')[0]
    with open(wf_file, 'rb') as f:
        data = pickle.load(f)
    # data is a dictionary, solns have key 'solns' which is a dictionary
    # with simulation indexes as keys, 
    # and values are arrays w/shape (timesteps, soln_dim)
    # the domain and solns are likely not in the correct ordering, 
    # so use u_t[data['idxs']] to correct this 
    solns = data['solns']
    theta = data['theta']

    # Phi is terminal soln
    sample_idxs, u_t = zip(*solns.items())
    Phi = np.zeros((len(sample_idxs), u_t[0].shape[1]))
    for i in range(len(Phi)):
        Phi[i] = u_t[i][-1]
    Phi_train, Phi_val, theta_train, theta_val = train_test_split(Phi, theta)

    if debug_mode:
        Phi_train = Phi_train[0:500]
        Phi_val = Phi_val[0:500]
    solver = WrightFisherOnePop(nx=99, T=2, dt=0.2, deg=1, debug_mode=debug_mode)
    model = fit(Phi_train, Phi_val, solver, max_epochs=40, debug_mode=debug_mode)
    return model

def plot_theta():
    # use poisson plotter module
    pass


if __name__ == '__main__':
    from utilities import poisson
    import warnings
    import os
    warnings.filterwarnings('ignore')
    set_log_active(False)

    train_WF1P()
