import numpy as np
import os 
import matplotlib.pyplot as plt
from Poisson.plotter import subplot_theta_fit

# torch imports
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import StepLR, CyclicLR
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

# fenics imports 
from fenics import *
from fenics_adjoint import *


class PoissonSolver:
    """ Solves the 1D Poisson equation """

    def __init__(self):
        # creates mesh, function space, and LHS weak form
        self.mesh = UnitIntervalMesh(99)
        self.V = FunctionSpace(self.mesh, 'P', 1)
        self.dofs = self.V.dofmap().dofs(self.mesh, 0)
        u = TrialFunction(self.V)
        self.v = TestFunction(self.V)
        self.a = inner(grad(u), grad(self.v)) * dx

    def solve(self, theta):
        """ Solves Poisson equation given parameters """

        c, b0, b1 = theta
        c = Constant(c)
        L = c * self.v * dx

        # Define boundary condition
        u_D = Expression('x[0] == 0 ? b0: b1',
                        b0 = b0,
                        b1 = b1,
                        degree = 1)
        bd_vals = Function(self.V)
        bd_vals.assign(project(u_D, self.V))
        bc = DirichletBC(self.V, bd_vals, 'on_boundary')

        # Compute solution
        u = Function(self.V)
        solve(self.a == L, u, bc)
        controls = [Control(c), Control(bd_vals)]
        return u, controls 

    def forward(self, theta_batch):
        """ Batch solutions to Poisson equation """

        batch_solns = dict()
        self.controls = dict()
        self.solns = dict()

        if isinstance(theta_batch, torch.Tensor):
            theta_array = theta_batch.clone().numpy()

        if theta_batch.ndim == 1:
            self.solns[0], self.controls[0] = self.solve(theta_array)
            batch_solns[0] = self.solns[0].compute_vertex_values(self.mesh)
        else:
            batch_size = theta_array.shape[0]
            
            # compute batch solutions
            for idx, theta in enumerate(theta_array):
                u, self.controls[idx] = self.solve(theta)
                self.solns[idx] = u
                batch_solns[idx] = u.compute_vertex_values(self.mesh)

        np_solns = np.array([u for u in batch_solns.values()])
        return torch.tensor(np_solns, dtype=torch.float)

    def backward(self, Phi_batch):
        """ Compute batch gradients dJ/dtheta """

        def solve_adjoint(out, data, controls):
            # Compute L2 loss
            Phi = Function(self.V)
            Phi.vector().set_local(data[self.dofs])
            J = assemble(0.5 * inner(out - Phi, out - Phi) * dx)

            # compute loss gradient
            dJdc, dJdb = compute_gradient(J, controls)
            dJdc = dJdc.values().item()

            # convert grads to scalars
            dJdb = dJdb.compute_vertex_values(self.mesh)
            grads = [dJdc, dJdb[0], dJdb[-1]]
            return grads

        grads = dict()
        if isinstance(Phi_batch, torch.Tensor):
            Phi_batch = Phi_batch.detach().numpy()

        if Phi_batch.ndim == 1:
            batch_size = 1
            Phi_batch = np.expand_dims(Phi_batch, axis=0)
        else:
            batch_size = Phi_batch.shape[0]
        for idx in range(batch_size):
            grads[idx] = solve_adjoint(self.solns[idx], 
                                       Phi_batch[idx], 
                                       self.controls[idx])

        grads = np.array([x for x in grads.values()])
        return grads


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
        out = solver.forward(theta)
        ctx.Phi = Phi
        ctx.solver = solver
        return out

    @staticmethod
    def backward(ctx, dout):
        Phi = ctx.Phi
        solver = ctx.solver
        dtheta = torch.tensor(solver.backward(Phi))
        return dtheta, None, None


class PoissonModel(pl.LightningModule):
    """ Dense network connected to Poisson solver """
    
    def __init__(self, solver):
        super().__init__()
        self.solver = solver
        self.linear1 = torch.nn.Linear(100, 20)
        self.linear2 = torch.nn.Linear(20, 3)
        self.fenics_solver = TorchFenicsSolver()
        self.example_input_array = torch.zeros(1, 100)
        self.batch_loss = None # used in progress bar
        self.train_history = dict()
        self.valid_history = dict()
        
    def forward(self, inputs):
        inputs = inputs[0] if len(inputs) == 1 else inputs
        x = self.linear1(inputs)
        self.theta = self.linear2(x)
        u = self.fenics_solver.apply(self.theta, inputs, self.solver)
        return u
    
    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=0.01)
        scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
        return [optimizer], [scheduler]
    
    def training_step(self, batch, batch_idx):
        u = self(batch)
        Phi, = batch
        loss = torch.nn.MSELoss()(u, Phi)
        self.batch_loss = loss.detach().item()
        result = pl.TrainResult(minimize=loss)
        result.log('train_loss', loss)
        return result

    def training_epoch_end(self, outputs):
        """ Record training loss history """
        loss = outputs['train_loss'].detach().mean()
        self.train_history[self.current_epoch] = loss
        return outputs

   # def validation_step(self, batch, batch_idx):
   #     u = self(batch)
   #     Phi, = batch
   #     loss = torch.nn.MSELoss()(u, Phi)
   #     return loss

   # def validation_epoch_end(self, outputs):
   #     """ Record validation loss history """
   #     print(outputs)
   #     loss = torch.tensor(outputs).mean()
   #     self.valid_history[self.current_epoch] = loss
   #     return outputs

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

    def get_progress_bar_dict(self):
        """ This prints a more accurate loss in the progress bar """
        tqdm_dict = super().get_progress_bar_dict()
        tqdm_dict['loss'] = self.batch_loss
        return tqdm_dict

    def get_theta(self, Phi):
        if not isinstance(Phi, torch.Tensor):
            Phi = torch.tensor(Phi, dtype=torch.float)
        activation = {}
        def get_activation(name):
            def hook(self, inputs, outputs):
                activation[name] = outputs.detach()
            return hook

        self.linear2.register_forward_hook(get_activation('linear2'))
        out = self(Phi)
        return activation['linear2']


class ResetTape(pl.Callback):
    """ Resets gradient tape for fenics adjoint to prevent memory leak """
    def on_batch_end(self, trainer, pl_module):
        tape = get_working_tape()
        tape.__init__()


def fit(train_data, valid_data, max_epochs=40):
    """ Creates solver, model, trainer, data loader, then fits the model """
    if not isinstance(train_data, torch.Tensor):
        train_data = torch.tensor(train_data, dtype=torch.float)
        train_data = TensorDataset(train_data)
    if not isinstance(valid_data, torch.Tensor):
        valid_data = torch.tensor(valid_data, dtype=torch.float)
        valid_data = TensorDataset(valid_data)

    solver = PoissonSolver()
    model = PoissonModel(solver)

    tb_logger = pl.loggers.TensorBoardLogger('lightning_logs/')
    print(tb_logger.version)
    checkpoint_filepath = 'lightning_logs/{epoch:02d}|{val_loss:.2e}'
    checkpointer = ModelCheckpoint(filepath=checkpoint_filepath, verbose=True)
    trainer = pl.Trainer(max_epochs=max_epochs, 
                         callbacks=[ResetTape()],
                         checkpoint_callback=checkpointer)
                        # logger=tb_logger)

    train_loader = DataLoader(train_data, batch_size=25, num_workers=8)
    valid_loader = DataLoader(valid_data, batch_size=25, num_workers=8)
    trainer.fit(model, train_loader, valid_loader)

    # move checkpoint file
    vnum = trainer.logger.version
    model.logdir = f'lightning_logs/version_{vnum}/'
    os.system('mv lightning_logs/*.ckpt ' + model.logdir)

    return model

def plot_loss(model, verbose=True):
    train = [x.item() for x in model.train_history.values()]
    valid = [x.item() for x in model.valid_history.values()]
    epochs = [i for i in range(len(train))]

    fig, ax = plt.subplots(figsize=(15,10))
    ax.plot(epochs, train, label='training')
    ax.plot(epochs, valid, label='validation')
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')
    fig.legend()

    logdir = model.logdir
    filename = logdir + 'loss.png'
    plt.savefig(filename)
    if verbose:
        plt.show()
        plt.close()

def predict_theta(model, Phi):
    theta_hat = model.get_theta(Phi).detach().numpy()
    return theta_hat

def plot_theta(model, Phi, theta, verbose=True):
    theta_hat = predict_theta(model, Phi)
    fig, ax = plt.subplots(1,3, sharey=True, figsize=(20,10))
    subplot_theta_fit(fig, ax, theta, theta=theta_hat)

    # compute mse for titles
    theta_mse = np.mean((theta_hat - theta)**2, axis=0)
    for i in range(3):
        ax[i].set_title(f'mse {theta_mse[i]:.1e}', loc='right', fontsize=25)

    # save image
    logdir = model.logdir
    filename = logdir + 'theta.png'
    plt.savefig(filename)
    if verbose:
        plt.show()
        plt.close()
