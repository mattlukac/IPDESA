import numpy as np
from .tools import *
from copy import deepcopy
import imageio
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch

# global pyplot settings
plt.style.use(['seaborn-bright'])
plt.rcParams.update({'font.size' : 26,
                     'figure.dpi' : 200,
                     'legend.shadow' : True,
                     'legend.handlelength' : 2})


####################
# HELPER FUNCTIONS #
####################

def identity(ax):
    """ Plot the identity map y=x """
    x = np.array(ax.get_xlim())
    y = x 
    ax.plot(x, y, c='r', lw=3, alpha=0.5)

def suptitle(fig, title, pad=0):
    """ Adds title above subplots """
    fig.add_subplot(111, frame_on=False)
    plt.tick_params(labelcolor='none', bottom=False, left=False)
    plt.title(title, pad=pad)

def show():
    """ 
    Plot methods don't show by default, 
    so call this in network plot methods 
    """
    plt.show()
    plt.close()

############
# SAVE MP4 #
############

def save_frame(frame_num, frame_path, frame_plot):
    """ Used at the end of any method that plots a frame to be saved """
    # frame plot
    frame_plot()
    plt.savefig(frame_path + str(frame_num) + '.png')
    plt.close()

def save_mp4(frame_saver, settings):
    """
    Creates mp4 from collection of frames.
      frame_saver - function that saves frame to frame_path
      settings - dictionary containing:
        mp4_path - path (including file name) to the mp4
        frame_path - path to directory containing frames
        num_frames - number of frames to make
        duration - length of video (in seconds)
    """
    # unpack dictionary
    mp4_path = settings['mp4_path']
    frame_path = settings['frame_path']
    num_frames = settings['num_frames']
    duration = settings['duration']

    # make video
    fps = int(num_frames / duration)
    with imageio.get_writer(mp4_path, mode='I', fps=fps) as writer:
        for frame in range(num_frames):
            frame_saver(frame)
            writer.append_data(
                    imageio.imread(
                        frame_path + str(frame) + '.png'
                        )
                    )

#############
# SOLUTIONS #
#############

def solution(domain, solution, theta):
    """
    Plots solution(s) given domain, solution and theta
    Solution and theta should have shapes (num_plots, soln_shape)
    and (num_plots, theta_shape) respectively.
    """
    num_plots = len(solution)
    commas = [', ', ', ', r'$)$']
    fig, ax = plt.subplots(1, num_plots, figsize=(20,10))
    for i in range(num_plots):
        title = r'$\theta = ($'
        for j in range(theta.shape[1]):
            #title += '%s' % name
            title += r'$%.2f$' % theta[i,j] 
            title += commas[j]
        ax[i].plot(domain, solution[i], linewidth=3)
        ax[i].set_xlabel(r'$x$', fontsize=30)
        ax[i].set_xticks([0,1])
        ax[i].set_title(title, fontsize=28)
    ax[0].set_ylabel(r'$u_\theta = u(x)$', fontsize=30)

#############
# MODEL FIT # 
#############

def theta_fit(Phi, theta_Phi, theta, 
        sigma=0, 
        transform=True, 
        verbose=True):
    """
    General plotting function to plot latent space theta vs theta_Phi,
    Inputs:
        Phi - the test set data
        theta_Phi - the test set theta
        theta - latent theta values
        sigma - standard deviation for normally distributed noise
        transorm - plot theta components on common scale
        verbose - prints mean square errors
    """
    if verbose:
        # evaluate, predict, and plot with trained model
        theta_mse = np.mean((theta - theta_Phi) ** 2, axis=0)
        print('theta MSE:', theta_mse)

    # initialize axes
    plot_cols = 1
    theta_len = theta_Phi.shape[1]
    fig, ax = plt.subplots(theta_len, 1,
            sharey=transform,
            sharex=transform,
            figsize=(20,20))

    # plot transformed thetas
    if transform:
        theta_Phi, theta = rescale_thetas(theta_Phi, theta)
        
    if verbose:
        # evaluate, predict, and plot with trained model
        theta_tform_mse = np.mean((theta - theta_Phi) ** 2, axis=0)
        print('transformed theta MSE:', theta_tform_mse)

    subplot_theta_fit(fig, ax, theta_Phi, theta)

def subplot_theta_fit(fig, ax, theta_Phi, theta=None, resids=None):
    """
    Plots theta vs theta_Phi
    Agnostic to data transformation
    """
    theta_names = ['$c$', '$b_0$', '$b_1$']
    num_plots = len(ax)

    # scientific notation for residuals
    formatter = mticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((0,0))

    # compute residuals if not given
    if resids is None:
        resids = theta - theta_Phi
    ymin = np.min(resids) * 1.25
    ymax = np.max(resids) * 1.25
    for i in range(num_plots):
        residuals = resids[:,i]
        xmin = 1.1 * np.min(theta_Phi[:,i])
        xmax = 1.1 * np.max(theta_Phi[:,i])
        ax[i].set_ylim(ymin, ymax)
        ax[i].scatter(theta_Phi[:,i], residuals,
                alpha=0.7, label=theta_names[i] + ' predictions')
        ax[i].plot([xmin, xmax], [0,0], lw=3, c='k', ls='dashed')
        ax[i].yaxis.set_major_formatter(formatter)
        ax[i].legend(loc='upper left', fontsize=20)
    suptitle(fig, '')
    plt.xlabel('Truth', fontsize=26)
    plt.ylabel('Residuals', fontsize=26, labelpad=40.0)

def solution_fit(Phi, noise, theta_Phi, theta, u_theta, 
        sigma=0, 
        seed=23, 
        ylims=None,
        verbose=True):
    """
    General plotting function to plot Phi, Phi+noise (for nonzero sigma),
    and the reconstructed Phi u_theta
    Inputs:
        Phi - the test set data
        theta_Phi - the test set theta
        theta_from_Phi - function that gets latent theta from Phi
        u_from_Phi - function that reconstructs Phi given Phi
        sigma - standard deviation for normally distributed noise
        seed - random seed for reprodicibility
        ylims - ylim for each subplot
        verbose - print mean square errors
    """
    domain = np.linspace(0, 1, Phi.shape[1])
    theta_names = ['$c$', '$b_0$', '$b_1$']

    if verbose:
        # compute mean square errors
        theta_mse = np.mean((theta - theta_Phi) ** 2, axis=0)
        Phi_mse = np.mean((u_theta - Phi) ** 2)
        print('Latent theta MSE:', theta_mse)
        print('Reconstructed Phi MSE:', Phi_mse)

    num_plots = len(Phi)
    idx = np.array([x for x in range(num_plots)]).reshape(3, 3)
    fig, ax = plt.subplots(nrows=3, ncols=3,
                           sharex=True,
                           figsize=(20,20))
    labs = [r'$u_\theta$', r'$\Phi$', r'$\Phi +$noise']
    for i in np.ndindex(3, 3):
        # plot reconstructed Phi
        ax[i].plot(domain, u_theta[idx[i]], label=labs[0], lw=3)
        # plot noiseless Phi
        ax[i].plot(domain, Phi[idx[i]] - noise[idx[i]],
                label=labs[1],
                lw=3,
                ls='dashed',
                c='k')
        # plot noisy Phi
        if sigma != 0: 
            ax[i].plot(domain, Phi[idx[i]],
                    label=labs[2],
                    lw=3,
                    alpha=0.2,
                    c='k')
        if i[0] == 2:
            ax[i].set_xlabel(r'$x$', fontsize=26)
        ax[i].set_xticks([0, 1])
        if ylims is not None:
            ax[i].set_ylim(ylims[idx[i]])
        labs = ['' for lab in labs] # don't label other plots
    cols = 3 if sigma != 0 else 2
    fig.legend(fontsize=26, loc='upper center', ncol=cols)

    suptitle(fig, 'Test set sample', pad=20)

#################
# BOOTSTRAPPING #
#################

def theta_boot(test, boot_data, sigmas, se='bootstrap', verbose=False):
    """ Plot bootstrap results in parameter space, resids vs truth """
    # compute bootstrap means and confidence bounds
    test_theta = deepcopy(test[1]) # ground truth test theta
    stats = boot_stats(boot_data, test_theta)
    theta_Phi = stats['rescaled_true_theta']
    param = ['$c$', '$b_0$', '$b_1$']
    if verbose:
        print('%d intervals' % len(test_theta))

    # errorbar plots
    fig, ax = plt.subplots(3,1, figsize=(20,20), sharex=True, sharey=True)
    for i in range(3):
        # plot bootstrap means
        resids_means = stats['means'][:,i]

        # errors
        if se == 'bootstrap':
            errs = 2. * stats['SE'][:,i]
            upper = resids_means + errs 
            lower = resids_means - errs
        elif se == 'quantile':
            errs = stats['credint95'][i]
            upper = stats['upper95'][:,i]
            lower = stats['lower95'][:,i]

        # count intervals containing zero
        below = (upper < 0)
        above = (lower > 0)
        outside = below + above
        num_outside = np.sum(outside, axis=0)
        percent_outside = 100. * num_outside / len(test_theta)
        if verbose:
            msg = '%d ' % num_outside
            msg += r'' + param[i] + ' intervals (%.2f%%) ' % percent_outside
            msg += 'do not contain 0'
            print(msg)
        colors = ['r' if out else 'C0' for out in outside]
        
        ax[i].scatter(theta_Phi[:,i], resids_means, 
                label=param[i] + ' boot means', c=colors, alpha=0.7)
        # plot bootstrap credible regions
        ax[i].errorbar(theta_Phi[:,i], resids_means, 
                yerr=errs,
                ecolor=colors,
                alpha=0.25,
                fmt='|',
                label='95% credible region')
        # y=0
        xmin = np.min(theta_Phi[:,i])
        xmax = np.max(theta_Phi[:,i])
        ax[i].plot([xmin, xmax], [0,0], 
                c='k', 
                lw=3, 
                ls='dashed')
        ax[i].set_ylabel('Residuals')
        ax[i].legend(loc='upper left', fontsize=20)
    ax[2].set_xlabel('Truth')
    train_sigma, test_sigma = sigmas
    title = noise_title(train_sigma, test_sigma)
    suptitle(fig, title, pad=20)

def solution_boot(test, boot_data, u_from_theta, sigmas, se='bootstrap'):
    """
    Plot bootstrap results for a sample of 9 solutions
    For each Phi in the sample, plots:
      denoised and noisy Phi
      bootstrap mean
      credible regions
    Credible regions are quantiles from theta bootstrap results
    """
    # make title and add noise to Phi
    train_sigma, test_sigma = sigmas 
    title = noise_title(train_sigma, test_sigma)
    Phi, theta_Phi = deepcopy(test)
    Phi, noise = add_noise(Phi, test_sigma) # noisy test inputs

    # get bootstrap statistics
    num_boots = len(boot_data)
    stats = boot_stats(boot_data)

    # generate sample
    num_plots = 9
    sample_idx = np.random.randint(0, len(Phi)-1, num_plots)
    Phi = Phi[sample_idx]
    noise = noise[sample_idx]
    theta_means = stats['means'][sample_idx]
    means = u_from_theta(theta_means)
    if se == 'bootstrap':
        up95 = theta_means + 2. * stats['SE_boot'][sample_idx]
        lo95 = theta_means - 2. * stats['SE_boot'][sample_idx]
        upper95 = u_from_theta(up95)
        lower95 = u_from_theta(lo95)
        up68 = theta_means + stats['SE_boot'][sample_idx]
        lo68 = theta_means - stats['SE_boot'][sample_idx]
        upper68 = u_from_theta(up68)
        lower68 = u_from_theta(lo68)
    else:
        upper95 = u_from_theta(stats['upper95'][sample_idx])
        lower95 = u_from_theta(stats['lower95'][sample_idx])
        upper68 = u_from_theta(stats['upper68'][sample_idx])
        lower68 = u_from_theta(stats['lower68'][sample_idx])

    # for indexing plot grid
    idx = np.array([x for x in range(num_plots)]).reshape(3, 3)

    fig, ax = plt.subplots(nrows=3, ncols=3,
                           sharex=True,
                           figsize=(20,20))
    labs = [r'$u_\theta$ boot mean', 
            '95% credible region', 
            '68% credible region',
            r'$\Phi$', 
            r'$\Phi +$noise']
    domain = np.linspace(0, 1, Phi.shape[1])

    for i in np.ndindex(3, 3):
        # u
        ax[i].plot(domain, means[idx[i]], label=labs[0], lw=3)

        # credible intervals
        ax[i].fill_between(domain, lower95[idx[i]], upper95[idx[i]], 
                color='C0', 
                alpha=0.12,
                label=labs[1])
        ax[i].fill_between(domain, lower68[idx[i]], upper68[idx[i]], 
                color='C0', 
                alpha=0.25,
                label=labs[2])

        # denoised Phi
        ax[i].plot(domain, Phi[idx[i]] - noise[idx[i]],
                label=labs[3],
                lw=3,
                ls='dashed',
                alpha=1.0,
                c='k')
        # noisy Phi
        if test_sigma != 0: 
            ax[i].plot(domain, Phi[idx[i]],
                    label=labs[4],
                    lw=3,
                    alpha=0.2,
                    c='k')

        # label x axes
        if i[0] == 2:
            ax[i].set_xlabel(r'$x$', fontsize=26)
        ax[i].set_xticks([0, 1])
        labs = ['' for lab in labs] # don't label other plots

    # title
    suptitle(fig, title, pad=20)
    
    # legend in proper order
    handles, labels = ax[0,0].get_legend_handles_labels()
    if test_sigma != 0:
        new_order = [0,3,4,1,2]
    else:
        new_order = [0,2,3,1]
    handles = [handles[i] for i in new_order]
    labels = [labels[i] for i in new_order]
    fig.legend(handles, labels, 
            fontsize=26, 
            loc='upper center', 
            ncol = len(new_order))


############################
# ADJOINT GRADIENT DESCENT #
############################

class AdjointPlotter:
    """
    Plotter class for adjoint equation gradient descent.
    Last line of AdjClass.descend() should be
      self.plt = plotter.AdjointDescent(AdjClass)
    """

    def __init__(self, adjoint):
        # initialized with adjoint eq grad descent class
        self.adjoint = adjoint
        self.theta_names = [r'$c$', r'$b_0$', r'$b_1$']
        self.theta_Phi_names = [r'$\kappa$', r'$\beta_0$', r'$\beta_1$']


    ####################
    # BOUNDARY CONTROL #
    ####################

    def grad_descent_bc(self):
        """ Plots boundary control gradient descent results """
        # get parameters
        c, kappa = (self.adjoint.thetas[:,0], self.adjoint.theta_true[0])
        b0, beta0 = (self.adjoint.thetas[:,1], self.adjoint.theta_true[1])
        b1, beta1 = (self.adjoint.thetas[:,2], self.adjoint.theta_true[2])

        # 2x2 grid (b0, b1, \\ c, u)
        fig, ax = plt.subplots(2, 2, figsize=(18,15))

        # plot c, b0, b1
        self.subplot_theta_descent(ax[1,0], c, 0)
        self.subplot_theta_descent(ax[0,0], b0, 1)
        self.subplot_theta_descent(ax[0,1], b1, 2)

        # plot solutions
        self.subplot_solution_descent(ax[1,1])

        # make title
        title = r'%d iterations ' % max(self.adjoint.iterations)
        title += 'at learning rate $\gamma = %.1f$' % self.adjoint.lr
        suptitle(fig, title)

        plt.show()
        plt.close()

    def subplot_theta_descent(self, ax, theta, idx):
        """
        Plots parameter convergence during gradient descent
        Inputs:
            ax - axis object for plotting
            param - parameter to plot vs iterations
            idx - index for true parameter (0:kappa, 1:beta0, 2:beta1)
        """
        theta_Phi = self.adjoint.theta_true[idx]
        iterations = self.adjoint.iterations

        # plot theta converging to theta_Phi
        ax.plot(iterations, theta, color='g', lw=3)
        ax.hlines(theta_Phi, 
                xmin=0, 
                xmax=max(iterations), 
                lw=3, 
                ls='dashed', 
                color='k')

        ax.set_xlabel('iterations')
        ax.legend([self.theta_names[idx], self.theta_Phi_names[idx]])

    def subplot_solution_descent(self, ax, title=''):
        """ Plots gradient descent in solution space """

        # plot solution convergence
        u_lab = r'$u$'
        for i, u in enumerate(self.adjoint.solutions):
            alpha_i = 1./(i+1)
            ax.plot(self.adjoint.x, u, 
                    label=u_lab, 
                    lw=3, 
                    c='C0', 
                    alpha=alpha_i)
            u_lab = ''

        # plot Phi without noise
        Phi = self.adjoint.Phi - self.adjoint.noise
        ax.plot(self.adjoint.x, Phi, 
                label=r'$\Phi$',
                lw=3, 
                c='k',
                ls='dashed')

        # plot Phi with noise if exists
        if self.adjoint.sigma != 0:
            ax.plot(self.adjoint.x, self.adjoint.Phi, 
                    label=r'$\Phi + $noise',
                    lw=3, 
                    c='0.5', 
                    ls='dashed')
        
        ax.set_xlabel(r'$\Omega$')
        ax.set_title(title)
        ax.legend()

    def curve_convergence(self):
        """ Plots descent in solution space """
        fig, ax = plt.subplots(1, 1, figsize=(20,15)) 

        title = r'%d iterations ' % max(self.adjoint.iterations)
        title += 'at learning rate $\gamma = %.1f$' % self.adjoint.lr
        self.subplot_solution_descent(ax, title)
        ax.legend(loc='upper center', ncol=2)

        plt.show()
        plt.close()

    def biplot_descent_bc(self):
        """ Plots boundary control descent in parameter space """
        fig, ax = plt.subplots(self.adjoint.theta_dim, 1, figsize=(18,18))

        for i in range(self.adjoint.theta_dim):
            j = (i + 1) % 3
            thetas = [self.adjoint.thetas[:,i], self.adjoint.thetas[:,j]]
            idxs = [i, j]
            self.biplot_theta(ax[i], thetas, idxs)

        title = '%d iterations to convergence, ' % max(self.adjoint.iterations)
        title += r'$\gamma = %.1f$' % self.adjoint.lr
        suptitle(fig, title)

        plt.show()
        plt.close()

    def biplot_theta(self, ax, thetas, idxs, title='', color='g'):
        """
        Plots gradient descent in parameter space with param_y vs param_x
        Inputs:
            ax - the axis plotting object
            thetas - list or tuple of parameter data [param_x, param_y]
            idxs - list or tuple of parameter indices from theta_true
                   (0:kappa, 1:beta0, 2:beta1)
        """
        # make names and get true parameter
        theta_x, theta_y = thetas
        x, y = idxs

        # make ordered pairs for legend
        ordered_pair = lambda x, y : '(' + x + ', ' + y + ')'
        curve = ordered_pair(self.theta_names[x], self.theta_names[y])
        optim = ordered_pair(self.theta_Phi_names[x], self.theta_Phi_names[y])

        # make the plot
        ax.plot(theta_x, theta_y, 
                label=curve, 
                lw=3, 
                c=color)
        ax.scatter(self.adjoint.theta_true[x], self.adjoint.theta_true[y], 
                   label=optim, 
                   s=12 ** 2, 
                   c='k')
        ax.legend()
        ax.set_title(title)

    def learning_rates(self, data, max_its):
        """
        Plots number of iterations to convergence
        as a function of learning rate given by lr_bounds
        """
        # get plotting data
        lr, counts = zip(*data.items())
        lr_opt = lr[np.argmin(counts)]

        # plot
        fig, ax = plt.subplots(1, 1, figsize=(18,10))
        ax.plot(lr, counts, linewidth=5, color='C1')
        ax.set_xlabel('Learning Rate ' + r'$\gamma$')
        ax.set_ylabel('iterations to convergence (%d max)' % max_its)
        ax.set_title(r'%d iterations at $\gamma = %.2f$' 
                     % (np.min(counts), lr_opt))
        plt.show()
        plt.close()


    #########################
    # LEFT BOUNDARY CONTROL #
    #########################

    def grad_descent_lbc(self, c, kappa, b0, beta0):
        """ Plots left boundary control gradient descent """
        fig, ax = plt.subplots(2, 2, figsize=(15, 15), dpi=200)
        fig.subplots_adjust(hspace=0.25)

        # c, b0 convergence top row
        self.subplot_theta_descent(ax[0,0], c, 0)
        self.subplot_theta_descent(ax[0,1], b0, 1)
        
        # solution convergence bottom row
        for ax in ax[1,:]: ax.remove()
        gs = GridSpec(2,2)
        soln_ax = fig.add_subplot(gs[1,:])
        self.subplot_solution_descent(soln_ax)

        # make title
        title = r'%d iterations ' % max(self.adjoint.iterations)
        title += 'at learning rate $\gamma = %.1f$' % self.adjoint.lr
        suptitle(fig, title)

        plt.show()
        plt.close()

    def biplot_descent_lbc(self):
        """ Theta biplot of left boundary control grad descent """
        fig, ax = plt.subplots(1, 1, figsize=(18,15))

        # make plot
        thetas = [self.adjoint.thetas[:,0], self.adjoint.thetas[:,1]]
        idxs = [0, 1]
        self.biplot_theta(ax, thetas, idxs)

        # make title
        title = '%d iterations to convergence, ' % max(self.adjoint.iterations)
        title += r'$\gamma = %.1f$' % self.adjoint.lr
        ax.set_title(title)

        plt.show()
        plt.close()

    def loss_contour(self, c, b0, log_losses):
        """
        Plots the -log(loss) contour as a function of c and b0
        with the gradient descent path overlayed on top
        """
        c_min, c_max, b0_min, b0_max = self.adjoint.ranges
        lines = np.linspace(np.min(log_losses), np.max(log_losses), 20)
        grad = self.adjoint.get_grad([c[-1], b0[-1], self.adjoint.b1])

        fig, ax = plt.subplots(1, 1, figsize=(18,10))

        # make contour plot
        cont = ax.contourf(self.adjoint.c_tensor, 
                self.adjoint.b0_tensor, 
                log_losses, 
                lines)
        cbar = fig.colorbar(cont)
        cbar.set_label(r'$-\log(\mathcal{J})$', rotation=90)

        # make theta biplot
        title = r'$\frac{d\mathcal{J}}{d\theta} = '
        title += '(%.7f, %.7f)$' % (grad[0,0], grad[0,1])
        self.biplot_theta(ax, [c, b0], [0, 1], title, color='r')

        # make optimum point
        ax.scatter(c[-1], b0[-1], c='r', s=10**2)

        ax.set_xlim([c_min, c_max])
        ax.set_ylim([b0_min, b0_max])
        ax.set_xlabel(r'$c$')
        ax.set_ylabel(r'$b_0$')

    #######################
    # NO BOUNDARY CONTROL #
    #######################

    def grad_descent(self, c, kappa):
        """ Plots no boundary control grad descent results """
        fig, ax = plt.subplots(2, 1, figsize=(15, 15))
        fig.subplots_adjust(hspace=0.25)
        self.subplot_theta_descent(ax[0], c, 0)
        self.subplot_solution_descent(ax[1])

        # make title
        title = r'%d iterations ' % max(self.adjoint.iterations)
        title += 'at learning rate $\gamma = %.1f$' % self.adjoint.lr
        suptitle(fig, title)

        plt.show()
        plt.close()

    def loss(self, c_init, c_range, grad_line, Jc, dJdc):
        """ Plots loss J as a function of force parameter c """
        fig, ax = plt.subplots(1, 1, figsize=(18, 15), dpi=200)

        # plot tangent line
        ax.plot(c_range, grad_line,
                label=r'$gradient -\gamma \frac{d\mathcal{J}}{dc}$',
                linewidth=5, 
                c='r',
                alpha=0.5)

        # plot (c, J(c))
        ax.scatter(c_init, Jc, 
                   label=r'descending parameter $(c, \mathcal{J}(c))$',
                   s=12**2, 
                   c='g',
                   zorder=2.5)

        # plot gradient vector
        dc = -self.adjoint.lr * dJdc 
        ax.arrow(x=c_init, y=Jc, dx=dc, dy=0, 
                #length_includes_head=True,
                head_length=0.05,
                color='r',
                alpha=0.5)

        # compute and plot loss
        loss = np.zeros(c_range.shape)
        for i, c in enumerate(c_range):
            theta = [c, self.adjoint.b0, self.adjoint.b1]
            loss[i] = self.adjoint.get_loss(theta)
        loss_plot = ax.plot(c_range, loss, 
                label=r'loss $\mathcal{J}(c)$',
                linewidth=5, 
                c='mediumblue')

        # set labels and title
        title = r'$\frac{d\mathcal{J}}{dc} = %.7f$' % dJdc
        title += r', $\gamma = %.2f$' % self.adjoint.lr
        ax.set_title(title)
        ax.set_xlabel(r'force parameter $c$')
        ax.set_ylabel(r'Loss Functional $\mathcal{J}$')
        ax.set_ylim(bottom=-0.01)
        ax.legend(loc='lower left')
        # make kappa a tick label
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(
            lambda x, pos : r'$\kappa$' if x == self.adjoint.kappa else x))

