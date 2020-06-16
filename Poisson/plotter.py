import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from sklearn import preprocessing
plt.rcParams['font.size'] = 26
hfont = {'fontname':'Helvetica'}

def theta_fit(Phi, theta_Phi, theta_from_Phi, transform=True, sigma=0, seed=23):
    """
    General plotting function to plot latent space theta vs theta_Phi,
    Inputs:
        Phi - the test set data
        theta_Phi - the test set theta
        theta_from_Phi - function that gets latent theta from Phi
        sigma - standard deviation for normally distributed noise
        seed - random seed for reprodicibility
    """
    # get input and targets
    Phi, _ = add_noise(Phi, sigma=sigma, seed=seed)
    theta = theta_from_Phi(Phi)

    # evaluate, predict, and plot with trained model
    theta_mse = np.mean((theta - theta_Phi) ** 2, axis=0)
    print('theta MSE:', theta_mse)

    # initialize axes
    num_plots = theta_Phi.shape[1]
    fig, ax = plt.subplots(nrows=1, ncols=num_plots,
            sharey=transform,
            figsize=(20,10),
            dpi=200)

    # plot transformed thetas
    if transform:
        theta_Phi, theta = rescale_thetas(theta_Phi, theta)
        
        # evaluate, predict, and plot with trained model
        theta_tform_mse = np.mean((theta - theta_Phi) ** 2, axis=0)
        print('transformed theta MSE:', theta_tform_mse)

    subplot_theta_fit(fig, ax, theta_Phi, theta)

    plt.show()
    plt.close()

def subplot_theta_fit(fig, ax, theta_Phi, theta):
    """
    Plots theta vs theta_Phi
    Agnostic to data transformation
    """
    theta_names = ['$c$', '$b_0$', '$b_1$']
    num_plots = len(ax)
    for i in range(num_plots):
        ax[i].scatter(theta_Phi[:,i], theta[:,i],
                alpha=0.7)
        ax[i].set_title(theta_names[i], fontsize=22)
        identity(ax[i])
    fig.add_subplot(111, frame_on=False)
    plt.tick_params(labelcolor='none', bottom=False, left=False)
    plt.xlabel('Truth', fontsize=26)
    plt.ylabel('Prediction', fontsize=26, labelpad=40.0)

def solution_fit(Phi, theta_Phi, theta_from_Phi, u_from_Phi, sigma=0, seed=23):
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
    """
    # make noise
    Phi, noise = add_noise(Phi, sigma=sigma, seed=seed)
    theta = theta_from_Phi(Phi)
    u_theta = u_from_Phi(Phi)
    domain = np.linspace(0, 1, Phi.shape[1])
    theta_names = ['$c$', '$b_0$', '$b_1$']

    # compute mean square errors
    theta_mse = np.mean((theta - theta_Phi) ** 2, axis=0)
    Phi_mse = np.mean((u_theta - Phi) ** 2)
    print('Latent theta MSE:', theta_mse)
    print('Reconstructed Phi MSE:', Phi_mse)

    # generate sample
    num_plots = 9
    sample_idx = np.random.randint(0, len(Phi)-1, num_plots)
    Phi = Phi[sample_idx]
    noise = noise[sample_idx]
    u_theta = u_theta[sample_idx]
    idx = np.array([x for x in range(num_plots)]).reshape(3, 3)

    commas = [',  ', ',  ', '']
    fig, ax = plt.subplots(nrows=3, ncols=3,
                           sharex=True,
                           figsize=(20,20),
                           dpi=200)
    labs = [r'$u_\theta$', r'$\Phi$', r'$\Phi +$noise']
    for i in np.ndindex(3, 3):
        # make the title
        title = ''
 #       for j, name in enumerate(theta_names):
 #           title += r'%s' % name
 #           title += r'$= %.2f$' % theta[i,j]
 #           title += commas[j]
        # make the plots
        ax[i].plot(domain, u_theta[idx[i]], label=labs[0], lw=3)
        ax[i].plot(domain, Phi[idx[i]] - noise[idx[i]],
                label=labs[1],
                lw=3,
                ls='dashed',
                c='k')
        if sigma != 0: # plot Phi with noise
            ax[i].plot(domain, Phi[idx[i]],
                    label=labs[2],
                    lw=3,
                    #ls='dashed',
                    c='0.6')
        ax[i].set_xlabel(r'$x$', fontsize=26)
        ax[i].set_xticks([0, 1])
        ax[i].set_title(title, fontsize=18)
        labs = ['' for lab in labs] # don't label other plots
    fig.legend(fontsize=28)
    plt.show()
    plt.close()

def theta_boot(test, preds, sigma):
    # compute bootstrap means and confidence bounds
    test_theta = deepcopy(test[1]) # test is assumed to be (inputs, outputs)
    num_boots = len(preds)
    boot_num, results = zip(*preds.items())
    preds = np.array(results)

    boot_means = np.mean(preds, axis=0)
    boot_ub = np.quantile(preds, 0.975, axis=0)
    boot_lb = np.quantile(preds, 0.025, axis=0)

    # make errorbars
    num_points = preds.shape[1]
    err_shape = (2, num_points)
    boot_errs = [np.zeros(err_shape),
                 np.zeros(err_shape),
                 np.zeros(err_shape)]
    for i in range(3):
        boot_errs[i][0] = boot_ub[:,i] - boot_means[:,i]
        boot_errs[i][1] = boot_means[:,i] - boot_lb[:,i]

    # errorbar plots
    fig, ax = plt.subplots(3,1, figsize=(20,20), dpi=200)
    param = [r'$c$', r'$b_0$', r'$b_1$']
    title = '%d bootstrap samples' % num_boots
    # noise is mean 0 mean Gaussian with stdev sigma
    if sigma != 0:
        title += r', noise$\sim\operatorname{Normal}(0,\sigma=%d)$' % sigma
    ax[0].set_title(title)
    for i in range(3):
        ax[i].scatter(test_theta[:,i], boot_means[:,i], 
                label=param[i] + ' boot means')
        ax[i].errorbar(test_theta[:,i], boot_means[:,i], 
                yerr=boot_errs[i],
                alpha=0.25,
                fmt='|',
                label='95% credible region')
        identity(ax[i])
        ax[i].set_ylabel('Predictions')
        ax[i].legend(loc='upper left')
    ax[2].set_xlabel('Truth')
    plt.show()
    plt.close()

def solution_boot(test, preds):
    pass

####################
# HELPER FUNCTIONS #
####################

def add_noise(x, sigma, seed=23):
    """
    Adds Gaussian noise with stdev sigma to x
    returns noisy x and the noise so 
    we can denoise x later if we wish
    """
    np.random.seed(seed)
    noise = np.random.randn(x.size)
    noise = np.reshape(noise, x.shape)
    noise *= sigma
    return x+noise, noise

def rescale_thetas(theta_Phi, theta):
    """
    Rescales true and predicted thetas so the pred vs true plot
    is bounded by -1 and 1
    Returns rescaled theta_Phi and theta
    """
    # combine theta and theta_Phi to transform
    thetas = np.hstack((theta, theta_Phi))
    thetas_tformer = preprocessing.MaxAbsScaler()
    thetas_tform = thetas_tformer.fit_transform(thetas)

    # transformed thetas
    theta = thetas_tform[:,:3]
    theta_Phi = thetas_tform[:,3:]

    return theta_Phi, theta

def identity(ax):
    """ Plot the identity map y=x """
    x = np.array(ax.get_xlim())
    y = x 
    ax.plot(x, y, c='r', lw=3)
