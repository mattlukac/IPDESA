import numpy as np
from copy import deepcopy
from sklearn import preprocessing
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 26
plt.rcParams['figure.dpi'] = 200
plt.style.use(['seaborn-bright'])


##################
# PLOT SOLUTIONS #
##################

def solution(domain, solution, theta):
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

    plt.show()
    plt.close()

##################
# PLOT MODEL FIT # 
##################

def theta_fit(Phi, theta_Phi, theta_from_Phi, sigma=0, seed=23, transform=True):
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
            sharey=True,
            figsize=(20,10))

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
        ax[i].set_title(theta_names[i], fontsize=26)
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
        labs = ['' for lab in labs] # don't label other plots
    cols = 3 if sigma != 0 else 2
    fig.legend(fontsize=26, loc='upper center', shadow=True, ncol=cols)

    plt.show()
    plt.close()

###################
# BOOTSTRAP PLOTS #
###################

def theta_boot(test, boot_data, sigmas):
    # compute bootstrap means and confidence bounds
    test_theta = deepcopy(test[1]) # ground truth test theta
    num_boots = len(boot_data)
    stats = boot_stats(boot_data)

    # errorbar plots
    fig, ax = plt.subplots(3,1, figsize=(20,20))
    param = [r'$c$', r'$b_0$', r'$b_1$']
    train_sigma, test_sigma = sigmas
    title = noise_title(train_sigma, test_sigma)
    ax[0].set_title(title)
    for i in range(3):
        # plot bootstrap means
        ax[i].scatter(test_theta[:,i], stats['means'][:,i], 
                label=param[i] + ' boot means')
        # plot bootstrap credible regions
        ax[i].errorbar(test_theta[:,i], stats['means'][:,i], 
                yerr=stats['credint95'][i],
                alpha=0.25,
                fmt='|',
                label='95% credible region')
        # y=x
        identity(ax[i])
        ax[i].set_ylabel('Predictions')
        ax[i].legend(loc='upper left', fontsize=26, shadow=True)
    ax[2].set_xlabel('Truth')
    plt.show()
    plt.close()

def solution_boot(test, boot_data, u_from_theta, sigmas):
    # compute bootstrap means and credible region
    # credible region quantiles are computed over MSE(Phi, u)
    train_sigma, test_sigma = sigmas 
    Phi, theta_Phi = deepcopy(test)
    domain = np.linspace(0, 1, Phi.shape[1])
    Phi, noise = add_noise(Phi, test_sigma) # noisy test inputs
    num_boots = len(boot_data)
    stats = boot_stats(boot_data)

    # generate sample
    num_plots = 9
    sample_idx = np.random.randint(0, len(Phi)-1, num_plots)
    Phi = Phi[sample_idx]
    noise = noise[sample_idx]
    means = u_from_theta(stats['means'][sample_idx])
    upper95 = u_from_theta(stats['upper95'][sample_idx])
    lower95 = u_from_theta(stats['lower95'][sample_idx])
    upper68 = u_from_theta(stats['upper68'][sample_idx])
    lower68 = u_from_theta(stats['lower68'][sample_idx])

    # for indexing plot grid
    idx = np.array([x for x in range(num_plots)]).reshape(3, 3)

    # for each curve Phi, we plot:
    # denoised Phi, bootstrap mean curve, 
    # fill between boot_ub and boot_lb solutions
    # title should say how much noise was added
    fig, ax = plt.subplots(nrows=3, ncols=3,
                           sharex=True,
                           figsize=(20,20))
    labs = [r'$u_\theta$ boot mean', 
            '95% credible region', 
            '68% credible region',
            r'$\Phi$', 
            r'$\Phi +$noise']
    for i in np.ndindex(3, 3):
        # plot u
        ax[i].plot(domain, means[idx[i]], label=labs[0], lw=3)
        # 95% credible interval
        ax[i].fill_between(domain, lower95[idx[i]], upper95[idx[i]], 
                color='C0', 
                alpha=0.12,
                label=labs[1])
        # 68% credible interal
        ax[i].fill_between(domain, lower68[idx[i]], upper68[idx[i]], 
                color='C0', 
                alpha=0.25,
                label=labs[2])
        # plot denoised Phi
        ax[i].plot(domain, Phi[idx[i]] - noise[idx[i]],
                label=labs[3],
                lw=3,
                ls='dashed',
                alpha=1.0,
                c='k')
        # plot noisy Phi
        if test_sigma != 0: 
            ax[i].plot(domain, Phi[idx[i]],
                    label=labs[4],
                    lw=3,
                    alpha=0.2,
                    c='k')
        if i[0] == 2:
            ax[i].set_xlabel(r'$x$', fontsize=26)
        ax[i].set_xticks([0, 1])
        labs = ['' for lab in labs] # don't label other plots

    # title
    fig.add_subplot(111, frame_on=False)
    plt.tick_params(labelcolor='none', bottom=False, left=False)
    title = noise_title(train_sigma, test_sigma)
    plt.title(title, pad=20)
    
    # legend 
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
            shadow=True,
            ncol = len(new_order))
    
    plt.show()
    plt.close()


####################
# HELPER FUNCTIONS #
####################

def boot_stats(boot_data):
    # compute means and quantiles 
    _, results = zip(*boot_data.items())
    boot_data = np.array(results)
    stats = dict()
    stats['means'] = np.mean(boot_data, axis=0)
    stats['upper95'] = np.quantile(boot_data, 0.975, axis=0)
    stats['upper68'] = np.quantile(boot_data, 0.84, axis=0)
    stats['lower68'] = np.quantile(boot_data, 0.16, axis=0)
    stats['lower95'] = np.quantile(boot_data, 0.025, axis=0)

    # make errorbars
    num_points = boot_data.shape[1]
    err_shape = (2, num_points)
    stats['credint95'] = [np.zeros(err_shape),
                 np.zeros(err_shape),
                 np.zeros(err_shape)]
    stats['credint68'] = deepcopy(stats['credint95'])

    for i in range(3):
        stats['credint95'][i][0] = stats['upper95'][:,i] - stats['means'][:,i]
        stats['credint95'][i][1] = stats['means'][:,i] - stats['lower95'][:,i]
        stats['credint68'][i][0] = stats['upper68'][:,i] - stats['means'][:,i]
        stats['credint68'][i][1] = stats['means'][:,i] - stats['lower68'][:,i]

    return stats

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
    ax.plot(x, y, c='r', lw=3, alpha=0.5)

def noise_title(train_sigma, test_sigma):
    title = ''
    if train_sigma != 0:
        title += 'training noise'
        title += r'$\sim\operatorname{Normal}(0,\sigma=%.1f)$' % train_sigma
        title += '\n'
    else:
        title += 'no training noise\n'
    if test_sigma != 0:
        title += 'testing noise'
        title += r'$\sim\operatorname{Normal}(0,\sigma=%.1f)$' % test_sigma
    else:
        title += 'no testing noise'
    return title
