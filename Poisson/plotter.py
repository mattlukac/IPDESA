import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
plt.rcParams['font.size'] = 26


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
    np.random.seed(seed)
    noise = np.random.randn(Phi.size)
    noise = np.reshape(noise, Phi.shape)
    Phi += sigma * noise
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
        # combine theta and theta_Phi to transform
        thetas = np.hstack((theta, theta_Phi))
        thetas_tformer = preprocessing.MaxAbsScaler()
        thetas_tform = thetas_tformer.fit_transform(thetas)

        # transformed thetas
        theta = thetas_tform[:,:3]
        theta_Phi = thetas_tform[:,3:]

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
        ax[i].plot([0,1], [0,1],
                transform=ax[i].transAxes,
                c='r',
                linewidth=3)
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
    np.random.seed(seed) # reproducible noise
    noise = np.random.randn(Phi.size)
    noise = sigma * np.reshape(noise, Phi.shape)
    Phi += noise
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
                    ls='dashed',
                    c='0.5')
        ax[i].set_xlabel(r'$x$', fontsize=26)
        ax[i].set_xticks([0, 1])
        ax[i].set_title(title, fontsize=18)
        labs = ['' for lab in labs] # don't label other plots
    fig.legend(fontsize=28)
    plt.show()
    plt.close()

