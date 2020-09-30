""" General helper functions for Poisson package """

import numpy as np
from copy import deepcopy
from sklearn import preprocessing

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
    x += noise
    return x, noise

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

def noise_title(train_sigma, test_sigma):
    """
    Generate title showing amount of noise used
    during bootstrap training and testing
    """
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

def boot_stats(boot_data, theta_true):
    """
    Given dictionary containing bootstrap data,
    compute bootstrap means, 68%, and 95% credible regions

    The values in boot_data have shape
      (boots, test_set_size, prediction_shape)

    The bootstrap standard error for the mean is the
    sample standard devation of the bootstrap sample means
    """
    # rescale thetas and compute bootstrap residuals
    _, results = zip(*boot_data.items())
    boot_data = np.array(results) # bootstrap thetas
    print('boot_data shape:', boot_data.shape)
    boot_resids = np.zeros(boot_data.shape)
    boots = len(boot_data)
    for b in range(boots):
        theta_Phi, theta = rescale_thetas(theta_true, boot_data[b])
        boot_resids[b] = theta - theta_Phi

    # compute mean, SE, quantiles
    stats = dict()
    stats['rescaled_true_theta'] = theta_Phi
    stats['means'] = np.mean(boot_resids, axis=0) # mean across boot samples
    stats['SE'] = np.std(boot_resids, axis=0, ddof=1)
    print('means', stats['means'])

    # quantiles
    stats['upper95'] = np.quantile(boot_resids, 0.975, axis=0)
    stats['upper68'] = np.quantile(boot_resids, 0.84, axis=0)
    stats['lower68'] = np.quantile(boot_resids, 0.16, axis=0)
    stats['lower95'] = np.quantile(boot_resids, 0.025, axis=0)
    print('upper95', stats['upper95'])
    print('lower95', stats['lower95'])

    # make errorbars
    num_points = boot_resids.shape[1]
    err_shape = (2, num_points)
    stats['credint95'] = [np.zeros(err_shape),
                 np.zeros(err_shape),
                 np.zeros(err_shape)]
    stats['credint68'] = deepcopy(stats['credint95'])

    for i in range(3):
        stats['credint95'][i][1] = stats['upper95'][:,i] - stats['means'][:,i]
        stats['credint95'][i][0] = stats['means'][:,i] - stats['lower95'][:,i]
        stats['credint68'][i][1] = stats['upper68'][:,i] - stats['means'][:,i]
        stats['credint68'][i][0] = stats['means'][:,i] - stats['lower68'][:,i]

    return stats
