import torch
import numpy as np
import math
import matplotlib.pylab as plt
import matplotlib as mpl

import os
import sys
import platform

import warnings
warnings.simplefilter("ignore", UserWarning)

os.chdir("..")

from optbnn.gp.models.gpr import GPR
from optbnn.gp import kernels, mean_functions
from optbnn.bnn.reparam_nets import GaussianMLPReparameterization
from optbnn.bnn.nets.mlp import MLP
from optbnn.bnn.likelihoods import LikGaussian
from optbnn.bnn.priors import FixedGaussianPrior, OptimGaussianPrior
from optbnn.prior_mappers.wasserstein_mapper import MapperWasserstein, WassersteinDistance
from optbnn.utils.rand_generators import MeasureSetGenerator, GridGenerator
from optbnn.utils.normalization import normalize_data, zscore_normalization, zscore_unnormalization
from optbnn.metrics.sampling import compute_rhat_regression
from optbnn.utils import util
from optbnn.sgmcmc_bayes_net.regression_net import RegressionNet

mpl.rcParams['figure.dpi'] = 100

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

file_path = os.path.abspath(__file__)
OUT_DIR = f"{os.path.dirname(os.path.dirname(file_path))}{os.sep}exp{os.sep}prior1d"
print(OUT_DIR)
util.ensure_dir(OUT_DIR)


def plot_samples(X, samples, var=None, n_keep=12, color="xkcd:bluish", smooth_q=False, ax=None):
    if ax is None:
        ax = plt.gca()
    if samples.ndim > 2:
        samples = samples.squeeze()
    n_keep = int(samples.shape[1]/10) if n_keep is None else n_keep
    keep_idx = np.random.permutation(samples.shape[1])[:n_keep]
    mu = samples.mean(1)
    if var is None:
        ub, lb = mu + 3 * samples.std(1), mu - 3 * samples.std(1)
    else:
        ub = mu + 3 * np.sqrt(var)
        lb = mu - 3 * np.sqrt(var)
    ####
    ax.fill_between(X.flatten(), ub, lb, color=color, alpha=0.25, lw=0)
    ax.plot(X, samples[:, keep_idx], color=color, alpha=0.8)
    ax.plot(X, mu, color='xkcd:red')


if __name__ == '__main__':
    util.set_seed(1)
    DEBUG = True

    # setup
    # data
    N = 1024  # number of training points
    M = 100  # number of test points
    a, b = -1, 1  # input domain

    # gp
    sn2 = 0.1  # noise variance
    leng = 1.0  # lengthscale
    ampl = 1.0  # amplitude

    # bnn
    width = 128              # Number of units in each hidden layer
    depth = 2               # Number of hidden layers
    transfer_fn = "tanh"    # Activation function
    con = 3

    # optimization
    n_meas_set = 200  # number of points in measurement set
    if DEBUG:
        mapper_num_iters = 2
        n_samples = 10
    else:
        mapper_num_iters = 800
        n_samples = 512

    params_mapper = {'wasserstein_steps': (0, 1000), 'wasserstein_lr': 0.08, 'n_data': n_meas_set, 'n_gpu': 1,
                     'gpu_gp': True}
    params_opt = {'num_iters': mapper_num_iters, 'n_samples': n_samples, 'lr': 0.01, 'save_ckpt_every': 50,
                  'print_every': 20, 'debug': True}

    # save tag
    tag = f'cmf_[{depth}x{width}]_{transfer_fn}_con={con}_iters={mapper_num_iters}_rbf_ls={leng}'

    # Generate data
    X = np.random.rand(N, 1) * (b-a) + a
    y = np.full(X.shape, np.nan)
    Xtest = np.linspace(a, b, M).reshape(-1, 1)

    # Normalize the dataset
    X_, X_mean, X_std = zscore_normalization(X)
    y_, y_mean, y_std = zscore_normalization(y)
    Xtest_, _, _ = zscore_normalization(Xtest, X_mean, X_std)

    Xtest_tensor = torch.from_numpy(Xtest_).to(device)

    # Initialize Priors
    kernel = kernels.RBF(
            input_dim=1, ARD=True,
            lengthscales=torch.tensor([leng], dtype=torch.double),
            variance=torch.tensor([ampl], dtype=torch.double))

    gpmodel = GPR(X=torch.from_numpy(X_).to(device),
                  Y=torch.from_numpy(y_).reshape([-1, 1]).to(device),
                  kern=kernel, mean_function=mean_functions.Zero())
    gpmodel.likelihood.variance.set(sn2)
    gpmodel = gpmodel.to(device)

    gp_samples = gpmodel.sample_functions(
       Xtest_tensor, 10).detach().cpu().numpy().squeeze()

    # Initialize Gaussian prior.
    if con == 0:
        prior_per = 'parameter'
    elif con == 3:
        prior_per = 'layer'
    else:
        raise NotImplementedError

    # Fixed Prior
    std_bnn = GaussianMLPReparameterization(input_dim=1, output_dim=1, activation_fn=transfer_fn,
                                            hidden_dims=[width]*depth, prior_per=prior_per)

    # Prior to be optimized
    opt_bnn = GaussianMLPReparameterization(input_dim=1, output_dim=1, activation_fn=transfer_fn,
                                            hidden_dims=[width]*depth, prior_per=prior_per)

    std_bnn = std_bnn.to(device)
    opt_bnn = opt_bnn.to(device)

    if not os.path.exists(os.path.join(OUT_DIR, f"{tag}.ckpt")):
        # Optimize Prior
        data_generator = GridGenerator(-1, 1)

        # Initialize the Wasserstein optimizer
        mapper = MapperWasserstein(gpmodel, opt_bnn, data_generator, out_dir=OUT_DIR, **params_mapper)

        # Start optimizing the prior
        w_hist = mapper.optimize(**params_opt)
        path = os.path.join(OUT_DIR, "wsr_values.log")
        np.savetxt(path, w_hist, fmt='%.6e')

        # Visualize progression of the prior optimization
        wdist_file = os.path.join(OUT_DIR, "wsr_values.log")
        wdist_vals = np.loadtxt(wdist_file)

        if platform.system() == "Windows" and DEBUG:
            fig = plt.figure(figsize=(6, 3.5))
            indices = np.arange(mapper_num_iters)[::5]
            plt.plot(indices, wdist_vals[indices], "-ko", ms=4)
            plt.ylabel(r"$W_1(p_{gp}, p_{nn})$")
            plt.xlabel("Iteration")
            plt.show()

        ckpt_path = os.path.join(OUT_DIR, "ckpts", "it-{}.ckpt".format(mapper_num_iters))
        torch.save(torch.load(ckpt_path), os.path.join(OUT_DIR, f"{tag}.ckpt"))

    # Visualize Prior
    # Load the optimize prior
    opt_bnn.load_state_dict(torch.load(os.path.join(OUT_DIR, f"{tag}.ckpt")))

    # Draw functions from the priors
    n_plot = 4000
    util.set_seed(8)

    gp_samples = gpmodel.sample_functions(
       Xtest_tensor, n_plot).detach().cpu().numpy().squeeze()

    std_bnn_samples = std_bnn.sample_functions(
        Xtest_tensor.float(), n_plot).detach().cpu().numpy().squeeze()

    opt_bnn_samples = opt_bnn.sample_functions(
        Xtest_tensor.float(), n_plot).detach().cpu().numpy().squeeze()

    fig, axs = plt.subplots(1, 3, figsize=(14, 3))
    plot_samples(Xtest_, gp_samples, ax=axs[0], n_keep=5)
    axs[0].set_title('GP Prior')
    axs[0].set_ylim([-5, 5])

    plot_samples(Xtest_, std_bnn_samples, ax=axs[1], color='xkcd:grass', n_keep=5)
    axs[1].set_title('BNN Prior (Fixed)')
    axs[1].set_ylim([-5, 5])

    plot_samples(Xtest_, opt_bnn_samples, ax=axs[2], color='xkcd:yellowish orange', n_keep=5)
    axs[2].set_title('BNN Prior (GP-induced)')
    axs[2].set_ylim([-5, 5])

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"{tag}.pdf"))
    if platform.system() == "Windows" and DEBUG:
        plt.show()

