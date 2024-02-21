import torch
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import pickle

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

OUT_DIR = "./exp/1D_snelson"
FIG_DIR = os.path.join(OUT_DIR, "figures")
util.ensure_dir(OUT_DIR)
util.ensure_dir(FIG_DIR)


def pickle_load(tag):
    if not ".npy" in tag or ".pickle" in tag:
        tag = f"{tag}.pickle"
    pickle_in = open(tag, "rb")
    if "npy" in tag:
        to_return = np.load(pickle_in)
    else:
        to_return = pickle.load(pickle_in)
    pickle_in.close()
    return to_return

def plot_samples(X, samples, var=None, n_keep=12, color="xkcd:bluish", ax=None):
    if ax is None:
        ax = plt.gca()
    if samples.ndim > 2:
        samples = samples.squeeze()
    n_keep = int(samples.shape[1]/10) if n_keep is None else n_keep
    keep_idx = np.random.permutation(samples.shape[1])[:n_keep]
    mu = samples.mean(1)
    if var is None:
        q=97.72  ## corresponds to 2 stdevs in Gaussian
        # q = 99.99  ## corresponds to 3 std
        Q = np.percentile(samples, [100-q, q], axis=1)
        # ub, lb = Q[1,:], Q[0,:]
        ub, lb = mu + 3 * samples.std(1), mu - 3 * samples.std(1)
    else:
        ub = mu + 3 * np.sqrt(var)
        lb = mu - 3 * np.sqrt(var)
    ####
    ax.fill_between(X.flatten(), ub, lb, color=color, alpha=0.25, lw=0)
    ax.plot(X, samples[:, keep_idx], color=color, alpha=0.8)
    ax.plot(X, mu, color='xkcd:red')


# Generate data
util.set_seed(1)

N = 64
M = 100
a, b = -10, 10

# load data
data = pickle_load(f"{OUT_DIR}/data")
x = data['train']['x']
y = data['train']['y']
x_plot = data['plot']['x']
y_plot = data['plot']['y']

fig = plt.figure()
plt.plot(x, y, "ko", ms=5)
plt.title("Dataset")
plt.show()

# Initialize Priors
util.set_seed(1)

# GP hyper-parameters
sn2 = 0.176   # noise variance # \todo set to optimal parameters
leng = 0.473  # lengthscale
ampl = 1.0  # amplitude

# Initialize GP Prior
kernel = kernels.RBF(
        input_dim=1, ARD=True,
        lengthscales=torch.tensor([leng], dtype=torch.double),
        variance=torch.tensor([ampl], dtype=torch.double))

gpmodel = GPR(X=x.to(device),
              Y=y.to(device),
              kern=kernel, mean_function=mean_functions.Zero())
gpmodel.likelihood.variance.set(sn2)
gpmodel = gpmodel.to(device)

util.set_seed(1)

# Initialize BNN Priors
width = 32              # Number of units in each hidden layer
depth = 1               # Number of hidden layers
transfer_fn = "tanh"    # Activation function
connectivity_params = 3
if connectivity_params == 0:
    prior_per = 'parameter'
elif connectivity_params == 3:
    prior_per = 'layer'
else:
    raise ValueError

# Initialize Gaussian prior.
# Fixed Prior
std_bnn = GaussianMLPReparameterization(input_dim=1, output_dim=1, activation_fn=transfer_fn,
    hidden_dims=[width]*depth, scaled_variance=False)

# Prior to be optimized
opt_bnn = GaussianMLPReparameterization(input_dim=1, output_dim=1, activation_fn=transfer_fn,
    hidden_dims=[width]*depth, prior_per=prior_per, scaled_variance=False)

std_bnn = std_bnn.to(device)
opt_bnn = opt_bnn.to(device)

# # # Optimize Prior
# #
# # # We use a grid of 200 data points in [-6, 6] for the measurement set
# util.set_seed(1)
# data_generator = GridGenerator(x_plot.min(), x_plot.max())
#
# mapper_num_iters = 2 # 800 # Define the number of iterations of Wasserstein optimization
#
# # Initiialize the Wasserstein optimizer
# util.set_seed(1)
# mapper = MapperWasserstein(gpmodel, opt_bnn, data_generator,
#                            out_dir=OUT_DIR,
#                            wasserstein_steps=(0, 1000),
#                            wasserstein_lr=0.08,
#                            n_data=200, n_gpu=1, gpu_gp=True)
#
# # Start optimizing the prior
# w_hist = mapper.optimize(num_iters=mapper_num_iters, n_samples=512, lr=0.01,
#                          save_ckpt_every=50, print_every=20, debug=True)
# path = os.path.join(OUT_DIR, "wsr_values.log")
# np.savetxt(path, w_hist, fmt='%.6e')
#
# # Visualize progression of the prior optimization
# wdist_file = os.path.join(OUT_DIR, "wsr_values.log")
# wdist_vals = np.loadtxt(wdist_file)
#
# fig = plt.figure(figsize=(6, 3.5))
# indices = np.arange(mapper_num_iters)[::5]
# plt.plot(indices, wdist_vals[indices], "-ko", ms=4)
# plt.ylabel(r"$W_1(p_{gp}, p_{nn})$")
# plt.xlabel("Iteration")
# plt.show()


# # # Visualize Prior
# # Load the optimize prior
# util.set_seed(1)
# ckpt_path = os.path.join(OUT_DIR, "ckpts", "it-{}.ckpt".format(mapper_num_iters))
# opt_bnn.load_state_dict(torch.load(ckpt_path))
#
# ## save prior in readable format
# import os
#
# state_dict = torch.load(ckpt_path)
# to_save = dict()
# first_layer_tag = 'layers.0'
# last_layer_tag = 'output_layer'
# param_tags = ['W_std', 'b_std']
#
# def get_info_layer(layer_tag):
#     return {tag: state_dict[f"{layer_tag}.{tag}"] for tag in param_tags}
#
# to_save[0] = get_info_layer(first_layer_tag)
# for idx in range(1, depth):
#     to_save[idx] = get_info_layer(f"layers.linear_{idx}")
# to_save[depth] = get_info_layer(last_layer_tag)
#
# tag = f"{OUT_DIR}/[{depth}x{width}]_{transfer_fn}_con={connectivity_params}"
# torch.save(to_save, tag)
#
# Draw functions from the priors
n_plot = 4000
util.set_seed(8)

gp_samples = gpmodel.sample_functions(
   x_plot.to(device), n_plot).detach().cpu().numpy().squeeze()
# gp_samples = zscore_unnormalization(gp_samples, y_mean, y_std)

std_bnn_samples = std_bnn.sample_functions(
    x_plot.to(device), n_plot).detach().cpu().numpy().squeeze()
# std_bnn_samples = zscore_unnormalization(std_bnn_samples, y_mean, y_std)

opt_bnn_samples = opt_bnn.sample_functions(
    x_plot.to(device), n_plot).detach().cpu().numpy().squeeze()
# opt_bnn_samples = zscore_unnormalization(opt_bnn_samples, y_mean, y_std)


fig, axs = plt.subplots(1, 3, figsize=(14, 3))
plot_samples(x_plot, gp_samples, ax=axs[0], n_keep=15)
axs[0].set_title('GP Prior')
# axs[0].set_ylim([-5, 5])

plot_samples(x_plot, std_bnn_samples, ax=axs[1], color='xkcd:grass', n_keep=15)
axs[1].set_title('BNN Prior (Fixed)')
# axs[1].set_ylim([-5, 5])

plot_samples(x_plot, opt_bnn_samples, ax=axs[2], color='xkcd:yellowish orange', n_keep=15)
axs[2].set_title('BNN Prior (GP-induced)')
axs[2].set_ylim([-5, 5])

plt.tight_layout()
plt.show()

#
# # # Posterior Inference
# #
# #
#
# # ## GP
# #
#
# # In[ ]:
#
#
# # Make predictions
# util.set_seed(1)
# gp_preds = gpmodel.predict_f_samples(Xtest_tensor, 1000)
# gp_preds = gp_preds.detach().cpu().numpy().squeeze()
# gp_preds = zscore_unnormalization(gp_preds, y_mean, y_std)
#
#
# # ## BNN with Fixed Prior
#
# # SGHMC Hyper-parameters
# sampling_configs = {
#     "batch_size": 32,           # Mini-batch size
#     "num_samples": 30,          # Total number of samples for each chain
#     "n_discarded": 10,          # Number of the first samples to be discared for each chain
#     "num_burn_in_steps": 2000,  # Number of burn-in steps
#     "keep_every": 200,          # Thinning interval
#     "lr": 0.01,                 # Step size
#     "num_chains": 4,            # Number of chains
#     "mdecay": 0.01,             # Momentum coefficient
#     "print_every_n_samples": 5
# }
#
# # Initialize the prior
# util.set_seed(1)
# prior = FixedGaussianPrior(std=1.0)
#
# # Setup likelihood
# net = MLP(1, 1, [width]*depth, transfer_fn)
# likelihood = LikGaussian(sn2)
#
# # Initialize the sampler
# saved_dir = os.path.join(OUT_DIR, "sampling_std")
# util.ensure_dir(saved_dir)
# bayes_net_std = RegressionNet(net, likelihood, prior, saved_dir, n_gpu=0)
#
# # Start sampling
# bayes_net_std.sample_multi_chains(X, y, **sampling_configs)
#
#
# # In[ ]:
#
#
# # Make predictions
# util.set_seed(1)
# _, _, bnn_std_preds = bayes_net_std.predict(Xtest, True)
#
# # Convergence diagnostics using the R-hat statistic
# r_hat = compute_rhat_regression(bnn_std_preds, sampling_configs["num_chains"])
# print(r"R-hat: mean {:.4f} std {:.4f}".format(float(r_hat.mean()), float(r_hat.std())))
# bnn_std_preds = bnn_std_preds.squeeze().T
#
# # Save the predictions
# posterior_std_path = os.path.join(OUT_DIR, "posterior_std.npz")
# np.savez(posterior_std_path, bnn_samples=bnn_std_preds)
#
#
# # ## BNN with Optimized Prior
# #
# #
#
# # In[ ]:
#
#
# # Load the optimized prior
# ckpt_path = os.path.join(OUT_DIR, "ckpts", "it-{}.ckpt".format(mapper_num_iters))
# prior = OptimGaussianPrior(ckpt_path)
#
# # Setup likelihood
# net = MLP(1, 1, [width]*depth, transfer_fn)
# likelihood = LikGaussian(sn2)
#
# # Initialize the sampler
# saved_dir = os.path.join(OUT_DIR, "sampling_optim")
# util.ensure_dir(saved_dir)
# bayes_net_optim = RegressionNet(net, likelihood, prior, saved_dir, n_gpu=0)
#
# # Start sampling
# bayes_net_optim.sample_multi_chains(X, y, **sampling_configs)
#
#
# # In[ ]:
#
#
# # Make predictions
# util.set_seed(1)
# _, _, bnn_optim_preds = bayes_net_optim.predict(Xtest, True)
#
# # Convergence diagnostics using the R-hat statistic
# r_hat = compute_rhat_regression(bnn_optim_preds, sampling_configs["num_chains"])
# print(r"R-hat: mean {:.4f} std {:.4f}".format(float(r_hat.mean()), float(r_hat.std())))
# bnn_optim_preds = bnn_optim_preds.squeeze().T
#
# # Save the predictions
# posterior_optim_path = os.path.join(OUT_DIR, "posterior_optim.npz")
# np.savez(posterior_optim_path, bnn_samples=bnn_optim_preds)
#
#
# # ## Visualize Predictive Posterior
#
# # In[ ]:
#
#
# util.set_seed(8)
# fig, axs = plt.subplots(1, 3, figsize=(14, 3))
#
# plot_samples(Xtest, gp_preds, ax=axs[0], n_keep=16)
# axs[0].plot(X, y, 'ok', zorder=10, ms=5)
# axs[0].set_title('GP Posterior')
# axs[0].set_ylim([-4, 4])
#
# plot_samples(Xtest, bnn_std_preds, ax=axs[1], color='xkcd:grass', n_keep=16)
# axs[1].plot(X, y, 'ok', zorder=10, ms=5)
# axs[1].set_title('BNN Posterior (Fixed)')
# axs[1].set_ylim([-4, 4])
#
# plot_samples(Xtest, bnn_optim_preds, ax=axs[2], color='xkcd:yellowish orange', n_keep=16)
# axs[2].plot(X, y, 'ok', zorder=10, ms=5)
# axs[2].set_title('BNN Posterior (GP-induced)')
# axs[2].set_ylim([-4, 4])
#
# plt.tight_layout()
# plt.show()

