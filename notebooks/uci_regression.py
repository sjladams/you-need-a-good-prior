#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[42]:


import torch
import pandas as pd
import numpy as np
import math
import matplotlib.pylab as plt
import matplotlib as mpl

import os
import sys


# In[3]:


os.chdir("..")


# In[39]:


from optbnn.gp.models.gpr import GPR
from optbnn.gp import kernels, mean_functions, priors
from optbnn.bnn.reparam_nets import GaussianMLPReparameterization
from optbnn.bnn.nets.mlp import MLP
from optbnn.bnn.likelihoods import LikGaussian
from optbnn.bnn.priors import FixedGaussianPrior, OptimGaussianPrior
from optbnn.prior_mappers.wasserstein_mapper import MapperWasserstein, WassersteinDistance
from optbnn.utils.rand_generators import MeasureSetGenerator, GridGenerator
from optbnn.utils.normalization import normalize_data
from optbnn.utils.exp_utils import get_input_range
from optbnn.metrics.sampling import compute_rhat_regression
from optbnn.metrics import uncertainty as uncertainty_metrics
from optbnn.sgmcmc_bayes_net.regression_net import RegressionNet
from optbnn.utils import util


# In[18]:


mpl.rcParams['figure.dpi'] = 100


# In[5]:


SEED = 123
util.set_seed(SEED)


# In[6]:


# Network architecture
n_units = 100
n_hidden = 1
activation_fn = "tanh"


# In[7]:


# Dataset configurations
n_splits = 10
dataset = "boston"
data_dir = "./data/uci"
noise_var = 0.1


# # 1. Optimized Gaussian Prior

# In[8]:


out_dir = "./exp/uci/optim_gaussian"
util.ensure_dir(out_dir)


# ## 1.1 Optimize the prior

# In[9]:


num_iters = 200  # Number of iteterations of Wasserstein optimization
lr = 0.05        # The learning rate
n_samples = 128  # The mini-batch size


# In[10]:


for split_id in range(n_splits):
    print("Loading split {} of {} dataset".format(split_id, dataset))
    # Load the dataset
    saved_dir = os.path.join(out_dir, str(split_id))
    X_train, y_train, X_test, y_test = util.load_uci_data(
            data_dir, split_id, dataset)
    X_train_, y_train_, X_test_, y_test_, y_mean, y_std = normalize_data(
            X_train, y_train, X_test, y_test)
    x_min, x_max = get_input_range(X_train_, X_test_)
    input_dim, output_dim = int(X_train.shape[-1]), 1
    
    # Initialize the measurement set generator
    rand_generator = MeasureSetGenerator(X_train_, x_min, x_max, 0.7)
    
    # Initialize the mean and covariance function of the target hierarchical GP prior
    mean = mean_functions.Zero()
    
    lengthscale = math.sqrt(2. * input_dim)
    variance = 1.
    kernel = kernels.RBF(input_dim=input_dim,
                         lengthscales=torch.tensor([lengthscale], dtype=torch.double),
                         variance=torch.tensor([variance], dtype=torch.double), ARD=True)

    # Place hyper-priors on lengthscales and variances
    kernel.lengthscales.prior = priors.LogNormal(
            torch.ones([input_dim]) * math.log(lengthscale),
            torch.ones([input_dim]) * 1.)
    kernel.variance.prior = priors.LogNormal(
            torch.ones([1]) * 0.1,
            torch.ones([1]) * 1.)
        
    # Initialize the GP model
    gp = GPR(X=torch.from_numpy(X_train_), Y=torch.from_numpy(y_train_).reshape([-1, 1]),
             kern=kernel, mean_function=mean)
    gp.likelihood.variance.set(noise_var)
    
    # Initialize tunable MLP prior
    hidden_dims = [n_units] * n_hidden
    mlp_reparam = GaussianMLPReparameterization(input_dim, output_dim,
        hidden_dims, activation_fn, scaled_variance=True)
    
    mapper = MapperWasserstein(gp, mlp_reparam, rand_generator, out_dir=saved_dir,
                               output_dim=output_dim, n_data=100,
                               wasserstein_steps=(0, 200),
                               wasserstein_lr=0.02,
                               logger=None, wasserstein_thres=0.1,
                               n_gpu=0, gpu_gp=False)
    
    w_hist = mapper.optimize(num_iters=num_iters, n_samples=n_samples,
                             lr=lr, print_every=10, save_ckpt_every=10, debug=True)
    path = os.path.join(saved_dir, "wsr_values.log")
    np.savetxt(path, w_hist, fmt='%.6e')
    print("----" * 20)


# In[28]:


# Visualize the convergence
wdist_data = []
for i in range(0, n_splits):
    wdist_file = os.path.join(out_dir, str(i), "wsr_values.log")
    wdist_data.append(np.loadtxt(wdist_file))

wdist_vals = np.stack(wdist_data)
x = np.arange(wdist_vals.shape[1])
mean = wdist_vals.mean(0)
std = wdist_vals.std(0)

fig = plt.figure(figsize=(6, 3))
plt.plot(x[::2], mean[::2], "-ok", ms=2)
plt.fill_between(x[::2], mean[::2] - std[::2],
                 mean[::2] + std[::2], alpha=0.18, color="k")
plt.xlabel("Iteration")
plt.ylabel(r"$W_1(p_{gp}, p_{nn})$")
plt.show()


# ## 1.2 Posterior Inference

# In[29]:


# Configure the SGHMC sampler
sampling_configs = {
    "batch_size": 32,
    "num_samples": 40,
    "n_discarded": 10,
    "num_burn_in_steps": 2000,
    "keep_every": 2000,
    "lr": 1e-2,
    "num_chains": 4,
    "mdecay": 1e-2,
    "print_every_n_samples": 5
}


# In[41]:


results = {"rmse": [], "nll": []}

for split_id in range(n_splits):
    print("Loading split {} of {} dataset".format(split_id, dataset))
    saved_dir = os.path.join(out_dir, str(split_id))
    
    # Load the dataset
    X_train, y_train, X_test, y_test = util.load_uci_data(
            data_dir, split_id, dataset)
    input_dim, output_dim = int(X_train.shape[-1]), 1
    
    # Initialize the neural network and likelihood modules
    net = MLP(input_dim, output_dim, [n_units] * n_hidden, activation_fn)
    likelihood = LikGaussian(noise_var)
    
    # Load the optimized prior
    ckpt_path = os.path.join(out_dir, str(split_id), "ckpts", "it-{}.ckpt".format(num_iters))
    prior = OptimGaussianPrior(ckpt_path)
    
    # Initialize bayesian neural network with SGHMC sampler
    saved_dir = os.path.join(out_dir, str(split_id))
    bayes_net = RegressionNet(net, likelihood, prior, saved_dir, n_gpu=0)
    
    # Start sampling
    bayes_net.sample_multi_chains(X_train, y_train, **sampling_configs)
    pred_mean, pred_var, preds, raw_preds = bayes_net.predict(X_test, True, True)
    r_hat = compute_rhat_regression(raw_preds, sampling_configs["num_chains"])
    print("R-hat: mean {:.4f} std {:.4f}".format(float(r_hat.mean()), float(r_hat.std())))

    rmse = uncertainty_metrics.rmse(pred_mean, y_test)
    nll = uncertainty_metrics.gaussian_nll(y_test, pred_mean, pred_var)
    print("> RMSE = {:.4f} | NLL = {:.4f}".format(rmse, nll))
    results['rmse'].append(rmse)
    results['nll'].append(nll)


# In[45]:


result_df = pd.DataFrame(results)
result_df.to_csv(os.path.join(out_dir, "optim_results.csv"), sep="\t", index=False)


# In[46]:


print("Final results")
print("> RMSE: mean {:.4e}; std {:.4e} | NLL: mean {:.4e} std {:.4e}".format(
        float(result_df['rmse'].mean()), float(result_df['rmse'].std()),
        float(result_df['nll'].mean()), float(result_df['nll'].std())))


# # 2. Standard Gaussian Prior

# In[47]:


results = {"rmse": [], "nll": []}

for split_id in range(n_splits):
    print("Loading split {} of {} dataset".format(split_id, dataset))
    saved_dir = os.path.join(out_dir, str(split_id))
    
    # Load the dataset
    X_train, y_train, X_test, y_test = util.load_uci_data(
            data_dir, split_id, dataset)
    input_dim, output_dim = int(X_train.shape[-1]), 1
    
    # Initialize the neural network and likelihood modules
    net = MLP(input_dim, output_dim, [n_units] * n_hidden, activation_fn)
    likelihood = LikGaussian(noise_var)
    
    # Initialize the standard gaussian prior
    prior = FixedGaussianPrior(std=1.0)
    
    # Initialize bayesian neural network with SGHMC sampler
    saved_dir = os.path.join(out_dir, str(split_id))
    bayes_net = RegressionNet(net, likelihood, prior, saved_dir, n_gpu=0)
    
    # Start sampling
    bayes_net.sample_multi_chains(X_train, y_train, **sampling_configs)
    pred_mean, pred_var, preds, raw_preds = bayes_net.predict(X_test, True, True)
    r_hat = compute_rhat_regression(raw_preds, sampling_configs["num_chains"])
    print("R-hat: mean {:.4f} std {:.4f}".format(float(r_hat.mean()), float(r_hat.std())))

    rmse = uncertainty_metrics.rmse(pred_mean, y_test)
    nll = uncertainty_metrics.gaussian_nll(y_test, pred_mean, pred_var)
    print("> RMSE = {:.4f} | NLL = {:.4f}".format(rmse, nll))
    results['rmse'].append(rmse)
    results['nll'].append(nll)


# In[48]:


result_df = pd.DataFrame(results)
result_df.to_csv(os.path.join(out_dir, "std_results.csv"), sep="\t", index=False)


# In[49]:


print("Final results")
print("> RMSE: mean {:.4e}; std {:.4e} | NLL: mean {:.4e} std {:.4e}".format(
        float(result_df['rmse'].mean()), float(result_df['rmse'].std()),
        float(result_df['nll'].mean()), float(result_df['nll'].std())))

