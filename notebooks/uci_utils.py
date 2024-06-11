import torch
import pandas as pd
import numpy as np
import math
import matplotlib.pylab as plt
import matplotlib as mpl
import platform

import os

os.chdir("..")

from optbnn.gp.models.gpr import GPR
from optbnn.gp import kernels, mean_functions, priors
from optbnn.bnn.reparam_nets import GaussianMLPReparameterization
from optbnn.prior_mappers.wasserstein_mapper import MapperWasserstein, WassersteinDistance
from optbnn.utils.rand_generators import MeasureSetGenerator, GridGenerator
from optbnn.utils.normalization import normalize_data
from optbnn.utils.exp_utils import get_input_range
from optbnn.utils import util

mpl.rcParams['figure.dpi'] = 100

SEED = 123
util.set_seed(SEED)

device = torch.device('cpu')

GP_SETTING = {
    'boston': {
        'default_lengthscale': 2.35,
        'lengthscale': torch.tensor([[3.5353, 7.3788, 8.8827, 9.4492, 0.5933, 2.9849, 3.3829, 2.8475, 2.2092, 1.0507, 5.4306, 6.9466, 1.2560]]),
        'noise': torch.tensor([0.0372])
    },
    'concrete': {
        'default_lengthscale': 0.477,
        'lengthscale': torch.tensor([[1.8192, 2.5673, 1.8738, 1.1345, 1.0913, 0.8679, 1.1286, 0.2449]]),
        'noise': torch.tensor([0.0175])
    },
    'energy': {
        'default_lengthscale': 1.9,
        'lengthscale': torch.tensor([[1.8874, 2.2916, 0.9664, 2.6369, 2.7830, 4.8407, 2.0737, 4.3158]]),
        'noise': torch.tensor([0.0014])
    },
    'kin8nm': {
        'default_lengthscale': 2.35,
        'lengthscale': torch.tensor([[2.9203, 2.5037, 1.5968, 1.7356, 1.5186, 1.4593, 1.5254, 1.8031]]),
        'noise': torch.tensor([0.0295])
    }
}

DATA_SETTING = {
    'boston': {'len_train_dataset': 448},
    'concrete': {'len_train_dataset': 896},
    'energy': {'len_train_dataset': 640},
    'kin8nm': {'len_train_dataset': 7424}
}


def runner(dataset_name, debug, out_dir, basis_dir):
    util.ensure_dir(out_dir)

    # Network architecture
    width = 64              # Number of units in each hidden layer
    depth = 2               # Number of hidden layers
    transfer_fn = "tanh"    # Activation function
    con = 3
    nr_seeds = 1

    # Optimization
    n_meas_set = 100  # number of points in measurement set
    if debug:
        mapper_num_iters = 2
        n_samples = 10
    else:
        mapper_num_iters = 200
        n_samples = 128

    params_mapper = {'wasserstein_steps': (0, 200), 'wasserstein_lr': 0.02, 'n_data': n_meas_set, 'n_gpu': 0,
                     'gpu_gp': False, 'logger': None, 'wasserstein_thres': 0.1}

    params_opt = {'num_iters': mapper_num_iters, 'n_samples': n_samples, 'lr': 0.05, 'save_ckpt_every': 10,
                  'print_every': 10, 'debug': True}

    # 1. Optimized Gaussian Prior

    for random_seed in range(nr_seeds):
        # save tag
        tag = f'cmf_[{depth}x{width}]_{transfer_fn}_con={con}_iters={mapper_num_iters}_rbf_ls=' \
              f'{GP_SETTING[dataset_name]["default_lengthscale"]}_seed={random_seed}'
        save_dir = os.path.join(out_dir, tag)

        if not os.path.exists(os.path.join(out_dir, f"{tag}.ckpt")):
            folder_root = f"{basis_dir}{os.sep}data{os.sep}{dataset_name}"
            data_root = f"{folder_root}{os.sep}data.txt.gz"
            if os.path.exists(data_root):
                data = np.loadtxt(data_root)
                data = torch.from_numpy(data).to(torch.float32)
                inputs, outputs = data[:, :-1], data[:, -1:]
            else:
                raise ValueError

            indices_root = f"{folder_root}{os.sep}train_indices_size={DATA_SETTING[dataset_name]['len_train_dataset']}_seed={random_seed}.txt.gz"
            if os.path.exists(indices_root):
                indices = np.loadtxt(indices_root).astype(int)
                inputs, outputs = inputs[indices], outputs[indices]
            else:
                raise ValueError

            X_train_, y_train_, y_mean, y_std = normalize_data(inputs.detach().numpy(), outputs.detach().numpy())
            x_min, x_max = get_input_range(X_train_, X_train_)
            input_dim, output_dim = int(X_train_.shape[-1]), 1

            # Initialize the measurement set generator
            rand_generator = MeasureSetGenerator(X_train_, x_min, x_max, 0.7)

            # Initialize the mean and covariance function of the target hierarchical GP prior
            mean = mean_functions.Zero()

            variance = 1.
            kernel = kernels.RBF(input_dim=input_dim,
                                 lengthscales=GP_SETTING[dataset_name]["lengthscale"].squeeze(),
                                 variance=torch.tensor([variance], dtype=torch.double), ARD=True)

            # Initialize the GP model
            gp = GPR(X=torch.from_numpy(X_train_), Y=torch.from_numpy(y_train_).reshape([-1, 1]),
                     kern=kernel, mean_function=mean)
            gp.likelihood.variance.set(GP_SETTING[dataset_name]["noise"].squeeze())

            # Initialize tunable MLP prior
            if con == 0:
                prior_per = 'parameter'
            elif con == 3:
                prior_per = 'layer'
            else:
                raise NotImplementedError

            hidden_dims = [width] * depth
            mlp_reparam = GaussianMLPReparameterization(input_dim=input_dim, output_dim=output_dim, hidden_dims=hidden_dims,
                                                        activation_fn=transfer_fn, prior_per=prior_per)

            mapper = MapperWasserstein(gp, mlp_reparam, rand_generator, out_dir=save_dir,
                                       output_dim=output_dim, **params_mapper)

            w_hist = mapper.optimize(**params_opt)
            path = os.path.join(save_dir, "wsr_values.log")
            np.savetxt(path, w_hist, fmt='%.6e')

            ckpt_path = os.path.join(save_dir, "ckpts", "it-{}.ckpt".format(mapper_num_iters))
            torch.save(torch.load(ckpt_path), os.path.join(out_dir, f"{tag}.ckpt"))

            print("----" * 20)

    # Visualize the convergence
    if platform.system() == "Windows" and debug:
        wdist_data = []
        for i in range(0, nr_seeds):
            wdist_file = os.path.join(save_dir, "wsr_values.log")
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

