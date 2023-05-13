import enum

import torch
from typing import Union, Optional

from pytorch3d.ops import utils as oputil
from torch import optim

from chsel.types import ICPSolution, SimilarityTransform
from chsel.costs import VolumetricCost
from pytorch_kinematics.transforms import matrix_to_rotation_6d, rotation_6d_to_matrix

from chsel_experiments.svgd import RBF, SVGD
import logging

from chsel.registration_util import plot_poke_losses, apply_init_transform

logger = logging.getLogger(__name__)


class Optimization(enum.Enum):
    SGD = 0
    CMAES = 1
    CMAME = 2
    SVGD = 3
    CMAMEGA = 4


class CostProb:
    def __init__(self, cost, scale=1.):
        self.cost = cost
        self.scale = scale

    def log_prob(self, X, s):
        # turn into R, T
        q = X[:, :6]
        T = X[:, 6:9]
        R = rotation_6d_to_matrix(q)
        c = self.cost(R, T, s)
        # p = N exp(-c * self.scale)
        # logp \propto -c * self.scale
        return -c * self.scale


def iterative_closest_point_volumetric_svgd(
        volumetric_cost: VolumetricCost,
        X: Union[torch.Tensor, "Pointclouds"],
        init_transform: Optional[SimilarityTransform] = None,
        max_iterations: int = 300,
        lr=0.005,
        kernel_scale=0.01,  # None indicates to use the median heuristic
        cost_scale=5.,
        save_loss_plot=True,
) -> ICPSolution:
    # make sure we convert input Pointclouds structures to
    # padded tensors of shape (N, P, 3)
    Xt, num_points_X = oputil.convert_pointclouds_to_tensor(X)
    Xt, R, T, s = apply_init_transform(Xt, init_transform)

    converged = False

    # initialize the transformation history
    t_history = []
    losses = []

    # SVGD
    K = RBF(kernel_scale)

    q = matrix_to_rotation_6d(R)
    params = torch.cat([q, T], dim=1)
    svgd = SVGD(CostProb(volumetric_cost, scale=cost_scale), K, optim.Adam([params], lr=lr))
    for i in range(max_iterations):
        # convert back to R, T, s
        logprob = svgd.step(params, s)
        cost = -logprob / svgd.P.scale
        losses.append(cost)

    # convert ES back to R, T
    q = params[:, :6]
    T = params[:, 6:9]
    R = rotation_6d_to_matrix(q)
    rmse = volumetric_cost(R, T, s)

    if save_loss_plot:
        plot_poke_losses(losses)

    if oputil.is_pointclouds(X):
        Xt = X.update_padded(Xt)  # type: ignore

    return ICPSolution(converged, rmse, Xt, SimilarityTransform(R, T, s), t_history)


