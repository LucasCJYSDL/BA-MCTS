import math
import torch
import torch.nn as nn
import numpy as np
from torch.distributions import constraints
from torch.distributions.transforms import Transform
from torch.nn.functional import softplus

class GaussianMSELoss(nn.Module):

    def __init__(self):
        super(GaussianMSELoss, self).__init__()

    def forward(self, mu_logvar, target, logvar_loss = True):
        mu, logvar = mu_logvar.chunk(2, dim=1)
        inv_var = (-logvar).exp()
        if logvar_loss:
            return (logvar + (target - mu)**2 * inv_var).mean()
        else:
            return ((target - mu)**2).mean()

def get_trunc_normal_std(input_dim):
    """
    Returns the truncated normal standard deviation required for weight initialization
    """
    return 1 / (2 * np.sqrt(input_dim))

def reinitialize_fc_layer_(fc_layer):
    """
    Helper function to initialize a fc layer to have a truncated normal over the weights, and zero over the biases
    """
    input_dim = fc_layer.weight.shape[1]
    std = get_trunc_normal_std(input_dim)
    torch.nn.init.trunc_normal_(fc_layer.weight, std=std, a=-2 * std, b=2 * std)
    torch.nn.init.zeros_(fc_layer.bias)

def get_weight_bias_parameters_with_decays(fc_layer, decay):
    """
    For the fc_layer, extract only the weight from the .parameters() method so we don't regularize the bias terms
    """
    decay_params = []
    non_decay_params = []
    for name, parameter in fc_layer.named_parameters():
        if 'weight' in name:
            decay_params.append(parameter)
        elif 'bias' in name:
            non_decay_params.append(parameter)

    decay_dicts = [{'params': decay_params, 'weight_decay': decay}, {'params': non_decay_params, 'weight_decay': 0.}]

    return decay_dicts

# Taken from: https://github.com/pytorch/pytorch/pull/19785/files
# The composition of affine + sigmoid + affine transforms is unstable numerically
# tanh transform is (2 * sigmoid(2x) - 1)
# Old Code Below:
# transforms = [AffineTransform(loc=0, scale=2), SigmoidTransform(), AffineTransform(loc=-1, scale=2)]
class TanhTransform(Transform):
    r"""
    Transform via the mapping :math:`y = \tanh(x)`.
    It is equivalent to
    ```
    ComposeTransform([AffineTransform(0., 2.), SigmoidTransform(), AffineTransform(-1., 2.)])
    ```
    However this might not be numerically stable, thus it is recommended to use `TanhTransform`
    instead.
    Note that one should use `cache_size=1` when it comes to `NaN/Inf` values.
    """
    domain = constraints.real
    codomain = constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        # We do not clamp to the boundary here as it may degrade the performance of certain algorithms.
        # one should use `cache_size=1` instead
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        # We use a formula that is more numerically stable, see details in the following link
        # https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/bijectors/tanh.py#L69-L80
        return 2. * (math.log(2.) - x - softplus(-2. * x))