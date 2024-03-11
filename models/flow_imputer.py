import numpy as np
import torch
import torch.nn as nn
from torch import transpose as tr
from torch.linalg import inv
from .flowplusplus.models import FlowPlusPlus





class FlowPlusPLusimputer(nn.Module):
    """
    Probabilistic PCA proposal.

    Attributes:
    ----------
    input_size : tuple
        The size of the input.
    n_latents : int
        The number of latents in the probabilistic PCA model.
    nb_sample_estimate : int
        The number of samples to use to estimate the gaussian mixture model.

    Methods:
    --------
    log_prob_simple(x): compute the log probability of the proposal.
    sample_simple(nb_sample): sample from the proposal.
    """

    def __init__(
        self,
        input_size,
        data,
        **kwargs
    ) -> None:
        super().__init__()
        self.model = FlowPlusPlus(in_shape=input_size, use_attn=False, **kwargs)
        self.model.forward(data[:64], reverse=False)
        self.input_size = input_size
        self.k = 256

    # def sample_simple(self, nb_sample=1):
        # z = torch.randn((nb_sample, self.n_latents)).to(self.mu.device) * self.prior_sigma
        
    
    def forward(self, z):
        x, sjld = self.model(z, reverse=True)
        x= torch.tanh(x)
        return x.reshape(z.shape[0], *self.input_size)

    def log_prob_simple(self, x):
        sample = x.reshape(-1, *self.input_size)
        z, sldj = self.model(sample, reverse = False)
        prior_ll = -0.5 * (z ** 2 + np.log(2 * np.pi))
        if torch.any(torch.isnan(prior_ll)):
            print("prior_ll", prior_ll)
        prior_ll = prior_ll.flatten(1).sum(-1) \
            - np.log(self.k) * np.prod(z.size()[1:])
        ll = prior_ll + sldj
        # nll = -ll.mean()
        # assert False
        return ll

