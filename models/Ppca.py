import numpy as np
import torch
import torch.nn as nn
from torch import transpose
from torch.linalg import inv



def get_ProbabilisticPCA(
    input_size,
    dataset,
    cfg,
):
    return ProbabilisticPCA(
        input_size,
        dataset,
        cfg.n_components,
        cfg.nb_sample_estimate,
        cfg.prior_sigma,
    )


def tr(x):
    return transpose(x, 0, 1)


def get_C(w, mu, sigma):
    return w @ tr(w) + sigma**2 * torch.eye(w.shape[0])


class PPCA(object):
    def __init__(self, latent=2, sigma=1.0):
        self.latent = latent
        self.prior_sigma = torch.tensor(sigma, dtype=torch.float64)
        self.distribution = None
        self.w = None
        self.mu = None
        self.sigma = None

    def get_C(self):
        return self.w @ tr(self.w) + self.sigma**2 * torch.eye(self.w.shape[0]).to(self.w.device)

    def fit(self, x, em=False):
        self.d = x.shape[1]
        [self.w, self.mu, self.sigma] = self.__fit_ml(x)

    def sample(self, nb_sample=1000):
        z = torch.randn((nb_sample, self.latent)) * self.prior_sigma
        x_c = self.mu + tr(self.w @ tr(z))
        x = x_c + torch.randn(x_c.shape) * self.sigma
        return x_c, x

    def get_marginal(self):
        C = self.get_C()
        self.distribution = torch.distributions.multivariate_normal.MultivariateNormal(
            self.mu, C
        )
        return self.distribution

    def log_prob(
        self,
        x,
    ):
        x_flatten = x.flatten(1)
        batch_size = x.shape[0]
        marginal = self.get_marginal()
        log_prob = marginal.log_prob(x_flatten)
        return log_prob

    def __fit_ml(self, x):
        mu = x.mean(0)
        A = tr(x - mu) @ (x - mu) / x.shape[0]
        [u, s, v] = torch.linalg.svd(
            tr(x - mu) / np.sqrt(x.shape[0]), full_matrices=False
        )
        s = s
        if self.latent > len(s):
            ss = torch.zeros(self.latent)
            ss[: len(s)] = s
        else:
            ss = s[: self.latent]
        if self.latent < self.d:
            sigma = torch.sqrt(
                1.0 / (self.d - self.latent) * torch.sum(s[self.latent :] ** 2)
            )
        else:
            sigma = 0.0

        ss = torch.sqrt(torch.maximum(torch.zeros(1), ss**2 - sigma**2))
        w = u[:, : self.latent] @ torch.diag(ss)
        return (w, mu, sigma)


class ProbabilisticPCA(nn.Module):
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
        n_latents=10,
        nb_sample_estimate=10000,
        prior_sigma=1.0,
        **kwargs
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.n_latents = n_latents
        self.prior_sigma = prior_sigma

        self.n_features = np.prod(input_size)
        
        data += torch.randn_like(data) * 1e-2
        data = data.flatten(1)


        self.w = torch.nn.parameter.Parameter(torch.randn(self.n_features, n_latents))
        self.mu = torch.nn.parameter.Parameter(torch.randn(self.n_features))
        self.sigma = torch.nn.parameter.Parameter(torch.tensor(1.0))

        aux_w, aux_mu, aux_sigma = self.fit_ppca(data)
        self.w.data = aux_w
        self.mu.data = aux_mu
        self.sigma.data = aux_sigma

    def fit_ppca(self, x):
        mu = x.mean(0)
        A = tr(x - mu) @ (x - mu) / x.shape[0]
        [u, s, v] = torch.linalg.svd(
            tr(x - mu) / np.sqrt(x.shape[0]), full_matrices=False
        )
        s = s
        if self.n_latents > len(s):
            ss = torch.zeros(self.n_latents)
            ss[: len(s)] = s
        else:
            ss = s[: self.n_latents]

        if self.n_latents < self.n_features:
            sigma = torch.sqrt(
                1.0
                / (self.n_features - self.n_latents)
                * torch.sum(s[self.n_latents :] ** 2)
            )
        else:
            sigma = torch.tensor(0.0)

        ss = torch.sqrt(torch.maximum(torch.zeros(1), ss**2 - sigma**2))
        w = u[:, : self.n_latents] @ torch.diag(ss)
        return (w, mu, sigma)

    def sample_simple(self, nb_sample=1):
        z = torch.randn((nb_sample, self.n_latents)).to(self.mu.device) * self.prior_sigma
        x_c = self.mu + tr(self.w @ tr(z))
        x = x_c + torch.randn(x_c.shape).to(self.mu.device) * self.sigma
        return x.reshape(nb_sample, *self.input_size)
    
    def forward(self, z):
        x_c = self.mu + tr(self.w @ tr(z))
        x = x_c + torch.randn(x_c.shape).to(self.mu.device) * self.sigma
        return x.reshape(z.shape[0], *self.input_size)

    def get_C(self):
        return self.w @ tr(self.w) + self.sigma**2 * torch.eye(self.w.shape[0]).to(self.w.device)

    def get_C_inv(self):
        return inv(self.get_C())

    def log_prob_simple(self, x):
        sample = x.flatten(1)

        C_inv = self.get_C_inv()
        log_prob = (
                -0.5 * (((sample - self.mu) @ C_inv)*(sample - self.mu)).sum(-1)
                - 0.5 * self.n_features * np.log(2 * np.pi)
                - 0.5 * torch.logdet(self.get_C())
            ).reshape(x.shape[0])
        return log_prob
