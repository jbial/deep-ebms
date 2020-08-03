"""
File for different feature extraction models
"""
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from dataloader import mixture_params
from torch.distributions import MultivariateNormal


class FCNet(nn.Module):
    """Simple fully connected neural network"""

    def __init__(self, layers, num_labels, dropout=0.0):
        """
        Args:
            layers (list): list of integers denoting the layers and # of nodes
            num_labels (int): Number of classification labels
        """
        super(FCNet, self).__init__()
    
        self.fcs = [nn.Linear(*ij) for ij in zip(layers, layers[1:])]
        self.lns = [nn.LayerNorm(i) for i in layers[1:]]
        self.drops = [nn.Dropout(p=dropout) for _ in layers[1:]]
        self.last_layer = nn.Linear(layers[-1], num_labels)
        
    def forward(self, x):
        for i, (fc, ln) in enumerate(zip(self.fcs, self.lns)):
            x = self.drops[i](F.relu(ln(fc(x))))
        return self.last_layer(x)

    def energy(self, x):
        return torch.logsumexp(self.forward(x), axis=1)


class GMM():
    """Gaussian Mixture Model"""

    def __init__(self, gmm_params):
        self.gmm = gmm_params
        self.mus, self.covs = [torch.Tensor(p) for p in zip(*self.gmm)]
        self.var = gmm_params[0][1][0][0]

    def likelihood(self, x):
        pass

    def energy(self, x):
        squared_dist = [-0.5 * torch.diag((1/self.var)*(x-mu)@(x-mu).T) for mu in self.mus]
        return -torch.logsumexp(torch.stack(squared_dist), dim=0) 


class Gaussian():
    """Isotropic Gaussian distribution"""
    def __init__(self, mu):
        self.mu = torch.tensor(mu)
        
    def likelihood(self, x):
        pass

    def energy(self, x):
        return 0.5 * torch.diag((x - self.mu) @ (x - self.mu).T)


class Mish(nn.Module):     
    def __init__(self):         
        super().__init__()      
    def forward(self, x):           
        return x*( torch.tanh(torch.nn.functional.softplus(x)))


class VAE(nn.Module):
    """Vanilla VAE model"""

    def __init__(self, layers, zdim, dropout=None):
        """
        Args:
            layers (list): list of integers denoting the layers and # of nodes
            zdim (int): number of latent variables
        """
        super(VAE, self).__init__()
        self.zdim = zdim

        self.encs = [nn.Linear(*ij) for ij in zip(layers, layers[1:])]
        self.mu = nn.Linear(layers[-1], zdim)
        self.lvar = nn.Linear(layers[-1], zdim)

        self.decoder = nn.Sequential(
            nn.Linear(2, 100),
            nn.Linear(100, 2),
            nn.Tanh()
        )

    def encode(self, x):
        for fc in self.encs:
            x = F.relu(fc(x))
        return self.mu(x), self.lvar(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode(self, z):
        return self.decoder(z)

    def likelihood(self, x, samples=100):
        """Estimates the marginal likelihood via importance sampling
        """
#        # get the proposal distribution params conditioned on x
#        mus, logvar = self.encode(x)
#        covs = torch.diag_embed(logvar.exp())
#        
#        # proposal distribution
#        proposals = MultivariateNormal(mus, covs)
#
        # prior distribution
        prior = MultivariateNormal(torch.zeros(self.zdim), torch.eye(self.zdim))
        if x.dim() > 1:
            prior = prior.expand(batch_shape=torch.Size([x.size(0)]))

        # conditional (decoder distribution) 
        zs = prior.sample(sample_shape=torch.Size([samples]))
        locs = self.decode(zs)
        conditional = MultivariateNormal(locs, 0.5 * torch.eye(x.size(-1)))

        # compute the marginal likelihood
#        lratio = prior.log_prob(zs).exp() / proposals.log_prob(zs).exp()
        return (conditional.log_prob(x).exp()).mean(dim=0)

    def sample(self, n):
        zs = torch.randn(n, self.zdim)
        return self.decode(zs)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class WideResnet(nn.Module):

    def __init__(self):
        raise NotImplementedError

