import os
import time

import numpy as np


import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.autograd import Variable
from torch.autograd import grad as torch_grad

# fid_pytorch, inception
import metrics.fid_pytorch as fid_pytorch
from metrics.inception import InceptionV3
import torch

def wasserstein(true_data,fake_data,loss_type):
    if loss_type=='discriminator':
        return -true_data.mean() + fake_data.mean()
    else:
        return -fake_data.mean()
def logistic(true_data,fake_data,loss_type):
    if loss_type =='discriminator':
        loss = torch.nn.BCEWithLogitsLoss()(true_data, torch.ones(true_data.shape[0]).to(true_data.device)) + \
                    torch.nn.BCEWithLogitsLoss()(fake_data, torch.zeros(fake_data.shape[0]).to(fake_data.device))
        return loss
    else:
        loss = torch.nn.BCEWithLogitsLoss()(fake_data, torch.ones(fake_data.shape[0]).to(fake_data.device))
        return loss

def kale(true_data,fake_data,loss_type):
    if loss_type=='discriminator':
        return  true_data.mean() + torch.exp(-fake_data).mean()  - 1
    else:
        return -true_data.mean() - torch.exp(-fake_data).mean()  + 1


# calculates regularization penalty term for learning
def penalty_d(args, d, true_data, fake_data, device):
    penalty = 0.
    len_params = 0.
    # no penalty
    if args.penalty_type == 'none':
        pass
    # L2 regularization only
    elif args.penalty_type=='l2':
        for params in d.parameters():
            penalty += torch.sum(params**2)
    # gradient penalty only
    elif args.penalty_type=='gradient':
        penalty = _gradient_penalty(d, true_data, fake_data, device)
    # L2 + gradient penalty
    elif args.penalty_type=='gradient_l2':
        for params in d.parameters():
            penalty += torch.sum(params**2)
            len_params += np.sum(np.array(list(params.shape)))
        penalty = penalty/len_params
        g_penalty = _gradient_penalty(d, true_data, fake_data, device)
        penalty += g_penalty
    return penalty

# helper function to calculate gradient penalty
# adapted from https://github.com/EmilienDupont/wgan-gp/blob/master/training.py
def _gradient_penalty(d, true_data, fake_data, device):
    batch_size = true_data.size()[0]
    size_inter = min(batch_size,fake_data.size()[0])
    # Calculate interpolation
    alpha = torch.rand(size_inter,1,1,1)
    alpha = alpha.expand_as(true_data)
    alpha = alpha.to(device)
    
    interpolated = alpha*true_data.data[:size_inter] + (1-alpha)*fake_data.data[:size_inter]
    #interpolated = torch.cat([true_data.data,fake_data.data],dim=0)
    interpolated = Variable(interpolated, requires_grad=True).to(device)

    # Calculate probability of interpolated examples
    prob_interpolated = d(interpolated)

    # Calculate gradients of probabilities with respect to examples
    gradients = torch_grad(outputs=prob_interpolated, inputs=interpolated,
                           grad_outputs=torch.ones(prob_interpolated.size()).to(device),
                           create_graph=True, retain_graph=True)[0]

    # Gradients have shape (batch_size, num_channels, img_width, img_height),
    # so flatten to easily take norm per example in batch
    gradients = gradients.view(batch_size, -1)

    # Derivatives of the gradient close to 0 can cause problems because of
    # the square root, so manually calculate norm and add epsilon
    gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
    return ((gradients_norm - 1) ** 2).mean()



# get posterior samples using MLE information learned by the GAN
def sample_posterior(prior_z, g=None, h=None, device=None, T=100, extract_every=0, gamma=1e-2, kappa=4e-2):
    # total number of steps necessary to get that many samples
    sampler = torch.distributions.Normal(torch.zeros_like(prior_z).to(device), 1)
    t_extract_list = []
    Z_extract_list = []
    
    # don't use batchnorm
    h.eval()
    g.eval()

    def U_potential(z, h, g):
        return 1/2 * torch.norm(z, dim=1) ** 2 + h(g(z))

    Z_t = prior_z.clone().detach()
    t_extract_list.append(0)
    Z_extract_list.append(Z_t)

    # langevin monte carlo
    V_t = torch.zeros_like(Z_t)
    C = np.exp(-kappa * gamma)
    D = np.sqrt(1 - np.exp(-2 * kappa * gamma))
    for t in range(1, T+1):
        # reset computation graph
        Z_t.detach_()
        V_t.detach_()
        Z_half = Z_t + gamma / 2 * V_t
        Z_half.requires_grad_()
        # calculate potentials and derivatives
        U = U_potential(Z_half, h, g).sum()
        U.backward()
        dUdZ = Z_half.grad
        # update values
        V_half = V_t - gamma / 2 * dUdZ
        V_tilde = C * V_half + D * sampler.sample()
        V_t = V_tilde - gamma / 2 * dUdZ
        Z_t = Z_half + gamma / 2 * V_t

        # only if extracting the samples so we have a sequence of samples
        if extract_every != 0 and t % extract_every == 0:
            t_extract_list.append(t)
            Z_extract_list.append(Z_t.clone().detach().cpu())

            
    #h.train()
    #g.train()

    if extract_every == 0:
        return Z_t.clone().detach()
    return t_extract_list, Z_extract_list


# get posterior samples using MLE information learned by the GAN
def sample_posterior(prior_z, g=None, h=None, device=None, T=100, extract_every=0, gamma=1e-2, kappa=4e-2):
    # total number of steps necessary to get that many samples
    sampler = torch.distributions.Normal(torch.zeros_like(prior_z).to(device), 1)
    t_extract_list = []
    Z_extract_list = []
    
    # don't use batchnorm
    h.eval()
    g.eval()

    def U_potential(z, h, g):
        return 1/2 * torch.norm(z, dim=1) ** 2 + h(g(z))

    Z_t = prior_z.clone().detach()
    t_extract_list.append(0)
    Z_extract_list.append(Z_t)

    # langevin monte carlo
    V_t = torch.zeros_like(Z_t)
    C = np.exp(-kappa * gamma)
    D = np.sqrt(1 - np.exp(-2 * kappa * gamma))
    for t in range(1, T+1):
        # reset computation graph
        Z_t.detach_()
        V_t.detach_()
        Z_half = Z_t + gamma / 2 * V_t
        Z_half.requires_grad_()
        # calculate potentials and derivatives
        U = U_potential(Z_half, h, g).sum()
        U.backward()
        dUdZ = Z_half.grad
        # update values
        V_half = V_t - gamma / 2 * dUdZ
        V_tilde = C * V_half + D * sampler.sample()
        V_t = V_tilde - gamma / 2 * dUdZ
        Z_t = Z_half + gamma / 2 * V_t

        # only if extracting the samples so we have a sequence of samples
        if extract_every != 0 and t % extract_every == 0:
            t_extract_list.append(t)
            Z_extract_list.append(Z_t.clone().detach().cpu())

            
    #h.train()
    #g.train()

    if extract_every == 0:
        return Z_t.clone().detach()
    return t_extract_list, Z_extract_list


# compute the FID score
def compute_fid(args, device, images, fid_model, train_loader=None, test_loader=None):
    print('==> Computing FID')

    mu2, sigma2 = fid_pytorch.compute_stats(images, fid_model, device, args.b_size )
    try:
        mu1_train, sigma1_train = get_fid_stats(args.dataset+'_train')
    except:
        mu1_train, sigma1_train = compute_inception_stats(args, device, fid_model, 'train', train_loader)
    fid_train = fid_pytorch.calculate_frechet_distance(mu1_train, sigma1_train, mu2, sigma2)

    if test_loader is not None:
        try:
            mu1_valid, sigma1_valid = get_fid_stats(args.dataset+'_valid')
        except:
            mu1_valid, sigma1_valid = compute_inception_stats(args, device, fid_model, 'valid', test_loader)
        fid_valid = fid_pytorch.calculate_frechet_distance(mu1_valid, sigma1_valid, mu2, sigma2)

        return fid_train,fid_valid
        
    raise NotImplementedError()


def compute_inception_stats(args, device, fid_model, type_dataset, data_loader):
    mu_train,sigma_train= fid_pytorch.compute_stats_from_loader(fid_model,data_loader, device)
    #mu_1,sigma_1 = get_fid_stats(self.args.dataset+'_'+type_dataset)
    path = 'metrics/res/stats_pytorch/fid_stats_'+args.dataset+'_'+type_dataset+'.npz'
    np.savez(path, mu=mu_train, sigma=sigma_train)
    #if self.test_loader is not None:
    #   mu_test,sigma_test= fid_pytorch.compute_stats_from_loader(self.fid_model,self.test_loader,self.device)
    return mu_train,sigma_train



def get_fid_stats(dataset):
    if dataset=='cifar10_train':
        path = 'metrics/res/stats_pytorch/fid_stats_cifar10_train.npz'
    elif dataset=='cifar10_valid':
        path = 'metrics/res/stats_pytorch/fid_stats_cifar10_valid.npz'
    elif dataset=='imagenet_train':
        path = 'metrics/res/stats_pytorch/fid_stats_imagenet_train.npz'
    elif dataset=='imagenet_train':
        path = 'metrics/res/stats_pytorch/fid_stats_imagenet_valid.npz'
    elif dataset=='celeba':
        path = 'metrics/res/stats_pytorch/fid_stats_celeba.npz'

    f = np.load(path)
    mu1, sigma1 = f['mu'][:], f['sigma'][:]
    f.close()       
    return mu1, sigma1
