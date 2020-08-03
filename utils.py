"""Utility file for custom funcs/objects
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
import imageio
import seaborn as sns

from torch.distributions import MultivariateNormal
from models import GMM
from matplotlib import cm

MIN, MAX = -10, 10
sns.set_style('white')

class EarlyStopping:
    """Early stops if validation loss doesn't improve after a given patience.

        From https://github.com/Bjarten/early-stopping-pytorch
    """
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time
                            validation loss improved. Default: 7
            verbose (bool): If True, prints a message for each
                            validation loss improvement. Default: False
            delta (float): Minimum change in the monitored quantity
                           to qualify as an improvement. Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            #self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            #self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss {val_loss:.6f}.  Saving model ...')
        torch.save(model.state_dict(), 'checkpoint.pt')
        self.val_loss_min = val_loss


def normed_logprob_gradient_score(dataloader, model, estimator):
    """Returns the OOD score function from the paper
    """
    scores = []
    
    model.eval()
    for x, _ in dataloader:
        # require gradient of input
        x.requires_grad = True

        # forward pass
        outs = model(x)
        logprobs = estimator(outs)

        # get the logprob gradients
        x.retain_grad()
        logprobs.backward()
        scores.extend(-torch.norm(x.grad, dim=1).detach().numpy())

    return scores


def print_stats(tensor):
    """Prints various statistics of a tensor"""
    print(f"shape: {tensor.size()}" +
          f" | min/max: {tensor.min():.3f}/{tensor.max():.3f}" +
          f" | mean: {tensor.mean():.3f} | stddev: {tensor.std():.3f}")


def plot_density(samples, filename=None):
    """Plots probability density of generated samples

    Args:
        samples
    """
    fig, ax = plt.subplots()
    data = [list(l) for l in zip(*samples)]
    
    # plot parameters
    plt.xlim(MIN, MAX)
    plt.ylim(MIN, MAX)
    plt.title("Estimated Probability Density")
    plt.xlabel("x")
    plt.ylabel("y")
 
    sns.kdeplot(*data, shade=True, cmap="Blues", ax=ax)
    ax.collections[0].set_alpha(0)

    if filename is None:
        plt.show()
    else:
        plt.savefig(filename, dpi=300)
    plt.close()


def energy(x):
    """Returns numpy array of energy function outputs
    """
    return -torch.logsumexp(x, axis=1).detach().numpy()


def model_energies(model, n, bmin=-1, bmax=1):
    """Computes energies in a grid using a trained model
    """
    x = torch.linspace(bmax, bmin, n)
    y = torch.linspace(bmin, bmax, n)
    points = torch.stack(torch.meshgrid(x, y), dim=-1)

    # fill the density map
    energy = [model.energy(pts).detach().numpy() for pts in points]
    return np.array(energy, dtype=np.float32)


def density_map(model, 
                n, samples,
                bmin=-1, bmax=1,
                filename="images/density/dplot.jpg"):
    """Plots loglikehood heatmap of learned model

    Args:
        model (nn.Module): learned model
        bmin (float): min x/y value
        bmax (float): max x/y value
        n (int): number of points to plot sqrted
        samples (int): number of samples from prior
    """
    x = torch.linspace(bmax, bmin, n)
    y = torch.linspace(bmin, bmax, n)
    points = torch.stack(torch.meshgrid(x, y), dim=-1)

    # fill the density map
    density = [model.likelihood(pts, samples).detach().numpy() for pts in points]
    density = np.array(density, dtype=np.float32)

    # plot density map
    plt.figure()
    plt.imshow(density, extent=[bmin, bmax, bmin, bmax])
    plt.savefig(filename, dpi=300)


def plot_loss(losses, filename=None):
    """Plot losses from an array
    """
    plt.figure()
    plt.title("Loss Curve")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.plot(losses)

    if filename is not None:
        plt.savefig(filename, dpi=300)
    else:
        plt.show()


def energy_map(model, bmin, bmax, n, image=True):
    """Creates heatmap of energy function for 2D data

    Args:
        model (nn.Module): pytorch model
        bmin (float): min x/y value
        bmax (float): max x/y value
        n (float): number of points to interpolate in [bmin, bmax]
        image (bool): return an image for creating gifs
    """
    # compute the energies 
    logits = [model(torch.Tensor([[x, y] for y in np.linspace(bmin, bmax, n)]))
              for x in np.linspace(bmin, bmax, n)]
    energies = np.array([energy(l) for l in logits])

    if image:
        emin, emax = energies.min(), energies.max()
        return np.uint8(255 * cm.viridis((energies - emin) / (emax - emin)))
    return energies


def energy_plot(energies, bmin=-1, bmax=1, filename=None):
    """Saves a heatmap of the energy function

    Args:
        filename (str): filename for saving the plot
    """
    fig = plt.figure()
    plt.imshow(energies, extent=[bmin, bmax, bmin, bmax])
    plt.colorbar()
    plt.savefig(filename, dpi=300, bbox_inches='tight')


def makegif(gif_name, data, **kwargs):
    """Saves a GIF file
    """
    imageio.mimwrite(gif_name, data, **kwargs)

