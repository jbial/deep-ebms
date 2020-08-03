"""
Script to train a VAE on the toy dataset
"""
import os
import shutil
import torch
import torch.nn.functional as F
import torch.optim as optim
import argparse

from models import VAE
from torch.utils.data import DataLoader
from dataloader import *
from utils import *


def vae_loss(reconstructions, x, mu, logvar):
    """Defines the VAE loss function (ELBO)

    Args:
        reconstructions (tensor): VAE reconstructions
        x (tensor): i.i.d. samples from dataset
        mu (tensor): mean vector from VAE encoder
        logvar (tensor): log variance from VAE encoder
    """
    MSE = F.mse_loss(reconstructions, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return MSE + KLD


def vae_train(model, dataloader, params):
    """Trains a VAE by maximizing the ELBO

    Args:
        model (nn.Module): VAE model with encdec functions
        lr (float): learning rate
        epochs (int): Number of epochs to train for
        dataloader (Dataloader): torch dataloader containing train data
    """
    optimizer = optim.Adam(model.parameters(), lr=params["lr"])

    # for saving samples throughout training
    if os.path.exists(params["img_dir"]):
        shutil.rmtree(params["img_dir"])
    os.makedirs(params["img_dir"])

    # set model in training mode
    print("====== Begin training ======")
    model.train()
    for i in range(params["epochs"]):
        epoch_loss = 0
        for x, _ in dataloader:
            optimizer.zero_grad()

            # forward pass
            recon, mu, logvar = model(x)
            loss = vae_loss(recon, x, mu, logvar)

            # backward pass
            loss.backward()
            optimizer.step()

            # print statistics
            epoch_loss += loss.item()

        # sample from the model to track the progress
        samples = model.sample(params["samples"])
        if params["standardize"]:
            samples = dataloader.dataset.renorm(samples)
        plot_density(samples, filename=f"{params['img_dir']}/{i:04d}.png")

        print(f"Epoch: {i + 1:2d} | Loss: {epoch_loss/len(dataloader): 10.3f}")

    # make the gif and clear the training image directory
    os.system(f'convert -delay 10 -loop 0 \
              {params["img_dir"]}/????.png images/gifs/vae.gif')
    shutil.rmtree(params["img_dir"])

    print("======= End training =======")


def vae_sample(model, dataloader, params):
    """Samples from the learned model and creates a density plot
    """
    samples = model.sample(params["samples"])
    if params["standardize"]:
        samples = dataloader.dataset.renorm(samples)
    if params["density_file"]:
        plot_density(samples, filename=params["density_file"])
    else:
        plot_density(samples)
        

def main():

    # set the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of CPU workers for the dataloader")
    parser.add_argument("--batchsize", type=int, default=64,
                        help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-2,
                        help="Learning rate")
    parser.add_argument("--samples", type=int, default=1000,
                        help="Number of samples to approximate the EV")
    parser.add_argument("--zdim", type=int, default=2,
                        help="number of latent variables")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Number of training epochs")
    parser.add_argument("--dropout", type=float, default=0,
                        help="Dropout probability, must be in [0,1)")
    parser.add_argument("--hidden", type=int, default=40,
                        help="Number of hidden units")
    parser.add_argument("--seed", type=int, default=99,
                        help="Random seed")
    parser.add_argument("--train-file", type=str, 
                        default="data/mixture2d_X_train.csv",
                        help="File to load training data from")
    parser.add_argument("--standardize", action='store_true',
                        help="Standardize the input data")
    parser.add_argument("--layers", type=int, default=None,
                        help="Number of layers in the DNN")
    parser.add_argument("--density-file", type=str, default=None,
                        help="Filename for density plot")
    parser.add_argument("--img-dir", type=str, default="images/vae_samples",
                        help="Filename for density plot")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Path to save model to")
    params = vars(parser.parse_args())

    # set the random seed
    torch.manual_seed(params["seed"])

    # get the data
    trainset = GMMDataset2D(data_file=params["train_file"], 
                            standardize=params["standardize"])
    trainloader = DataLoader(trainset, 
                             batch_size=params["batchsize"],
                             num_workers=params["workers"],
                             shuffle=True)

    # create the model
    layers = [2] + [params["hidden"] // 2**i for i in range(params["layers"])]
    vae = VAE(layers, zdim=params["zdim"], dropout=params["dropout"])

    # train model
    vae_train(vae, trainloader, params)

    # visualize samples
    vae_sample(vae, trainloader, params)

    # save the model
    if params["model_path"] is not None:
        torch.save(vae.state_dict(), params["model_path"])
        print(f"Saved model at {params['model_path']}")


if __name__ == "__main__":
    main()
