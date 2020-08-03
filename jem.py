"""
File for training JEM
"""
import pdb
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse

from tqdm import tqdm
from utils import *
from models import FCNet, VAE, GMM, Gaussian
from torch.utils.data import DataLoader
from dataloader import *
from samplers import *
from experiment import *


def CDloss(sampler, n_samples):
    """Returns the log loss for the generative component

    Args:
        sampler (Sampler): model sampler object
        n_samples (int): Number of samples/steps to estimate expected value
    """
    def loss(x):
        return -(sampler.EV(n_samples) - torch.mean(torch.logsumexp(x, dim=-1)))
    return loss


def train(model, dataloader, valloader, sampler, params):
    """Train the model using the JEM framework

    Args:
        model (nn.Module): pytorch model
        dataloader (DataLoader): pytorch dataloader for the training set
        valloader (DataLoader): pytorch dataloader for the validation set
        sampler (Sampler): model sampler object
        params (dict): contains hyperparameters
    """

    # optimizer and learning rate schedule
    optimizer = optim.Adam(model.parameters(), lr=params["lr"])

    # loss functions
    xent = nn.CrossEntropyLoss()
    sampler_loss = CDloss(sampler, params["samples"])

    # early stop object
    early_stopping = EarlyStopping(patience=params["patience"])
    energies = []

    if params["ood"] is not None:
        print(f"INFO: OOD excluding cluster {params['ood']}")

    print("====== Begin training ======")
    for epoch in range(params["epochs"]):

        # model in training mode
        epoch_loss = 0
        model.train()
        for i, (x, y) in enumerate(dataloader):

            # zero the gradients
            optimizer.zero_grad()

            # compute the loss
            y_hat = model(x)
            xent_loss = xent(y_hat, y)

            # combined classifier + generative model loss
            loss = xent_loss
            if params["jem"]:
                loss -= sampler_loss(y_hat)

            # update params and lr
            loss.backward(retain_graph=True)
            optimizer.step()

            epoch_loss += loss.item()

        # test model on validation data
        if params["patience"] is not None:
            val_loss = validate(model, valloader, xent, sampler_loss, params)

            # Check for early stopping
            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                train_acc, _ = evaluate(model, dataloader)
                print(f"Early stop epoch {epoch + 1:2d}/{params['epochs']} | "+
                      f"Val Loss: {val_loss:.3f} | "+ 
                      f"Train Acc: {train_acc:.2f}")
                return train_acc, epoch + 1

        # save frames for gif of energy map
        if params["energy_gif"]:
            energies.append(energy_map(model, -10, 10, 1000))

        print(f"Epoch: {epoch + 1:2d} | Loss: {epoch_loss/len(dataloader): 10.3f}")


    if params["energy_gif"] is not None:
        makegif(params["energy_gif"], energies, fps=10)

    # evaulate the entire trainset
    acc, _ = evaluate(model, dataloader)
    print("----------------------------")
    print(f"Accuracy on train set: {acc:.2f}")
    print("----------------------------")
    print("======= End training =======")
    return acc, params["epochs"]


def max_entropy_update(k, goptimizer, generator, model, N, params):
    """Updates the explicit sampler for k steps

    Args:
        k (int): number of steps to optimize generator for
        goptimizer (torch.optim): Optimizer for the generator
        generator (nn.Module): generative model acting as the explicit sampler
        model (nn.Module): Energy-based model (classifier)
        N (int): number of samples for the monte-carlo estimates
    """
    loss, losses = 0, []

    # train the generator
    for i in range(k):
        # zero the optimizer gradients
        goptimizer.zero_grad()

        # sample from the generator
        zs = torch.randn(N, params["zdim"])
        xs = generator.decode(zs)

        # calcute the terms
        likelihood = generator.likelihood(xs)
        entropy = (likelihood @ torch.log(likelihood))
        energy = model.energy(xs).mean()
        max_entropy = 0.1 * energy - entropy
        loss += max_entropy.item()
        losses.append(max_entropy.item())

        # gradient steps
        max_entropy.backward()
        goptimizer.step()

        print(f"==> Iter: {i+1:2d} | Generator loss: {loss/k: 4.2f}", end='\r')

    return losses
        

def validate(model, valloader, criterion, jem_loss, params):
    """Validates the model on validation set
    """
    # model in validation mode
    val_loss = 0
    model.eval()
    for x, y in valloader:

        # run inference and compute the loss
        out = model(x)
        v_loss = criterion(out, y)
        if params["jem"]:
            v_loss -= jem_loss(out)

        # running average of the loss
        val_loss += v_loss.item()
    
    return val_loss / len(valloader)


def evaluate(model, dataloader, renormalization=None):
    """Run inference on the model

    Args:
        model (nn.Module): pytorch model
        dataloader (DataLoader): pytorch dataloader
        renormalization (tuple): tuple of (mean, std) for visualizing points
    """
    # set the model in evaluation mode
    model.eval()

    # evaluate
    correct = total = 0
    preds = []
    with torch.no_grad():
        for x, y in dataloader:
            # inference
            logits = model(x)
            _, predictions = torch.max(logits.data, dim=1)

            # recover the unnormalized datapoint
            sample = x.numpy()
            if renormalization is not None:
                sample = renormalization(sample)

            preds.extend(list(zip(sample, predictions.numpy())))
            total += y.size(0)
            correct += (predictions == y).sum().item()

    return 100 * correct / total, preds


def train_generator(k, model, params):
    """Train generator with fixed model
    """
    
    layers = [2] + [params["hidden"] // 2**i for i in range(params["layers"])]
    vae = VAE(layers, zdim=params["zdim"], dropout=params["dropout"])

    gopt = optim.Adam(vae.decoder.parameters(), lr=params["lr"])

    loss = max_entropy_update(50, gopt, vae, model, params["batchsize"], params)

    # plot stuff
    #energies = model_energies(model, 100)
    #energy_plot(energies, filename="images/max_entropy/gmm/energy.jpg") 
    density_map(vae, 100, 25, filename="images/max_entropy/gmm/dmap.jpg")
    plot_loss(loss, "images/max_entropy/gmm/loss.jpg")


def main():

    # set the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--label-file", type=str, default="data/color_labels.csv",
                        help="File that contains the color-label mapping")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of CPU workers for the dataloader")
    parser.add_argument("--jem", action="store_true",
                        help="Include the JEM loss")
    parser.add_argument("--batchsize", type=int, default=128,
                        help="Batch size for training")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate")
    parser.add_argument("--samples", type=int, default=50,
                        help="Number of samples to approximate the EV")
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
    parser.add_argument("--test-file", type=str, 
                        default="data/mixture2d_X_test.csv",
                        help="File to load testing data from")
    parser.add_argument("--val-file", type=str, default="data/mixture2d_X_val.csv",
                        help="File to load validation data from")
    parser.add_argument("--show-preds", action='store_true',
                        help="Show the predicted labels")
    parser.add_argument("--train-split", type=float, default=None,
                        help="The percentage of data for each cluster for training")
    parser.add_argument("--patience", type=int, default=None,
                        help="Patience for early stopping")
    parser.add_argument("--sgld", action='store_true',
                        help="Use the SGLD sampler")
    parser.add_argument("--standardize", action='store_true',
                        help="Standardize the input data")
    parser.add_argument("--eplot-file", type=str, default=None,
                        help="Filename of energy plot to save, if any")
    parser.add_argument("--ood", type=int, default=None,
                        help="Cluster index to exclude for OOD")
    parser.add_argument("--layers", type=int, default=None,
                        help="Number of layers in the DNN")
    parser.add_argument("--energy-gif", type=str, default=None,
                        help="Filename for energy plot evolution gif")
    parser.add_argument("--density-img", type=str, default=None,
                        help="Filename for density plot")
    parser.add_argument("--zdim", type=int, default=2,
                        help="number of latent variables")
    params = vars(parser.parse_args())

    # additional params
    params["sgld_step"] = 1
    params["sgld_noise"] = 0.8
    params["mixture"] = (0.7, 0.7, 1e-3)

    print("****** Hyperparameters ******")
    pprint.pprint(params)
    print("*****************************")

    # get the gmm params for visualization
    gmm_params = mixture_params(*params["mixture"])
    gmm_params = [gmm_params[i] for i in [1, -1]]

    # train
    trainset = GMMDataset2D(data_file=params["train_file"], 
                            standardize=params["standardize"],
                            split=params["train_split"],
                            ood=params["ood"])
    trainloader = DataLoader(trainset, 
                            batch_size=params["batchsize"],
                            num_workers=params["workers"],
                            shuffle=True)

    # test
    testset = GMMDataset2D(data_file=params["test_file"],
                           standardize=params["standardize"],
                           ood=params["ood"])
    testloader = DataLoader(testset, 
                            batch_size=params["batchsize"],
                            num_workers=params["workers"])

    # val
    valset = GMMDataset2D(data_file=params["val_file"],
                          standardize=params["standardize"],
                          params=gmm_params,
                          ood=params["ood"])
    valloader = DataLoader(valset,
                           batch_size=params["batchsize"],
                           num_workers=params["workers"])

    # set the random seed
    torch.manual_seed(params["seed"])
    np.random.seed(params["seed"])

    # create the model
    model = FCNet([2] + [params["hidden"] for _ in range(params["layers"])],
                  num_labels=testset.n_clusters,
                  dropout=params["dropout"])
    train_generator(10, GMM(gmm_params), params)
    exit()

    # create the sampler
    if params["sgld"]:
        train_sampler = LangevinSampler(
                model, trainset, params["batchsize"],
                params["sgld_step"], params["sgld_noise"])
    else:
        train_sampler = CategoricalSampler(model, trainset, len(trainloader))

    # train and evaluate
    train(model, trainloader, valloader, train_sampler, params)

    # evaluate on test set
    print("\n===== Evaluating model =====")
    test_acc, predictions = evaluate(model, testloader, trainset.renorm)
    print(f"Accuracy on test set: {test_acc:6.2f}")
    print("============================")

    if params["eplot_file"]:
        _ = energy_plot(model, -10, 10, 1000, params["eplot_file"])
    
    # visualize the results
    if params["show_preds"]:
        show_gmm_points(predictions, gmm_params, params["ood"])    

    # plot learned model distribution
    if params["density_img"]:
        if not params["sgld"]:
            samples = train_sampler.sample_posterior(1000, params["batchsize"])
        else:
            samples = [train_sampler.sample(params["samples"])\
                       for _ in range(1000 // params["batchsize"] + 1)]
            samples = torch.cat(samples)
        plot_density(samples, params["density_img"])


if __name__ == '__main__':
    main()
