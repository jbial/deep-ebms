"""Experiment driver file
"""
import time
import argparse
import torch
import pprint
import xlsxwriter

from models import FCNet
from torch.utils.data import DataLoader
from dataloader import *
from samplers import *
from jem import *
from sklearn.metrics import roc_auc_score
from utils import normed_logprob_gradient_score


def train_test_stop_experiment(worksheet, valloader, nodes, splits, Params):
    """Runs train-test-stop experiment
    """
    # test
    testset = GMMDataset2D(data_file=Params["test_file"],
                           standardize=Params["standardize"],
                           ood=Params["ood"])
    testloader = DataLoader(testset, 
                            batch_size=Params["batchsize"],
                            num_workers=Params["workers"])

    # run the experiment
    worksheet.write(0, 0, r"Nodes\%Data")
    start = time.time()
    for i, s in enumerate(splits):
        for j, node in enumerate(nodes):
            print("\n============================")
            print(f"INFO: {node} nodes, {s * 100:.1f}% of train data")


            # train
            trainset = GMMDataset2D(data_file=Params["train_file"], 
                                    standardize=Params["standardize"],
                                    split=s)
            trainloader = DataLoader(trainset, 
                                    batch_size=Params["batchsize"],
                                    num_workers=Params["workers"],
                                    shuffle=True)

            # create the model
            model = FCNet([2] + [node for _ in range(Params["layers"])],
                           num_labels=testset.n_clusters,
                           dropout=Params["dropout"])

            # create the sampler
            if Params["sgld"]:
                train_sampler = LangevinSampler(
                        model, trainset, Params["batchsize"], Params["sgld_step"])
            else:
                train_sampler = CategoricalSampler(model, trainset) 

            # train and evaluate
            train_acc, stop_it = train(
                    model, trainloader, valloader, train_sampler, Params
            )
    
            # evaluate on test set
            print("\n===== Evaluating model =====")
            test_acc, _ = evaluate(model, testloader, testset.data_stats())
            print(f"Accuracy on test set: {test_acc:6.2f}")
            print("============================")

            # write to spreadsheet
            worksheet.write(j + 1, 0, f"{node:3d}")
            worksheet.write(0, i + 1, f"{s:.2f}")
            worksheet.write(j + 1, i + 1, 
                            f"{train_acc:.2f}/{test_acc:.2f}/{stop_it}")

    end = time.time() - start
    print(f"INFO: Done in {end:6.2f} sec")
    print("============================")


def ood_experiment(worksheet, valloader, nodes, splits, Params):
    """Runs OOD experiments
    """
    # test
    testset = GMMDataset2D(data_file=Params["test_file"],
                           standardize=Params["standardize"],
                           ood=Params["ood"])
    testloader = DataLoader(testset, 
                            batch_size=Params["batchsize"],
                            num_workers=Params["workers"])

    # run the experiment
    worksheet.write(0, 0, r"Nodes\%Data")
    start = time.time()
    for i, s in enumerate(splits):
        for j, node in enumerate(nodes):
            print("\n============================")
            print(f"INFO: {node} nodes, {s * 100:.1f}% of train data")


            # train
            trainset = GMMDataset2D(data_file=Params["train_file"], 
                                    standardize=Params["standardize"],
                                    split=s,
                                    ood=Params["ood"])
            trainloader = DataLoader(trainset, 
                                    batch_size=Params["batchsize"],
                                    num_workers=Params["workers"],
                                    shuffle=True)

            # create the model
            model = FCNet([2] + [node for _ in range(Params["layers"])],
                           num_labels=testset.n_clusters,
                           dropout=Params["dropout"])

            # create the sampler
            if Params["sgld"]:
                train_sampler = LangevinSampler(
                        model, trainset, Params["batchsize"], Params["sgld_step"])
            else:
                train_sampler = CategoricalSampler(model, trainset) 

            # train and evaluate
            _, stop_it = train(model, trainloader, valloader, train_sampler, Params)
    
            # evaluate on test set
            print("\n===== Evaluating model =====")
            test_acc, _ = evaluate(model, testloader)
            print(f"Accuracy on test set: {test_acc:6.2f}")
            print("============================")

            estimator = logloss(train_sampler, Params["samples"])
            scores = normed_logprob_gradient_score(testloader, model, estimator)

            oodset = testset.oodset
            ood_scores = normed_logprob_gradient_score(
                    [(torch.FloatTensor(oodset[i: i + Params["batchsize"]]), None) \
                     for i in range(0, len(oodset), Params["batchsize"])],
                    model,
                    estimator
            )

            # append the ood scores and create the true labels
            true_scores = [1 for _ in scores]
            true_scores.extend([0 for _ in ood_scores])
            scores.extend(ood_scores)
            auroc = roc_auc_score(true_scores, scores)
            print(f"Area-Under-ROC score: {auroc:.2f}")
            print("============================")

            # write to spreadsheet
            worksheet.write(j + 1, 0, f"{node:3d}")
            worksheet.write(0, i + 1, f"{s:.2f}")
            worksheet.write(j + 1, i + 1, f"{auroc:.2f}/{test_acc:.2f}/{stop_it}")

    end = time.time() - start
    print(f"INFO: Done in {end:6.2f} sec")
    print("============================")


def run_experiment(experiment, xdim, ydim, Params):
    """Runs perturbation experiments on training data split + nodes
    """
    # results spreadsheet
    workbook = xlsxwriter.Workbook(Params["log_file"])
    worksheet = workbook.add_worksheet()

    # validation
    valset = GMMDataset2D(data_file=Params["val_file"],
                          standardize=Params["standardize"],
                          ood=Params["ood"])
    valloader = DataLoader(valset,
                           batch_size=Params["batchsize"],
                           num_workers=Params["workers"])

    # get the gmm params for visualization
    gmm_params = mixture2d_X_params(*Params["X_data_params"])

    # run experiment
    experiment(worksheet, valloader, xdim, ydim, Params)
    workbook.close()
    

def main():

    # set the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--label-file", type=str, default="data/color_labels.csv",
                        help="File that contains the color-label mapping")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of CPU workers for the dataloader")
    parser.add_argument("--jem", action="store_true",
                        help="Include the JEM loss")
    parser.add_argument("--batchsize", type=int, default=64,
                        help="Batch size for training")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate")
    parser.add_argument("--samples", type=int, default=50,
                        help="Number of samples to approximate the EV")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Number of training epochs")
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
    parser.add_argument("--patience", type=int, default=5,
                        help="Patience for early stopping")
    parser.add_argument("--sgld", action='store_true',
                        help="Use the SGLD sampler")
    parser.add_argument("--log-file", type=str, default="logs/exp",
                        help="Prefix of the log file name")
    parser.add_argument("standardize", action='store_true',
                        help="Standardize the input data")
    parser.add_argument("--ood", type=int, default=None,
                        help="Cluster index to exclude for OOD")
    parser.add_argument("--layers", type=int, default=None,
                        help="Number of layers in the DNN")
    parser.add_argument("--dropout", type=float, default=0.0,
                        help="Dropout probability, must be in [0,1)")
    parser.add_argument("--energy-gif", type=str, default=None,
                        help="Filename for energy plot evolution gif")
    parser.add_argument("--eplot-file", type=str, default=None,
                        help="Filename of energy plot to save, if any")
    Params = vars(parser.parse_args())

    # additional params
    Params["sgld_step"] = 2
    Params["sgld_noise"] = 0.01
    Params["X_data_params"] = (4, 4, 1)

    print("****** Hyperparameters ******")
    pprint.pprint(Params)
    print("*****************************")

    # set the random seeds
    torch.manual_seed(Params["seed"])
    np.random.seed(Params["seed"])

    # ablation parameters
    nodes = [20, 40, 80, 100]
    splits = [0.1, 0.25, 0.5, 0.75, 1.0]

    # run experiments
    if Params["ood"] is not None:
        run_experiment(ood_experiment, nodes, splits, Params)
    else:
        run_experiment(train_test_stop_experiment, nodes, splits, Params)


if __name__ == '__main__':
    main()
