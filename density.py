"""Saves density plots generated by trained samplers
"""
import torch
import argparse

from models import VAE
from utils import density_map

def main():

    # set the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--zdim", type=int, default=2,
                        help="number of latent variables")
    parser.add_argument("--dropout", type=float, default=0,
                        help="Dropout probability, must be in [0,1)")
    parser.add_argument("--hidden", type=int, default=100,
                        help="Number of hidden units")
    parser.add_argument("--seed", type=int, default=99,
                        help="Random seed")
    parser.add_argument("--layers", type=int, default=3,
                        help="Number of layers in the DNN")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Saved model path")
    parser.add_argument("--n", type=int, default=100,
                        help="Grid size")
    parser.add_argument("--samples", type=int, default=100,
                        help="Number of Monte-carlo estimator samples")
    params = vars(parser.parse_args())

    # set the random seed
    torch.manual_seed(params["seed"])

    # create the model
    layers = [2] + [params["hidden"] // 2**i for i in range(params["layers"])]
    vae = VAE(layers, zdim=params["zdim"], dropout=params["dropout"])

    # load the model
    vae.load_state_dict(torch.load(params["model_path"]))
    vae.eval()

    n, samples = params["n"], params["samples"]

    # plot density map
    prefix = "images/density"
    suffix = params["model_path"].split('.')[0].split('_')[-1]
    density_map(vae,
                n=n,
                samples=samples,
                filename=f"{prefix}/dplot_prob_{n}x{samples}_{suffix}")
    print(f"Density maps saved to {prefix}")
            

if __name__ == "__main__":
    main()
