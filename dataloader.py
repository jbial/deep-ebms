"""
File for dataset classes and utilities
"""
import torch
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.patches import Ellipse
from torch.utils.data import Dataset, DataLoader


def mixture_params(x, y, sigma):
    """Creates a gaussian mixture params of the shape:
                    o   o
                      o
                    o   o
        Args:
            x (float): horizontal offset of clusters from origin
            y (float): vertical offset of clusters from origin
            sigma (float): covariance parameter of clusters
    """
    return [(np.array(mean), np.diag([sigma, sigma])) for mean in \
            [(0, 0), (x, y), (-x, y), (x, -y), (-x, -y)]]


def build_GMMX_dataset(x_offset, y_offset, covar, split):
    """Builds the X dataset for 2d GMM data

    Args:
        x_offset (float): x distance from outer cluster means from (0, 0) 
        y_offset (float): y distance from outer cluster means from (0, 0) 
        covar (float): covariance of each cluster 
        split (list): Number of points per cluster for train/test/val
    """
    params = mixture2d_X_params(x_offset, y_offset, covar)
    ds = GMMDataset2D(params=params, n_points=sum(split)).points
        
    # separate the points
    train_points = [cluster[:split[0]] for cluster in ds]
    val_points = [cluster[split[0]: split[0] + split[1]] for cluster in ds]
    test_points = [cluster[split[0] + split[1]:] for cluster in ds]

    # dump them to csv
    dump(train_points, "data/mixture2d_X_train.csv")
    dump(val_points, "data/mixture2d_X_val.csv")
    dump(test_points, "data/mixture2d_X_test.csv")


def _read_color_labels(filename):
    """Returns the integer -> color mapping for the labels
    """
    line_parser = lambda line: (int(line.split(',')[0]), line.split(',')[-1])
    with open(filename, 'r') as labels:
        label_map = dict([line_parser(line.strip()) for line in labels])
    return label_map


def draw_ellipse(position, covariance, ax=None, num_contours=5, **kwargs):
    """Draw an ellipse with a given position and covariance"""
    ax = ax or plt.gca()
    
    # Convert covariance to principal axes
    U, s, Vt = np.linalg.svd(covariance)
    angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
    width, height = 2 * np.sqrt(s)
    
    # Draw the Ellipse
    for nsig in range(1, num_contours):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height,
                             angle, **kwargs))


def show_gmm_points(points, params, bmin=-10, bmax=10, ood=None, **kwargs):
    """Shows points and gaussians

    Args:
        predictions (iterable): contains tuples of (point, label)
        params (iterable): contains gmm params
    """
    color_map = _read_color_labels("data/color_labels.csv")
    plt.figure()
    plt.title(kwargs.get("title", "Points"))
    plt.xlabel("x")
    plt.ylabel("y")

    # plot the contours of the gaussians
    params = params or []
    for i, (mean, cov) in enumerate(params):
        if ood is not None:
            col = color_map[i - 1] if i > ood else (\
                  color_map[i] if i < ood else 'k')
        else:
            col = color_map[i]
        draw_ellipse(mean, cov, color=col, alpha=0.1)

    # plot the data with the predicted label
    for p, label in points:
        plt.scatter(*p, c=color_map[label], marker='.')

    plt.xlim(bmin, bmax)
    plt.ylim(bmin, bmax)
    plt.gca().set_aspect('equal', adjustable='box')

    if kwargs.get("filename"):
        plt.savefig(kwargs["filename"], dpi=200)
    else:
        plt.show()


def generate_GMM_params(n_clusters, spread, circular=True):
    """Generates random parameters for mixture of gaussians model
    """
    params = [[np.random.normal(scale=spread, size=2), 
               np.random.rand(2)] \
               for _ in range(n_clusters)]

    # create valid covariance matrices
    for i in range(n_clusters):
        S = abs(params[i][-1])
        if circular:
            # just pick the first stddev as the radius of the point cloud
             params[i][-1] = np.diag([S[0], S[0]])
        else:
             params[i][-1] = np.diag(S)
    return params


def dump(points, filename):
    """Dumps data to a file

    Args:
        points (iterable): list of points for every cluster
        filename (str): filename for the data file
    """
    with open(filename, 'w') as f:
        for i, pts in enumerate(points):
            for x, y in pts:
                f.write(f"{x:.3f},{y:.3f},{i}\n")
    print(f"Dumping data to {filename}...")
       

class GMMDataset2D(Dataset):
    """GMM dataset for 2D point clouds"""        

    def __init__(self, 
                 params=None,
                 n_points=50,
                 spread=4,
                 n_clusters=3,
                 circular=True,
                 label_file="data/color_labels.csv",
                 data_file=None,
                 standardize=False,
                 split=None,
                 ood=None):
        """
        Args:
            params (tuple): tuple of means and covariances
            n_points (int): number of points to sample from per cluster
            spread (int): std of the centers of the GMM clusters
            n_clusters (int): number of GMM components
            circular (bool): make the clusters circular
            label_file (str): file that contains the index-label mapping
            data_file (str): read in data from file instead of generating
            standardize (bool): standardize the data
            split (float): percentage of data to store from a data file
            ood (int): cluster index to exclude for ood detection
        """
        self.n_points = n_points
        self.n_clusters = n_clusters if not params else len(params)
        self.spread = spread
        self.circular = circular
        self.params = params
        self.data_file = data_file
        self.standardize = standardize
        self.split = split
        self.ood = ood

        # generate data points
        if data_file:
            self.points = self.load_data()
        else:
            self.points = self._generate_gmm_data()
        
        # simulate OOD    
        if ood is not None:
            self.oodset = self.points[ood]
            self.n_clusters -= 1
            self.points.pop(ood)
    
        # for standardizing data
        if standardize:
            self.min, self.max = np.min(self.points), np.max(self.points)

            for i in range(self.n_clusters):
                self.points[i] = 2 * ((self.points[i] - self.min)/\
                                 (self.max - self.min)) - 1

    def _generate_gmm_data(self):
        """Generates a 2D data from a GMM
        """
        
        # randomly generate means and covariances of GMM if params not given
        if not self.params:
            self.params = generate_GMM_params(self.n_clusters, self.spread) 

        # sample from the GMM
        return [np.random.multivariate_normal(mean, std, size=self.n_points) \
                for mean, std in self.params]

    def load_data(self):
        """Loads in data from a data file
        """
        raw_data = np.genfromtxt(self.data_file, delimiter=',')
        self.n_clusters = int(raw_data[-1][-1] + 1)
        self.n_points = len(raw_data) // self.n_clusters
        
        # group data according to label
        data = [raw_data[raw_data[:,-1] == i][:,:-1] \
                for i in range(self.n_clusters)]

        # take only a subset of the data
        if self.split:
            assert 0 <= self.split <= 1, "Split must be in [0, 1)"

            # update dataset info and print to stdout
            self.n_points = int(self.split * len(data[0]))
            subsampled = self.__len__() - int(self.ood is not None) * self.n_points
            print(f"INFO: Subsampled {subsampled}/{len(raw_data)} points")
            
            return [cluster[:self.n_points] for cluster in data]
        return data

    def __len__(self):
        return self.n_points * self.n_clusters

    def __getitem__(self, idx):
        label = idx // self.n_points
        point = idx % self.n_points

        if idx < 0:
            point = -(self.n_points - point)
            label += self.n_clusters

        sample = self.points[label][point]
        return torch.from_numpy(sample).float(), label

    def renorm(self, x):
        """Renormalizes data
        """
        return (x / 2 + 0.5) * (self.max - self.min) + self.min

    def show(self, **kwargs):
        """Shows the dataset in a scatter plot
        """
        show_gmm_points([(pt, i) for i, pts in enumerate(self.points)\
                        for pt in pts], self.params, **kwargs)

           
def main():
    
    X_params = (4, 4, 1)
    gmm_params = mixture2d_X_params(*X_params)
#
#    build_GMMX_dataset(*X_params, [4500, 500, 1500])
    ds = GMMDataset2D(data_file="data/mixture2d_X_test.csv", params=gmm_params)
    ds.show()


if __name__ == '__main__':
    main()

