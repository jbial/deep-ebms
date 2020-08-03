"""
File for different sampling procedures
"""
import numpy as np
import torch

class Sampler:

    def __init__(self, model, data):
        """
        Args:
            model (torch.nn.Module): pytorch model
            data (torch.utils.data.Dataset): pytorch dataset
        """
        self.model = model
        self.data = data
        self.length = len(data)
        self.dsize = len(self.data[0][0])

    def sumexp(self, x):
        """Compute the sum over the last dimension of exp(x)
        """
        return torch.exp(self.model(x)).sum(dim=-1)

    def E(self, x):
        """Computes the negative logsumexp i.e. the energy function
        """
        return torch.logsumexp(self.model(x), dim=-1)

    def sample(self, n):
        """Abstract method for producing samples
        """
        pass

    def EV(self, n):
        """Estimates expected value from n samples
        """
        return torch.mean(self.E(self.sample(n)))

    def sample_posterior(self, n, batchsize, granularity=1000):
        """Samples from the model distribution
        """
        data = np.random.uniform(-10, 10, size=(2 * self.length, self.dsize))
        unnormed_probs = [self.sumexp(torch.Tensor(data[i: i + batchsize])) \
                          for i in range(0, 2 * self.length, batchsize)]
        unnormed_probs = torch.cat(unnormed_probs)

        # sample from the unnormed probabilities
        return self.data[torch.multinomial(unnormed_probs, n, replacement=True)]


class LangevinSampler(Sampler):
    """Stochastic Gradient Langevin Sampler"""
    
    def __init__(self, 
                 model,
                 data,
                 batchsize,
                 stepsize=1,
                 noise=0.01,
                 reinit_prob=0.05,
                 buffer_size=10000,
                 decay=0.9999):
        """
        Args:
            batchsize (int): training batchsize
            stepsize (float): SGLD stepsize
            noise (float): Stddev of SGLD noise
            reinit_prob (float): Probability of reiniting a markov chain
            buffer_size (int): size of PCD buffer
        """
        super().__init__(model, data)

        self.buffer_size = buffer_size
        self.bs = batchsize
        self.reinit_prob = reinit_prob
        self.stepsize = stepsize
        self.noise = noise 
        self.decay = decay

        # get the data shape
        self.data_size = self.data[0][0].size()[0]
        self.replay_buff = self.randTensor(buffer_size, self.data_size)

    def randTensor(self, *size):
        """Generates float tensors sampled from uniform distribution
        """
        return torch.FloatTensor(*size).uniform_(-1, 1)

    def sample(self, n):
        """
        Runs SGLD with persistent contrastive divergence

        Args:
            n (int): SGLD steps to take
        """
        # initialize x
        buffer_inds = torch.randint(0, self.buffer_size, (self.bs,))
        x_0 = self.randTensor(self.bs, self.data_size)
        if self.model.training and np.random.rand() < self.reinit_prob:
            x_0 = self.replay_buff[buffer_inds]

        # run the chain
        self.model.eval()
        x = torch.autograd.Variable(x_0, requires_grad=True)
        for i in range(n):
            # compute the energy and get the gradient
            energy = self.E(x)

            # sgld update
            grad = torch.autograd.grad(energy.sum(), [x], retain_graph=True)[0]
            x.data += (self.stepsize * grad + \
                       self.noise * torch.randn_like(x)) 

        self.model.train()
        # add example to buffer
        if len(self.replay_buff) > 0 and self.model.training:
            self.replay_buff[buffer_inds] = x.detach()

        return x

    def EV(self, n):
        return self.E(self.sample(n)).mean()


class CategoricalSampler(Sampler):
    """Uniform categorical sampler from unnormalized probabilities"""

    def __init__(self, model, data, interval, interp=None, sample_bs=100):
        """
        Args:
            interval (int): interval to recompute the unnormed probabilities
            interp (float): interpolate uniformly random sampled pairs of samples
                            (doubles amount of samples if true)
            sample_bs (int): batchsize to evauluate sample rankings
        """
        super().__init__(model, data)
        self.interp = interp
        self.interval = interval
        self.sample_bs = sample_bs
        self.updater = 0
        self.probs = None
        
    def sample(self, n):
        """Samples from multinomial based on the unnormalized probs
        """
        # sample with replacement from the probs
        if self.updater % self.interval == 0:
            self.probs = torch.Tensor([
                self.sumexp(torch.FloatTensor(x).unsqueeze(0)) \
                for x, _ in self.data
            ])
        self.updater += 1

        sampled_inds = torch.multinomial(self.probs, n, replacement=True)        
        return torch.stack([self.data[i][0] for i in sampled_inds])


class MetroplisSampler(Sampler):
    """Sampler based on the metropolis hastings algorithm"""

    def __init__(self, model, data):
        """
        Args:
            steps (int): number of steps to simulate markov chains for
        """
        super().__init__(model, data)

    def tkernel(self, x):
        """Random walk kernel as proposal distribution
        """
        return x + torch.randn(2)

    def sample(self, n):
        """Samples using the MH sampling algorithm
        """
        # initialize x and run MH
        x = torch.zeros(2)
        for i in range(n):
            # generate new candidate using transition kernel
            y = self.tkernel(x)

            # choose whether or not to accept candidate
            accept = min(1, self.sumexp(y) / self.sumexp(x))
            if np.random.rand() <= accept:
                x = y

        # return the stationary sample
        return x

    def EV(self, n):
        return self.E(self.sample(n))


