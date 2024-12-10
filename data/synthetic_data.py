import torch
import numpy as np
from torch.distributions import multivariate_normal
from torch.utils.data import Subset


def generate_synthetic_dataset(
    size: int,
    mu_0: torch.tensor,
    sigma_0: torch.tensor,
    mu_1: torch.tensor,
    sigma_1: torch.tensor,
):
    """
    Create a binary classification subset.
    """
    images, labels = [], []
    for _ in range(size):
        label = np.random.randint(0, 2)
        if label == 0:
            image = generate_sample(mu_0, sigma_0)
        else:
            image = generate_sample(mu_1, sigma_1)
        labels.append(label)
        images.append(image)

    return Subset(images, labels)
    



def generate_sample(mu: torch.tensor, sigma: torch.tensor):
    """
    Draw a single sample from a multivariate gaussian random variable with parameters mu, sigma.
    """
    normal = multivariate_normal.MultivariateNormal(loc=mu, covariance_matrix=sigma)
    return normal.sample()