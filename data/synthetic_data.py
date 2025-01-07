import torch
import numpy as np
from torch.utils.data import Dataset, Subset

class SyntheticDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        """
        A custom dataset for synthetic data.

        Args:
            images (list of torch.Tensor): List of image tensors.
            labels (list of int): List of corresponding labels.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        # Ensure the transform works with tensors
        if self.transform:
            if isinstance(image, torch.Tensor):
                image = image.unsqueeze(0)  # Add a channel dimension for transforms expecting it
            image = self.transform(image)

        return image, label


def generate_synthetic_dataset(
    size: int,
    mu_0: float,
    sigma_0: float,
    mu_1: float,
    sigma_1: float,
    image_size: int,
    seed: int = None,
):
    """
    Create a binary classification subset with an optional seed for reproducibility.

    Args:
        size (int): Number of samples.
        mu_0 (float): Mean value for class 0.
        sigma_0 (float): Standard deviation for class 0.
        mu_1 (float): Mean value for class 1.
        sigma_1 (float): Standard deviation for class 1.
        image_size (int): Image size (square: image_size x image_size).
        seed (int, optional): Seed for random number generation.

    Returns:
        torch.utils.data.Subset: Synthetic dataset containing images and labels.
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    images, labels = [], []
    for _ in range(size):
        label = np.random.randint(0, 2)
        if label == 0:
            image = generate_sample(mu_0, sigma_0, image_size)
        else:
            image = generate_sample(mu_1, sigma_1, image_size)
        labels.append(label)
        images.append(image)

    full_dataset = SyntheticDataset(images, labels)

    return Subset(full_dataset, indices=list(range(size)))


def generate_sample(mu: float, sigma: float, image_size: int):
    """
    Generate a single image by sampling from a univariate Gaussian distribution.

    Args:
        mu (float): Mean value of the distribution.
        sigma (float): Standard deviation of the distribution.
        image_size (int): Size of the image (square: image_size x image_size).

    Returns:
        torch.Tensor: Generated image of size (image_size x image_size).
    """
    return torch.normal(mean=mu, std=sigma, size=(image_size, image_size))
