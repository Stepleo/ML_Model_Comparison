## Slice wasserstein estimation

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed
from scipy.stats import norm
from torchvision import datasets, transforms
import torch


def slice_wasserstein_generic(X, Y, n_samples=100, n_slices=100, seed=42):
    """
    Compute the sliced Wasserstein distance between the empirical distribution of two datasets X and Y
    X and Y are numpy arrays of dimension n_X x d and n_Y x d where n is the number of samples and d is the dimension of the samples
    n_samples is the number of uniform samples used to estimate the sliced Wasserstein distance
    n_slices is the number of slices (random direction) used to estimate the sliced Wasserstein distance
    """

    dim = X.shape[1]
    swd = 0.0

    # Generate vectors of the unit sphere
    np.random.seed(seed)
    random_vectors = np.random.randn(n_slices, dim)
    random_vectors = random_vectors / np.linalg.norm(random_vectors, axis=1, keepdims=True)

    for vector in random_vectors:
        # Project the data onto the current vector
        X_proj = np.dot(X, vector)
        Y_proj = np.dot(Y, vector)

        # Sort the samples here to avoid resorting them for each quantiles
        X_proj_sorted = np.sort(X_proj)
        Y_proj_sorted = np.sort(Y_proj)

        # Estimate 1-Wasserstein distance for the projections
        wd = 0
        random_quantiles = np.random.uniform(0, 1, n_samples)
        for q in random_quantiles:
            wd += np.abs(empirical_quantile(X_proj_sorted, q) - empirical_quantile(Y_proj_sorted, q))
        swd += wd / n_samples
    
    swd /= n_slices
    return swd


def empirical_quantile(sorted_samples, q):

    return np.percentile(sorted_samples, q * 100)

def replicated_slice_wasserstein_generic(X, Y, n_samples=100, n_slices=100, n_rep=10, n_jobs=-1):
    """
    Compute the replicated sliced Wasserstein distance between the empirical distributions of X and Y
    in parallel over repetitions.
    """
    def compute_rep(seed):
        """Compute the SWD for a single repetition with a given seed."""
        return slice_wasserstein_generic(X, Y, n_samples=n_samples, n_slices=n_slices, seed=seed)

    # Parallelize computation over repetitions
    results = Parallel(n_jobs=n_jobs)(delayed(compute_rep)(seed) for seed in range(n_rep))
    
    # Average over all repetitions
    r_swd = np.mean(results)
    return r_swd


def get_original_and_reconstructed_images(vae_model, dataloader, device='cpu'):
    """
    Extracts original images of 0s and 1s, along with their reconstructions from a VAE model.

    Args:
        vae_model (torch.nn.Module): Trained VAE model for reconstruction.
        dataloader (torch.utils.data.DataLoader): DataLoader containing the MNIST dataset.
        device (str): Device to run the VAE model ('cpu' or 'cuda').

    Returns:
        dict: A dictionary containing:
              - 'original_0': List of original images of 0s.
              - 'original_1': List of original images of 1s.
              - 'reconstructed_0': List of reconstructed images of 0s.
              - 'reconstructed_1': List of reconstructed images of 1s.
    """
    vae_model.to(device)
    vae_model.eval()
    
    original_0 = []
    original_1 = []
    reconstructed_0 = []
    reconstructed_1 = []

    with torch.no_grad():
        for batch in dataloader:
            images, labels = batch
            images = images.to(device)  # Move images to the specified device
            labels = labels.to(device)
            
            # Forward pass through the VAE model
            reconstructed_images, _, _, _ = vae_model(images)

            for i in range(images.size(0)):
                original_image = images[i].cpu().numpy().squeeze()
                reconstructed_image = reconstructed_images[i].cpu().numpy().squeeze()
                
                # Normalize the original and reconstructed images
                original_image = (original_image - original_image.min()) / (original_image.max() - original_image.min() + 1e-8)
                reconstructed_image = (reconstructed_image - reconstructed_image.min()) / (reconstructed_image.max() - reconstructed_image.min() + 1e-8)
                
                if labels[i] == 0:
                    original_0.append(original_image)
                    reconstructed_0.append(reconstructed_image)
                elif labels[i] == 1:
                    original_1.append(original_image)
                    reconstructed_1.append(reconstructed_image)
                    
    return {
        'original_0': np.array(original_0),
        'original_1': np.array(original_1),
        'reconstructed_0': np.array(reconstructed_0),
        'reconstructed_1': np.array(reconstructed_1),
    }