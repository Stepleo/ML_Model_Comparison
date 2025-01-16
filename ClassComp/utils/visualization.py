import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import torch
import os
import torch.nn as nn
from ClassComp.models.layers import conv_block, residual_block, decoder_block
from sklearn.linear_model import LinearRegression
from sklearn.manifold import TSNE
from matplotlib.colors import ListedColormap

class BaseVisualizationModel(nn.Module):
    """
    Base class providing visualization methods for neural network models.
    Includes methods to visualize filters and feature maps.
    """
    def visualize_filters(
        self,
        layer_name,
        num_filters=6,
        save_name=None,
    ):
        """
        Visualizes filters of a specified convolutional layer.

        Args:
            layer_name (str): Name of the layer whose filters to visualize.
            num_filters (int): Number of filters to visualize.
            save_name (str): Name of saved plot.
        """
        block = dict(self.named_modules()).get(layer_name, None)
        if block is None:
            raise ValueError(f"Layer '{layer_name}' not found in the model.")
        
        outside_conv = extract_conv_layer(block)

        filters = outside_conv.weight.detach().cpu()

        plot_filters(filters, num_filters, save_name)


    def visualize_feature_maps(
            self,
            layer_name,
            inputs,
            num_maps=6,
            relevance_based=False,
            target_class=None,
            save_name=None,
        ):
        """
        Visualizes the feature maps produced by a layer for a given input.

        Args:
            layer_name (str): Name of the layer whose feature maps to visualize.
            inputs (torch.Tensor): Input image batch (batch_size, channels, height, width).
            num_maps (int): Number of feature maps to visualize.
            relevance_based (bool): If True, visualize feature maps by relevance.
            target_class (int or None): Target class for relevance computation (used with relevance_based=True).
            save_name (str): Name of saved plot.
        """
        layer = dict(self.named_modules()).get(layer_name, None)
        if layer is None:
            raise ValueError(f"Layer '{layer_name}' not found in the model.")

        # Hook to capture feature maps and gradients
        def forward_hook(module, input, output):
            self._feature_maps = output.detach().cpu()

        def backward_hook(module, grad_input, grad_output):
            self._feature_maps_grad = grad_output[0].detach().cpu()

        forward_handle = layer.register_forward_hook(forward_hook)
        backward_handle = layer.register_backward_hook(backward_hook)

        # Forward and backward pass
        outputs = self(inputs)
        outputs = outputs.softmax(dim=1)  # Ensure proper class probabilities
        if target_class is not None:
            class_score = outputs[0, target_class]
        else:
            class_score = outputs[0].mean()
        class_score.backward()

        forward_handle.remove()
        backward_handle.remove()

        feature_maps = self._feature_maps[0]  # Visualize feature maps for the first input
        gradients = self._feature_maps_grad[0]

        if relevance_based:
            # Compute relevance as gradients * activations
            relevance_scores = (gradients * feature_maps).sum(dim=(1, 2))
            sorted_indices = relevance_scores.argsort(descending=True)
            feature_maps = feature_maps[sorted_indices]

        plot_feature_maps(feature_maps, num_maps, save_name)



def extract_conv_layer(block):
    """
    Extracts nn.Conv2d layers from a given conv_block or residual_block.
    Args:
        block : A conv_block or residual_block instance.
    Returns:
        layer (nn.Conv2d): last convolution of the block.
    """
    if isinstance(block, conv_block) or block._get_name() == "conv_block":
        all_layers = [layer for layer in block.convs if isinstance(layer, nn.Conv2d)]
        layer = all_layers[-1]
    elif isinstance(block, residual_block) or block._get_name() == "residual_block":
        # Return the convolution of the residual in this case
        all_layers = [layer for layer in block.downsample if isinstance(layer, nn.Conv2d)]
        layer = all_layers[-1]
    elif isinstance(block, nn.Conv2d):
        layer = block
    elif isinstance(block, decoder_block):
        raise Exception("Not implemented for conv transpose yet.")
    else:
        raise Exception(f"Block type {type(block)} not recognized.")
    return layer


def plot_filters(filters, num_filters=10, save_name=None):
    """
    Plots filters from a convolutional layer.
    
    Args:
        filters (np.ndarray): Filters of shape [out_channels, H, W].
        num_filters (int): Number of filters to display.
        save_name (str): Name of saved plot.
    """
    # Normalize filters for visualization
    min_val, max_val = filters.min(), filters.max()
    filters = (filters - min_val) / (max_val - min_val)
    
    # Plot filters
    fig, axes = plt.subplots(1, num_filters, figsize=(15, 5))
    for i in range(min(num_filters, filters.shape[0])):
        axes[i].imshow(filters[i][0], cmap="gray") # Only first kernel of each filter
        axes[i].axis("off")
    if save_name is not None:
        image_path = "results/image/"
        plt.savefig(image_path + save_name + ".png", bbox_inches="tight", dpi=300)
    plt.show()


def plot_feature_maps(feature_maps, num_features=10, save_name=None):
    """
    Plots feature maps from a given layer.
    
    Args:
        feature_maps (np.ndarray): Feature maps of shape [C, H, W].
        num_features (int): Number of feature maps to display.
        save_name (str): Name of saved plot.
    """
    # Normalize feature maps for visualization
    min_val, max_val = feature_maps.min(), feature_maps.max()
    feature_maps = (feature_maps - min_val) / (max_val - min_val)
    
    # Plot feature maps
    fig, axes = plt.subplots(1, num_features, figsize=(15, 5))
    for i in range(min(num_features, feature_maps.shape[0])):
        axes[i].imshow(feature_maps[i], cmap="gray")
        axes[i].axis("off")
    if save_name is not None:
        image_path = "results/image/"
        plt.savefig(image_path + save_name + ".png", bbox_inches="tight", dpi=300)
    plt.show()


def generate_hyperplane_points(svm_weights, svm_bias, latent_dim, num_points=1000):
    """
    Generate points in the latent space that satisfy the hyperplane equation
    w.T * x + b = 0.

    Args:
        svm_weights (np.ndarray): SVM weight vector of shape (latent_dim,).
        svm_bias (float): SVM bias term.
        latent_dim (int): Dimensionality of the latent space.
        num_points (int): Number of points to generate.

    Returns:
        np.ndarray: Points on the hyperplane of shape (num_points, latent_dim).
    """
    # Normalize weights to ensure consistent scaling
    svm_weights = svm_weights / np.linalg.norm(svm_weights)

    # Find basis vectors orthogonal to svm_weights
    orthogonal_basis = []
    identity_matrix = np.eye(latent_dim)

    for i in range(latent_dim):
        vec = identity_matrix[i]
        projection = np.dot(vec, svm_weights) * svm_weights
        orthogonal_vec = vec - projection

        if np.linalg.norm(orthogonal_vec) > 1e-6:  # Avoid numerical issues
            orthogonal_vec /= np.linalg.norm(orthogonal_vec)
            orthogonal_basis.append(orthogonal_vec)

    orthogonal_basis = np.array(orthogonal_basis)

    # Generate random linear combinations of basis vectors
    random_coefficients = np.random.randn(num_points, orthogonal_basis.shape[0])
    hyperplane_points = random_coefficients @ orthogonal_basis

    # Adjust points to satisfy the hyperplane equation
    adjustment = -(svm_bias + hyperplane_points @ svm_weights) / np.dot(svm_weights, svm_weights)
    hyperplane_points += adjustment[:, np.newaxis] * svm_weights

    return hyperplane_points


def plot_vae_tsne_with_svm_boundary(
    vae_model,
    dataloader,
    save_path: str = "results/image/vae_tsne_with_svm.jpg",
    title: str = "2D t-SNE of VAE Latent Space with SVM Boundary",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """
    Visualizes a VAE's latent space using t-SNE with an SVM decision boundary.

    Args:
        vae_model: The trained VAE model with an SVM classification layer.
        dataloader: A PyTorch DataLoader providing images and labels.
        save_path (str): Path to save the generated plot.
        title (str): Title of the plot.
        device (str): Device to use for computations ("cuda" or "cpu").
    """
    vae_model.to(device)
    vae_model.eval()

    latent_vectors = []
    labels = []

    # Extract latent vectors and labels
    with torch.no_grad():
        for images, lbls in dataloader:
            images = images.to(device)
            lbls = lbls.to(device)

            # Forward pass to get latent embeddings
            _, _, mu, _ = vae_model(images)
            latent_vectors.append(mu.cpu().numpy())
            labels.append(lbls.cpu().numpy())

    latent_vectors = np.concatenate(latent_vectors, axis=0)
    labels = np.concatenate(labels, axis=0)

    # Apply t-SNE to reduce latent space to 2D
    tsne = TSNE(n_components=2, perplexity=50, random_state=42)
    tsne_features = tsne.fit_transform(latent_vectors)

    # Compute the decision boundary in the latent space
    svm_weights = vae_model.svm_layer.weight.detach().cpu().numpy()[0]
    svm_bias = vae_model.svm_layer.bias.detach().cpu().numpy()[0]

    boundary_points = generate_hyperplane_points(svm_weights, svm_bias, latent_vectors.shape[1])
    all_points = np.vstack([latent_vectors, boundary_points])

    tsne = TSNE(n_components=2, perplexity=50, random_state=42)
    tsne_results = tsne.fit_transform(all_points)

    tsne_features = tsne_results[: len(latent_vectors)]
    tsne_boundary = tsne_results[len(latent_vectors):]

    # Plotting
    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = ListedColormap(["blue", "red"])

    scatter = ax.scatter(
        tsne_features[:, 0],
        tsne_features[:, 1],
        c=labels,
        cmap=cmap,
        s=20,
        edgecolor="none",
        label="Data Points"
    )

    # Overlay SVM decision boundary points
    ax.scatter(
        tsne_boundary[:, 0],
        tsne_boundary[:, 1],
        color="black",
        s=5,
        label="SVM Boundary Points"
    )

    ax.set_xlabel("t-SNE Dimension 1")
    ax.set_ylabel("t-SNE Dimension 2")
    ax.legend()

    if title:
        ax.set_title(title)

    # Add colorbar for class labels
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_ticks([0.25, 0.75])
    cbar.set_ticklabels(["Class 0", "Class 1"])

    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.show()


def plot_vae_tsne_with_kmeans(
    vae_model,
    dataloader,
    kmeans_dict,
    save_dir: str = "results/image/",
    title_actual: str = "2D t-SNE of VAE Latent Space (Actual Labels)",
    title_kmeans: str = "2D t-SNE of VAE Latent Space (KMeans Labels)",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """
    Visualizes a VAE's latent space using t-SNE, creating two plots:
    one with actual labels and another with KMeans cluster labels.

    Args:
        vae_model: The trained VAE model.
        dataloader: A PyTorch DataLoader providing images and labels.
        kmeans_dict: Trained KMeans object for clustering latent representations.
        save_dir (str): Directory to save the generated plots.
        title_actual (str): Title for the plot with actual labels.
        title_kmeans (str): Title for the plot with KMeans labels.
        device (str): Device to use for computations ("cuda" or "cpu").
    """
    os.makedirs(save_dir, exist_ok=True)
    vae_model.to(device)
    vae_model.eval()

    latent_vectors = []
    actual_labels = []

    # Extract latent vectors and labels
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass to get latent embeddings
            z = vae_model(images)  # Assumes the forward method outputs latent variables z
            latent_vectors.append(z.cpu().numpy())
            actual_labels.append(labels.cpu().numpy())

    latent_vectors = np.concatenate(latent_vectors, axis=0)
    actual_labels = np.concatenate(actual_labels, axis=0)

    # Apply t-SNE to reduce latent space to 2D
    tsne = TSNE(n_components=2, perplexity=50, random_state=42)
    tsne_features = tsne.fit_transform(latent_vectors)

    # Get KMeans cluster labels
    kmeans, inverted = kmeans_dict.values()
    kmeans_labels = kmeans.predict(latent_vectors)
    if inverted:
        kmeans_labels = 1 - kmeans_labels

    # Define function to create scatter plots
    def create_scatter_plot(features, labels, title, save_path, label_names):
        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(
            features[:, 0],
            features[:, 1],
            c=labels,
            cmap=ListedColormap(["blue", "red"]),
            s=20,
            edgecolor="none"
        )
        ax.set_xlabel("t-SNE Dimension 1")
        ax.set_ylabel("t-SNE Dimension 2")
        ax.set_title(title)

        # Add colorbar with label names
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_ticks(np.arange(len(label_names)))
        cbar.set_ticklabels(label_names)

        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        plt.show()

    # Create and save plots
    create_scatter_plot(
        tsne_features,
        actual_labels,
        title_actual,
        save_dir + "vae_tsne_actual_labels.jpg",
        label_names=[f"Class {i}" for i in np.unique(actual_labels)]
    )

    create_scatter_plot(
        tsne_features,
        kmeans_labels,
        title_kmeans,
        save_dir + "vae_tsne_kmeans_labels.jpg",
        label_names=[f"Cluster {i}" for i in np.unique(kmeans_labels)]
    )




def plot_vae_outputs(model, data_loader, device="cuda", num_images=8, save_path=None):
    """
    Visualizes the input and output of a Variational Autoencoder (VAE).
    
    Args:
        model (torch.nn.Module): The trained VAE model.
        data_loader (DataLoader): DataLoader with test or validation data.
        device (str): Device to use for computation ('cuda' or 'cpu').
        num_images (int): Number of images to visualize.
        save_path (str): Name of saved plot.
    """
    model.to(device)
    model.eval()
    
    # Get a batch of data from the data loader
    images, _ = next(iter(data_loader))
    batch_size = images.size(0)
    num_images = min(num_images, batch_size)  # Adjust number of images to plot based on batch size
    images = images[:num_images].to(device)
    
    # Get the reconstructed outputs
    with torch.no_grad():
        reconstructed, _, _, _ = model(images)
    
    images = images.cpu()
    reconstructed = reconstructed.cpu()
    
    # Plot original and reconstructed images
    fig, axes = plt.subplots(2, num_images, figsize=(15, 5))
    for i in range(num_images):
        # Original images
        axes[0, i].imshow(images[i].squeeze(0), cmap="gray")
        axes[0, i].axis("off")
        axes[0, i].set_title("Original")
        
        # Reconstructed images
        axes[1, i].imshow(reconstructed[i].squeeze(0), cmap="gray")
        axes[1, i].axis("off")
        axes[1, i].set_title("Reconstructed")
    
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.show()


def plot_vae_samples(model, num_samples=8, image_size=64, device="cuda", save_path=None):
    """
    Visualizes the samples generated by the Variational Autoencoder (VAE).
    
    Args:
        model (torch.nn.Module): The trained VAE model.
        num_samples (int): Number of samples to visualize.
        device (str): Device to use for computation ('cuda' or 'cpu').
        save_path (str): Name of saved plot.
    """
    model.to(device)
    model.eval()

    # Generate samples
    with torch.no_grad():
        if model._get_name() == "VAE_conv":
            samples = model.sample(num_samples, device)
        else:
            samples = model.sample(num_samples, image_size, device)

    samples = samples.cpu()

    # Plot the generated samples
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 5))
    for i in range(num_samples):
        axes[i].imshow(samples[i].squeeze(0), cmap="gray")
        axes[i].axis("off")
        axes[i].set_title(f"Sample {i+1}")
    
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.show()
    


def plot_training_metrics(metrics_list, save_name):
    """
    Plots training metrics from a list of dictionaries.
    
    Args:
        metrics_list (list): List of dictionaries containing training metrics.
            Each dictionary should have keys:
                - 'name': A string representing the name of the training run.
                - 'training_loss': List of training loss values.
                - 'training_accuracy': List of training accuracy values.
                - 'validation_accuracy': List of validation accuracy values.
                - 'training_time': List of training time values per epoch.
                - 'gradient_norm': List of gradient norms per epoch.
        save_name (str): suffix used for saving plot.
    """
    save_path = "results/image/" + save_name + ".png"
    if not metrics_list:
        print("No metrics provided for plotting.")
        return
    
    # Generate a consistent color map for all runs
    num_runs = len(metrics_list)
    color_map = cm.get_cmap("tab20b", num_runs)  # Use 'tab10' colormap with one color per run

    # Prepare figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    loss_ax, acc_ax, time_ax, grad_ax = axes.flatten()

    for idx, metrics in enumerate(metrics_list):
        name = metrics.get("name", f"Run {idx + 1}")
        color = color_map(idx)  # Get the color for this run
        epochs = list(range(1, len(metrics['training_loss']) + 1))
        
        # Plot training loss
        loss_ax.plot(epochs, metrics['training_loss'], label=f"{name} - Training Loss", color=color)
        
        # Plot training and validation accuracy
        acc_ax.plot(epochs, metrics['training_accuracy'], label=f"{name} - Training Accuracy", color=color)
        acc_ax.plot(epochs, metrics['validation_accuracy'], linestyle='--', label=f"{name} - Validation Accuracy", color=color)
        
        # Plot training time per epoch
        time_ax.plot(epochs, metrics['training_time'], label=f"{name} - Training Time", color=color)
        
        # Plot gradient norm per epoch
        grad_ax.plot(epochs, metrics['gradient_norm'], label=f"{name} - Gradient Norm", color=color)
    
    # Set titles and labels for each plot
    loss_ax.set_title("Training Loss")
    loss_ax.set_xlabel("Epochs")
    loss_ax.set_ylabel("Loss")
    loss_ax.legend()
    loss_ax.grid(True)
    
    acc_ax.set_title("Training and Validation Accuracy")
    acc_ax.set_xlabel("Epochs")
    acc_ax.set_ylabel("Accuracy")
    acc_ax.legend()
    acc_ax.grid(True)
    
    time_ax.set_title("Training Time per Epoch")
    time_ax.set_xlabel("Epochs")
    time_ax.set_ylabel("Time (seconds)")
    time_ax.legend()
    time_ax.grid(True)
    
    grad_ax.set_title("Gradient Norm per Epoch")
    grad_ax.set_xlabel("Epochs")
    grad_ax.set_ylabel("Gradient Norm")
    grad_ax.legend()
    grad_ax.grid(True)

    # Adjust layout and show plots
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.show()


def plot_vae_training_metrics(metrics, save_name):
    """
    Plots VAE training and validation metrics.

    Args:
        metrics (dict): Dictionary containing training and validation metrics.
            Keys should include:
                - 'training_loss': List of total training loss values.
                - 'training_recon_loss': List of training reconstruction loss values.
                - 'training_kld_loss': List of training KLD loss values.
                - 'validation_loss': List of total validation loss values.
                - 'validation_recon_loss': List of validation reconstruction loss values.
                - 'validation_kld_loss': List of validation KLD loss values.
        save_name (str): File name for saving the plot.
    """
    save_path = f"results/image/{save_name}.png"

    # Check if metrics are provided
    if not metrics:
        print("No metrics provided for plotting.")
        return

    # Create subplots for the different loss components
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Define the keys and titles for each subplot
    loss_components = [
        ("training_loss", "validation_loss", "Total Loss"),
        ("training_recon_loss", "validation_recon_loss", "Reconstruction Loss"),
        ("training_kld_loss", "validation_kld_loss", "KLD Loss"),
    ]

    for ax, (train_key, val_key, title) in zip(axes, loss_components):
        # Check if keys exist in metrics
        if train_key not in metrics or val_key not in metrics:
            print(f"Metrics do not contain keys: {train_key}, {val_key}. Skipping...")
            continue

        # Get the data for training and validation
        train_data = metrics[train_key]
        val_data = metrics[val_key]
        epochs = range(1, len(train_data) + 1)

        # Plot the metrics
        ax.plot(epochs, train_data, label="Training", color="blue")
        ax.plot(epochs, val_data, label="Validation", color="orange", linestyle="--")

        # Set titles and labels
        ax.set_title(title)
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(True)

    # Adjust layout and save the plot
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.show()
