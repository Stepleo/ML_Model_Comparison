import matplotlib.pyplot as plt
import torch.nn as nn
from models.layers import conv_block, residual_block, decoder_block

class BaseVisualizationModel(nn.Module):
    """
    Base class providing visualization methods for neural network models.
    Includes methods to visualize filters and feature maps.
    """
    def visualize_filters(
        self,
        layer_name,
        num_filters=6,
    ):
        """
        Visualizes filters of a specified convolutional layer.

        Args:
            layer_name (str): Name of the layer whose filters to visualize.
            num_filters (int): Number of filters to visualize.
        """
        block = dict(self.named_modules()).get(layer_name, None)
        if block is None:
            raise ValueError(f"Layer '{layer_name}' not found in the model.")
        
        outside_conv = extract_conv_layer(block)

        filters = outside_conv.weight.detach().cpu()

        plot_filters(filters, num_filters)


    def visualize_feature_maps(
            self,
            layer_name,
            inputs,
            num_maps=6,
            relevance_based=False,
            target_class=None,
        ):
        """
        Visualizes the feature maps produced by a layer for a given input.

        Args:
            layer_name (str): Name of the layer whose feature maps to visualize.
            inputs (torch.Tensor): Input image batch (batch_size, channels, height, width).
            num_maps (int): Number of feature maps to visualize.
            relevance_based (bool): If True, visualize feature maps by relevance.
            target_class (int or None): Target class for relevance computation (used with relevance_based=True).
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

        plot_feature_maps(feature_maps, num_maps)



def extract_conv_layer(block):
    """
    Extracts nn.Conv2d layers from a given conv_block or residual_block.
    Args:
        block : A conv_block or residual_block instance.
    Returns:
        layer (nn.Conv2d): last convolution of the block.
    """
    if isinstance(block, conv_block):
        all_layers = [layer for layer in block.convs if isinstance(layer, nn.Conv2d)]
        layer = all_layers[-1]
    elif isinstance(block, residual_block):
        # Return the convolution of the residual in this case
        all_layers = [layer for layer in block.downsample if isinstance(layer, nn.Conv2d)]
        layer = all_layers[-1]
    elif isinstance(block, nn.Conv2d):
        layer = block
    elif isinstance(block, decoder_block):
        raise Exception("Not implemented for conv transpose yet.")
    else:
        raise Exception("Block type not recognized.")
    return layer


def plot_filters(filters, num_filters=10):
    """
    Plots filters from a convolutional layer.
    
    Args:
        filters (np.ndarray): Filters of shape [out_channels, H, W].
        num_filters (int): Number of filters to display.
    """
    # Normalize filters for visualization
    min_val, max_val = filters.min(), filters.max()
    filters = (filters - min_val) / (max_val - min_val)
    
    # Plot filters
    fig, axes = plt.subplots(1, num_filters, figsize=(15, 5))
    for i in range(min(num_filters, filters.shape[0])):
        axes[i].imshow(filters[i][0], cmap="gray") # Only first kernel of each filter
        axes[i].axis("off")
    plt.show()

def plot_feature_maps(feature_maps, num_features=10):
    """
    Plots feature maps from a given layer.
    
    Args:
        feature_maps (np.ndarray): Feature maps of shape [C, H, W].
        num_features (int): Number of feature maps to display.
    """
    # Normalize feature maps for visualization
    min_val, max_val = feature_maps.min(), feature_maps.max()
    feature_maps = (feature_maps - min_val) / (max_val - min_val)
    
    # Plot feature maps
    fig, axes = plt.subplots(1, num_features, figsize=(15, 5))
    for i in range(min(num_features, feature_maps.shape[0])):
        axes[i].imshow(feature_maps[i], cmap="gray")
        axes[i].axis("off")
    plt.show()