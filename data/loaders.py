import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

def get_mnist():
    """
    Extracts digits 0 and 1 from the MNIST dataset and returns train and test subsets for binary classification.
    
    The images are not preprocessed or resized here; preprocessing should be applied later via transforms.
    
    Returns:
        Tuple[Subset, Subset]: Subsets of the MNIST train and test datasets containing only digits 0 and 1.
    """
    # Download the MNIST dataset
    mnist_train = datasets.MNIST(root='./data', train=True, download=True)
    mnist_test = datasets.MNIST(root='./data', train=False, download=True)
    
    # Filter indices for class 0 and 1
    train_indices = [i for i, label in enumerate(mnist_train.targets) if label in [0, 1]]
    test_indices = [i for i, label in enumerate(mnist_test.targets) if label in [0, 1]]
    
    # Create subsets
    train_subset = Subset(mnist_train, train_indices)
    test_subset = Subset(mnist_test, test_indices)
    
    return train_subset, test_subset


def preprocessing_transform(size: int = 224):
    """
    Returns a transformation pipeline for preprocessing MNIST images.
    
    The transformations include resizing to the specified size, converting to a tensor, 
    and normalizing the pixel values to the range [-1, 1].
    
    Args:
        size (int): The size to resize the image to (default: 224).
    
    Returns:
        torchvision.transforms.Compose: The transformation pipeline.
    """
    transform = transforms.Compose([
        transforms.Resize((size, size)),  # Resize images to (size, size)
        transforms.ToTensor(),  
        transforms.Normalize((0.5,), (0.5,))  # Normalize to range [-1, 1]
    ])
    return transform


def apply_class_imbalance(dataset: Subset, imbalance_ratio: float):
    """
    Introduces class imbalance into the dataset.
    
    Args:
        dataset (Subset): The dataset to modify.
        imbalance_ratio (float): Desired proportion of class 1. Must be in range (0, 1).
        
    Returns:
        Subset: The modified dataset with class imbalance.
    """
    # Extract labels using the subset indices
    labels = np.array(dataset.dataset.targets)[dataset.indices]
    
    # Find indices for each class
    class_1_indices = np.array(dataset.indices)[labels == 1]
    class_0_indices = np.array(dataset.indices)[labels == 0]

    # Determine the number of samples for each class
    num_class_1 = int(imbalance_ratio * len(labels))
    num_class_0 = len(labels) - num_class_1

    # Subsample the indices to match the desired imbalance
    selected_class_1_indices = np.random.choice(class_1_indices, size=num_class_1, replace=True)
    selected_class_0_indices = np.random.choice(class_0_indices, size=num_class_0, replace=True)

    # Combine and shuffle the selected indices
    balanced_indices = np.concatenate([selected_class_1_indices, selected_class_0_indices])
    np.random.shuffle(balanced_indices)

    # Return a new subset with the balanced indices
    return Subset(dataset.dataset, indices=balanced_indices.tolist())


def get_dataloader(
    train_subset: Subset, 
    test_subset: Subset, 
    transform: transforms.Compose = None, 
    size: int = 224, 
    batch_size: int = 16,
    class_imbalance: float = None, 
):
    """
    Preprocesses given subsets for binary classification
    and returns DataLoaders for training and testing.
    
    Adds options to introduce class imbalance.
    
    Args:
        train_subset (Subset): Subset of train data.
        test_subset (Subset): Subset of test data.
        transform (transforms.Compose, optional): Additional transformation pipeline to apply to the data.
        size (int, optional): The size to resize the image to (default: 224).
        batch_size (int): The batch size for the DataLoaders (default: 16).
        class_imbalance (float, optional): Percentage of class 1 in the training set. Must be in range (0, 1). 
                                           If None, no imbalance is applied (default: None).
    
    Returns:
        Tuple[DataLoader, DataLoader]: DataLoaders for the binary classification dataset.
    """
    # Define preprocessing transform
    preprocess = preprocessing_transform(size)
    
    # If an additional transform is provided, chain it with preprocessing
    if transform:
        combined_transform = transforms.Compose([preprocess, transform])
    else:
        combined_transform = preprocess
    
    # Apply transformations to the datasets
    train_subset.dataset.transform = combined_transform
    test_subset.dataset.transform = combined_transform
    
    # Handle class imbalance
    if class_imbalance is not None:
        assert 0 < class_imbalance < 1, "class_imbalance must be a float between 0 and 1."
        train_subset = apply_class_imbalance(train_subset, class_imbalance)
    
    # Create DataLoaders
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

