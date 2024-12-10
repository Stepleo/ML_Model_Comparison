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


def get_dataloader(
    train_subset: Subset, 
    test_subset: Subset, 
    transform: transforms.Compose = None, 
    size: int = 224, 
    batch_size: int = 16
):
    """
    Preprocesses given subsets for binary classification
    and returns DataLoaders for training and testing.
    
    Always applies the preprocessing transform, and conditionally applies additional transformations 
    if provided via the `transform` argument.
    
    Args:
        train_subset (Subset): Subset of train data.
        test_subset (Subset): Subset of test data.
        transform (transforms.Compose, optional): Additional transformation pipeline to apply to the data.
        size (int, optional): The size to resize the image to (default: 224).
        batch_size (int): The batch size for the DataLoaders (default: 16).
    
    Returns:
        Tuple[DataLoader, DataLoader]: DataLoaders for the binary classification dataset.
    """
    # TODO: Include class imbalance, noise etc overall ways to increase complexity
    # Maybe in utils create a class that can hold all these transformations that pertubate the data
    # and has a method to apply them that we can use here instead of transform
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
    
    # Create DataLoaders
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader
