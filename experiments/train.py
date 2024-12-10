import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

def train_binary_classifier(
        model: torch.nn.Module, 
        train_loader: DataLoader, 
        test_loader: DataLoader, 
        epochs: int = 10, 
        learning_rate: float = 1e-3, 
        device: str = "cuda",
    ):
    """
    Trains a binary classifier on a binary classification dataset.
    
    Args:
        model (torch.nn.Module): The PyTorch model to train.
        train_loader (DataLoader): DataLoader for training data.
        test_loader (DataLoader): DataLoader for testing data.
        epochs (int): Number of training epochs.
        learning_rate (float): Learning rate for the optimizer.
        device (str): Device to use for training ('cuda' or 'cpu').
    """
    # TODO: Add checkpoints, metrics and visualization
    # Move model to the specified device
    model.to(device)
    
    # Define loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        
        for images, labels in progress_bar:
            # Move data to the device
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(train_loader):.4f}")
        
        # Validation step
        model.eval()  # Set the model to evaluation mode
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        print(f"Validation Accuracy: {100 * correct / total:.2f}%")
