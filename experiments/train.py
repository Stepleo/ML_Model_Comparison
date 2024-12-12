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


def train_vae(
    model: torch.nn.Module, 
    train_loader: DataLoader, 
    test_loader: DataLoader, 
    epochs: int = 10, 
    learning_rate: float = 1e-3, 
    device: str = "cuda", 
    checkpoint_dir: str = "./results/checkpoints"
):
    """
    Trains a Variational Autoencoder (VAE) on a given dataset and saves model checkpoints.

    Args:
        model (torch.nn.Module): The VAE model to train.
        train_loader (DataLoader): DataLoader for training data.
        test_loader (DataLoader): DataLoader for testing data.
        epochs (int): Number of training epochs.
        learning_rate (float): Learning rate for the optimizer.
        device (str): Device to use for training ('cuda' or 'cpu').
        checkpoint_dir (str): Directory to save model checkpoints.
    """
    import os
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        running_recon = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)
        
        for images, _ in progress_bar:
            images = images.to(device)
            
            # Forward pass
            outputs, inputs, mu, log_var = model(images)
            losses = model.loss_function(inputs, outputs, mu, log_var)
            loss = losses['loss']
            recon_loss = losses['Reconstruction_Loss']
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            running_recon += recon_loss.item()
            progress_bar.set_postfix({'Loss': running_loss / len(train_loader)})
        
        print(f"Epoch [{epoch + 1}/{epochs}], Training Loss: {running_loss / len(train_loader):.4f}, Reconstruction Loss: {running_recon / len(train_loader):.4f}")
        
        # Validation phase
        model.eval()
        validation_loss = 0.0
        validation_recon = 0.0
        with torch.no_grad():
            for images, _ in test_loader:
                images = images.to(device)
                
                # Forward pass
                outputs, inputs, mu, log_var = model(images)
                losses = model.loss_function(inputs, outputs, mu, log_var)
                loss = losses['loss']
                recon_loss = losses['Reconstruction_Loss']
                
                validation_loss += loss.item()
                validation_recon += recon_loss.item()
        
        print(f"Epoch [{epoch + 1}/{epochs}], Validation Loss: {validation_loss / len(test_loader):.4f}, Validation Recon: {validation_recon / len(test_loader):.4f}")
        
        # Save model checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f"vae_epoch_{epoch + 1}.pth")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': running_loss / len(train_loader),
            'validation_loss': validation_loss / len(test_loader)
        }, checkpoint_path)
        print(f"Model checkpoint saved at {checkpoint_path}")            