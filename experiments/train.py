import os
import time
import pickle
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

def train_binary_classifier(
        model: torch.nn.Module, 
        train_loader: DataLoader, 
        test_loader: DataLoader, 
        criterion=torch.nn.CrossEntropyLoss(),
        epochs: int = 10, 
        learning_rate: float = 1e-3, 
        device: str = "cuda",
        save_results: bool = False,
        description: str = None,
    ):
    """
    Trains a binary classifier on a binary classification dataset and logs metrics.
    
    Args:
        model (torch.nn.Module): The PyTorch model to train.
        train_loader (DataLoader): DataLoader for training data.
        test_loader (DataLoader): DataLoader for testing data.
        criterion: Loss used for training.
        epochs (int): Number of training epochs.
        learning_rate (float): Learning rate for the optimizer.
        device (str): Device to use for training ('cuda' or 'cpu').
        save_results (bool): Whether to save training metrics as a pickle file.
        description (str): Additional description that appears in the pickle file name.
    """
    # Hardcoded directory for saving results
    results_dir = "/home/leo/Programmation/Python/AML_project/ML_Model_Comparison/results/training"
    os.makedirs(results_dir, exist_ok=True)

    # Move model to the specified device
    model.to(device)
    
    # Define loss and optimizer
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

    # Metrics dictionary to store training loss and accuracies
    metrics = {
        "training_loss": [],
        "training_accuracy": [],
        "validation_accuracy": [],
        "training_time": [],
        "gradient_norm": [],
    }
    
    for epoch in range(epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0
        correct_train, total_train = 0, 0
        grad_norm = 0.0

        # Start epoch timer
        start_time = time.time()
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

            # Compute gradient norm
            grad_norm += sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)

            optimizer.step()
            
            running_loss += loss.item()

            # Track training accuracy
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
        
        # End epoch timer
        epoch_time = time.time() - start_time
        epoch_loss = running_loss / len(train_loader)
        epoch_train_acc = 100 * correct_train / total_train
        epoch_grad_norm = grad_norm / len(train_loader)

        metrics["training_loss"].append(epoch_loss)
        metrics["training_accuracy"].append(epoch_train_acc)
        metrics["training_time"].append(epoch_time)
        metrics["gradient_norm"].append(epoch_grad_norm)

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.4f}, Training Accuracy: {epoch_train_acc:.2f}%, Time: {epoch_time:.2f}s, Gradient Norm: {epoch_grad_norm:.4f}")
        
        # Validation step
        model.eval()  # Set the model to evaluation mode
        correct_val, total_val = 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                if model._get_name() == "VAE":
                    _, predicted = torch.min(outputs, 1)
                else:
                    _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
        
        epoch_val_acc = 100 * correct_val / total_val
        metrics["validation_accuracy"].append(epoch_val_acc)

        print(f"Validation Accuracy: {epoch_val_acc:.2f}%")
    
    # Save metrics dictionary as a pickle file if save_results is True
    if save_results:
        file_name = f"{model._get_name()}_{epochs}_epochs_{learning_rate}_lr_{description}.pkl"
        file_path = os.path.join(results_dir, file_name)
        with open(file_path, "wb") as f:
            pickle.dump(metrics, f)
        
        print(f"Training metrics saved to {file_path}")


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
        # checkpoint_path = os.path.join(checkpoint_dir, f"vae_epoch_{epoch + 1}.pth")
        # torch.save({
        #     'epoch': epoch + 1,
        #     'model_state_dict': model.state_dict(),
        #     'optimizer_state_dict': optimizer.state_dict(),
        #     'loss': running_loss / len(train_loader),
        #     'validation_loss': validation_loss / len(test_loader)
        # }, checkpoint_path)
        # print(f"Model checkpoint saved at {checkpoint_path}")            