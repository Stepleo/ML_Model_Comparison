import os
import time
import pickle
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans

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

def train_vae_kmeans(
        vae: torch.nn.Module, 
        train_loader: DataLoader, 
        test_loader: DataLoader,  
        device: str = "cuda",
        save_results: bool = False,
        description: str = None,
    ):
    """
    Train and evaluate a VAE with KMeans clustering on its latent space.

    Args:
        vae (torch.nn.Module): The VAE model in "KMeans" mode.
        train_loader (DataLoader): Dataloader for the training data.
        test_loader (DataLoader): Dataloader for the validation data.
        device (str): The device to use for computations (default: "cuda").
        save_results (bool): Whether to save the training metrics as a pickle file (default: False).
        description (str): Additional description to add to the file name when saving metrics.

    Returns:
        tuple: A dictionary of metrics and the trained KMeans object.
    """
    # Hardcoded directory for saving results
    results_dir = "/home/leo/Programmation/Python/AML_project/ML_Model_Comparison/results/training"
    os.makedirs(results_dir, exist_ok=True)
    model_dir = "/home/leo/Programmation/Python/AML_project/ML_Model_Comparison/results/models"
    os.makedirs(results_dir, exist_ok=True)

    # Make sure the VAE is in KMeans mode
    vae.classification_mode = "KMeans"

    # Move model to device
    vae.to(device)
    vae.eval()

    # Metrics dictionary to store training loss and accuracies
    metrics = {
        "training_accuracy": [],
        "validation_accuracy": [],
        "training_time": [],
    }

    # Compute latent representations for the training set
    start_time = time.time()
    train_latent_representations = []
    train_labels = []
    inverted = False

    with torch.no_grad():
        for images, labels in train_loader:
            images = images.to(device)
            z = vae(images)  # Latent representations
            z_numpy = z.cpu().numpy()
            labels_numpy = labels.cpu().numpy()
            train_latent_representations.append(z_numpy)
            train_labels.append(labels_numpy)
    train_latent_representations = np.concatenate(train_latent_representations, axis=0)
    train_labels = np.concatenate(train_labels, axis=0)

    # Fit KMeans on latent representations
    num_clusters = len(np.unique(train_labels))  # Assuming class labels correspond to the number of clusters
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(train_latent_representations)
    train_time = time.time() - start_time
    metrics["training_time"].append(train_time)

    # Predict cluster labels for training set and compute accuracy
    train_predicted_labels = kmeans.predict(train_latent_representations)
    train_accuracy = accuracy_score(train_labels, train_predicted_labels)
    if train_accuracy < 1 - train_accuracy:
        inverted = True
        train_accuracy = 1 - train_accuracy
    metrics["training_accuracy"].append(train_accuracy)

    # Evaluate on the validation set
    validation_latent_representations = []
    validation_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            z = vae(images)  # Latent representations
            validation_latent_representations.append(z.cpu().numpy())
            validation_labels.append(labels.cpu().numpy())
    validation_latent_representations = np.concatenate(validation_latent_representations, axis=0)
    validation_labels = np.concatenate(validation_labels, axis=0)

    # Predict cluster labels for validation set and compute accuracy
    validation_predicted_labels = kmeans.predict(validation_latent_representations)
    if inverted:
        validation_predicted_labels = 1 - validation_predicted_labels
    validation_accuracy = accuracy_score(validation_labels, validation_predicted_labels)
    metrics["validation_accuracy"].append(validation_accuracy)

    # Save metrics dictionary as a pickle file if save_results is True
    if save_results:
        file_name = f"{vae._get_name()}_{description}_kmeans.pkl"
        file_path = os.path.join(results_dir, file_name)
        with open(file_path, "wb") as f:
            pickle.dump(metrics, f)
        print(f"Training metrics saved to {file_path}")

        file_name = f"{vae._get_name()}_{description}_kmeans.pkl"
        file_path = os.path.join(model_dir, file_name)
        with open(file_path, "wb") as f:
            pickle.dump({"kmeans": kmeans, "inverted": inverted}, f)
        print(f"Kmeans saved to {file_path}")


def train_vae(
    model: torch.nn.Module, 
    train_loader: DataLoader, 
    test_loader: DataLoader, 
    epochs: int = 10, 
    learning_rate: float = 1e-3, 
    device: str = "cuda", 
    save_results: bool = False,
    description: str = None,
):
    """
    Trains a Variational Autoencoder (VAE) on a given dataset and logs metrics.

    Args:
        model (torch.nn.Module): The VAE model to train.
        train_loader (DataLoader): DataLoader for training data.
        test_loader (DataLoader): DataLoader for testing data.
        epochs (int): Number of training epochs.
        learning_rate (float): Learning rate for the optimizer.
        device (str): Device to use for training ('cuda' or 'cpu').
        save_results (bool): Whether to save training metrics as a pickle file.
        description (str): Additional description that appears in the pickle file name.
    """
    # Hardcoded directory for saving results
    results_dir = "/home/leo/Programmation/Python/AML_project/ML_Model_Comparison/results/training"
    os.makedirs(results_dir, exist_ok=True)

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Metrics dictionary to store training and validation loss components
    metrics = {
        "training_loss": [],
        "training_recon_loss": [],
        "training_kld_loss": [],
        "validation_loss": [],
        "validation_recon_loss": [],
        "validation_kld_loss": [],
        "training_time": [],
        "gradient_norm": [],
    }

    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        running_recon = 0.0
        running_kld = 0.0
        grad_norm = 0.0

        # Start epoch timer
        start_time = time.time()
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)
        
        for images, _ in progress_bar:
            images = images.to(device)
            
            # Forward pass
            outputs, inputs, mu, log_var = model(images)
            losses = model.loss_function(inputs, outputs, mu, log_var)
            loss = losses['loss']
            recon_loss = losses['Reconstruction_Loss']
            kld_loss = losses['KLD']
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Compute gradient norm
            grad_norm += sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)

            optimizer.step()
            
            running_loss += loss.item()
            running_recon += recon_loss.item()
            running_kld += kld_loss.item()
            progress_bar.set_postfix({'Loss': running_loss / len(train_loader)})
        
        # End epoch timer
        epoch_time = time.time() - start_time
        epoch_loss = running_loss / len(train_loader)
        epoch_recon_loss = running_recon / len(train_loader)
        epoch_kld_loss = running_kld / len(train_loader)
        epoch_grad_norm = grad_norm / len(train_loader)

        metrics["training_loss"].append(epoch_loss)
        metrics["training_recon_loss"].append(epoch_recon_loss)
        metrics["training_kld_loss"].append(epoch_kld_loss)
        metrics["training_time"].append(epoch_time)
        metrics["gradient_norm"].append(epoch_grad_norm)

        print(f"Epoch [{epoch + 1}/{epochs}], Training Loss: {epoch_loss:.4f}, Reconstruction Loss: {epoch_recon_loss:.4f}, KLD Loss: {epoch_kld_loss:.4f}, Time: {epoch_time:.2f}s, Gradient Norm: {epoch_grad_norm:.4f}")
        
        # Validation phase
        model.eval()
        validation_loss = 0.0
        validation_recon = 0.0
        validation_kld = 0.0
        with torch.no_grad():
            for images, _ in test_loader:
                images = images.to(device)
                
                # Forward pass
                outputs, inputs, mu, log_var = model(images)
                losses = model.loss_function(inputs, outputs, mu, log_var)
                loss = losses['loss']
                recon_loss = losses['Reconstruction_Loss']
                kld_loss = losses['KLD']
                
                validation_loss += loss.item()
                validation_recon += recon_loss.item()
                validation_kld += kld_loss.item()

        epoch_val_loss = validation_loss / len(test_loader)
        epoch_val_recon_loss = validation_recon / len(test_loader)
        epoch_val_kld_loss = validation_kld / len(test_loader)

        metrics["validation_loss"].append(epoch_val_loss)
        metrics["validation_recon_loss"].append(epoch_val_recon_loss)
        metrics["validation_kld_loss"].append(epoch_val_kld_loss)

        print(f"Epoch [{epoch + 1}/{epochs}], Validation Loss: {epoch_val_loss:.4f}, Validation Recon: {epoch_val_recon_loss:.4f}, Validation KLD: {epoch_val_kld_loss:.4f}")

    # Save metrics dictionary as a pickle file if save_results is True
    if save_results:
        file_name = f"{model._get_name()}_{epochs}_epochs_{learning_rate}_lr_{description}_generation.pkl"
        file_path = os.path.join(results_dir, file_name)
        with open(file_path, "wb") as f:
            pickle.dump(metrics, f)

        print(f"Training metrics saved to {file_path}")
            