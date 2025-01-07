import torch
import gc
import tempfile
import os
import numpy as np
from torchvision import transforms
from data.synthetic_data import generate_synthetic_dataset
from data.loaders import get_dataloader
from models.vgg import VGG
from models.resnet import ResNet
from models.unet import UNet
from models.vae import VAE, SVMLoss
from experiments.train import train_binary_classifier, train_vae
from scipy.stats import norm


def error_analysis(
    mus: list,
    sigma: int,
    transform: transforms.Compose = None,
    epochs: int = 5,
    image_size: int = 224,
    dataset_size: int = 500,
    batch_size: int = 4,
):
    """
    Perform Estimation Error vs Approximation Error analysis for different models.

    Args:
        mus (list): Means for the two classes [mu_0, mu_1].
        sigma (int): Variance for both classes.
        transform (transforms.Compose): Optional transformations for the data.
        image_size (int): Size of the generated synthetic images (H=W=image_size).
        dataset_size (int): Number of samples in the synthetic dataset.
        batch_size (int): Batch size for data loading.

    Returns:
        dict: Dictionary containing estimation and approximation errors for each model.
    """
    # Generate synthetic data
    mu_0 = mus[0]
    mu_1 = mus[1]
    sigma_0 = sigma_1 = sigma

    train_subset = generate_synthetic_dataset(
        size=dataset_size, mu_0=mu_0, sigma_0=sigma_0, mu_1=mu_1, sigma_1=sigma_1, image_size=image_size,
    )
    test_subset = generate_synthetic_dataset(
        size=10 * dataset_size, mu_0=mu_0, sigma_0=sigma_0, mu_1=mu_1, sigma_1=sigma_1, image_size=image_size,
    )

    # Load datasets
    train_loader, test_loader = get_dataloader(
        train_subset=train_subset,
        test_subset=test_subset,
        transform=transform,
        size=image_size,
        batch_size=batch_size,
    )

    # Train models on train and test datasets
    model_paths_train = train_models(train_loader, test_loader, image_size, epochs)
    model_paths_test = train_models(test_loader, test_loader, image_size, epochs)

    # Compute Bayes risk
    r_star = get_bayes_risk(mu_0, sigma_0, mu_1, sigma_1, image_size)

    # Reload models and calculate errors
    errors = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # VGG Errors
    vgg_train = load_model(VGG, model_paths_train["VGG"], image_size, device=device)
    vgg_test = load_model(VGG, model_paths_test["VGG"], image_size, device=device)
    errors["VGG"] = get_errors(vgg_train, vgg_test, test_loader, r_star, device)
    del_model(vgg_train)
    del_model(vgg_train)

    # ResNet Errors
    resnet_train = load_model(ResNet, model_paths_train["ResNet"], image_size, device=device)
    resnet_test = load_model(ResNet, model_paths_test["ResNet"], image_size, device=device)
    errors["ResNet"] = get_errors(resnet_train, resnet_test, test_loader, r_star, device)
    del_model(resnet_train)
    del_model(resnet_test)

    # UNet Errors
    resnet_train = load_model(ResNet, model_paths_train["ResNet"], image_size, device=device)
    unet_train = load_model(UNet, model_paths_train["UNet"], device=device, dependency_model=resnet_train)
    del_model(resnet_train)
    resnet_test = load_model(ResNet, model_paths_test["ResNet"], image_size, device=device)
    unet_test = load_model(UNet, model_paths_test["UNet"], device=device, dependency_model=resnet_test)
    del_model(resnet_test)
    errors["UNet"] = get_errors(unet_train, unet_test, test_loader, r_star, device)
    del_model(unet_train)
    del_model(unet_test)

    # VAE Errors
    resnet_train = load_model(ResNet, model_paths_train["ResNet"], image_size, device=device)
    vae_train = load_model(VAE, model_paths_train["VAE"], device=device, dependency_model=resnet_train)
    del_model(resnet_train)
    resnet_test = load_model(ResNet, model_paths_test["ResNet"], image_size, device=device)
    vae_test = load_model(VAE, model_paths_test["VAE"], device=device, dependency_model=resnet_test)
    del_model(resnet_test)
    vae_train.classification_mode = True
    vae_test.classification_mode = True
    errors["VAE"] = get_errors(vae_train, vae_test, test_loader, r_star, device)
    del_model(vae_train)
    del_model(vae_test)

    # Cleanup temporary files
    for path in model_paths_train.values():
        os.remove(path)
    for path in model_paths_test.values():
        os.remove(path)

    return errors


def train_models(train_loader, test_loader, image_size, epochs):
    """
    Train multiple models on the provided data loaders and save them to temporary files.

    Args:
        train_loader (DataLoader): Dataloader for training data.
        test_loader (DataLoader): Dataloader for validation/test data.

    Returns:
        dict: Paths to the saved model files.
    """
    model_paths = {}

    # Temporary directory to store models
    temp_dir = tempfile.mkdtemp(dir=os.path.expanduser("~/Programmation/Python/AML_project/ML_Model_Comparison/results/temp"))
    # Train VGG
    print("Training VGG")
    vgg = VGG(input_img_size=image_size, input_img_c=1)
    train_binary_classifier(vgg, train_loader, test_loader, epochs=epochs)
    vgg_path = os.path.join(temp_dir, f"vgg.pth")
    torch.save(vgg.state_dict(), vgg_path)
    del_model(vgg)
    model_paths["VGG"] = vgg_path

    # Train ResNet
    print("Training ResNet")
    resnet = ResNet(input_img_size=image_size, input_img_c=1)
    train_binary_classifier(resnet, train_loader, test_loader, epochs=epochs)
    resnet_path = os.path.join(temp_dir, f"resnet.pth")
    torch.save(resnet.state_dict(), resnet_path)
    del_model(resnet)
    model_paths["ResNet"] = resnet_path

    # Train UNet
    print("Training UNet")
    resnet = ResNet(input_img_size=image_size, input_img_c=1)  # Load ResNet for UNet initialization
    resnet.load_state_dict(torch.load(resnet_path))
    unet = UNet(resnet.to("cuda"))
    train_binary_classifier(unet, train_loader, test_loader, epochs=epochs)
    unet_path = os.path.join(temp_dir, f"unet.pth")
    torch.save(unet.state_dict(), unet_path)
    del_model(unet)
    model_paths["UNet"] = unet_path

    # Train VAE
    print("Training VAE")
    resnet.load_state_dict(torch.load(resnet_path))  # Load ResNet for VAE initialization
    vae = VAE(resnet.to("cuda"))
    train_vae(vae, train_loader, test_loader, epochs=epochs)
    vae.classification_mode = True

    # Train VAE classification head
    criterion = SVMLoss()
    for param in vae.parameters():
        param.requires_grad = False
    for name, param in vae.named_parameters():
        if "svm_layer" in name:
            param.requires_grad = True

    train_binary_classifier(vae, train_loader, test_loader, criterion=criterion, epochs=epochs)
    vae_path = os.path.join(temp_dir, f"vae.pth")
    torch.save(vae.state_dict(), vae_path)
    del_model(vae)
    model_paths["VAE"] = vae_path

    return model_paths



def load_model(model_class, model_path, image_size=224, device="cpu", dependency_model=None):
    """
    Load a model from a saved file.

    Args:
        model_class: The class of the model to instantiate.
        model_path (str): Path to the saved model file.
        device (str): Device to load the model onto.
        dependency_model (torch.nn.Module, optional): Dependency model, if required.

    Returns:
        torch.nn.Module: The loaded model.
    """
    if dependency_model is not None:
        model = model_class(dependency_model).to(device)
    else:
        model = model_class(input_img_size=image_size, input_img_c=1).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def get_bayes_risk(mu_0, sigma_0, mu_1, sigma_1, dataset):
    """
    Compute the empirical Bayes risk for the given dataset using the Bayes optimal classifier.

    Args:
        mu_0 (float): Mean of class 0.
        sigma_0 (float): Standard deviation of class 0.
        mu_1 (float): Mean of class 1.
        sigma_1 (float): Standard deviation of class 1.
        dataset (dict): Dictionary with keys:
                        - "data": A numpy array of shape (n_samples,).
                        - "labels": A numpy array of shape (n_samples,).

    Returns:
        float: Empirical Bayes risk.
    """
    # Bayes optimal classifier: Decision rule
    def decision_boundary(x):
        x = x.squeeze().flatten()
        lhs = - (1 / sigma_0) * (x - mu_0).T @ (x - mu_0) + (1 / sigma_1) * (x - mu_1).T @ (x - mu_1)
        rhs = np.log(sigma_0 / sigma_1)
        return 0 if lhs > rhs else 1

    # Compute predictions for the dataset using the Bayes optimal classifier
    predictions, labels = np.array([]), np.array([])
    for im, label in dataset:
        pred = np.array([decision_boundary(x) for x in im])
        predictions = np.append(predictions, pred)
        labels = np.append(labels, label.numpy())

    # Compute empirical Bayes risk (misclassification rate)
    errors = predictions != labels
    bayes_risk = np.mean(errors)

    return bayes_risk


def get_errors(model, best_model, test_loader, r_star, device):
    """
    Calculate estimation and approximation errors.

    Args:
        model: Trained model.
        best_model: Model with minimal risk on the test set.
        test_loader (DataLoader): Dataloader for test data.
        r_star (float): Theoretical Bayes risk.

    Returns:
        list: [Estimation Error, Approximation Error]
    """
    empirical_r = get_risk(model, test_loader, device)
    inf_r = get_risk(best_model, test_loader, device)
    estimation_error = empirical_r - inf_r
    approximation_error = inf_r - r_star
    return [estimation_error, approximation_error]


def get_risk(model, test_loader, device):
    """
    Calculate risk (misclassification error) for a model on a test set.

    Args:
        model: Model to evaluate.
        test_loader (DataLoader): Test dataloader.

    Returns:
        float: Risk (error rate).
    """
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            if model._get_name() == "VAE":
                _, predicted = torch.min(outputs, 1)
            else:
                _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 1 - (correct / total)

def del_model(model):
    del model
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.reset_accumulated_memory_stats()
    torch.cuda.empty_cache()
    gc.collect()
