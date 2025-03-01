# Comparative Analysis of Classification Performance and Learning Dynamics in VGG, ResNet, U-Net, and a VAE-based Latent Space Clustering Model

---
## Usage

Start by setting up the environment by runnning:

``` bash
pip install -r requirements.txt
```

setup the module ClassComp by using 
``` bash
pip install -e .
```


Use the notebooks in order

- I- Train_Gaussian_noise: Runs reconstruction experiments on synthetic Gaussian Dataset and extracts results and metrcis  
- II - Train Mnist : Runs reconstruction experiment on Mnist dataset 
- III - Plot_graphs : plots results from Mnist-based models and metrics

In case you wish to check the plotting without fully training, we have also prepared a bunch of weights and results you can check out in ``III - plot\_graphs``. You can download it from [here](link)
---

## Introduction and Motivation
Deep learning models, especially convolutional neural networks (CNNs), have achieved remarkable results in classification tasks. Standard architectures like VGG, ResNet, and U-Net are foundational in tasks such as image classification and segmentation. On the other hand, unsupervised approaches, such as Variational Autoencoders (VAEs) combined with clustering on latent space representations, offer exciting possibilities, particularly in contexts where labeled data is scarce or expensive.

This project seeks to systematically compare VGG, ResNet, U-Net, and a VAE + clustering model in terms of:
1. **Empirical Risk Minimization**: How quickly and effectively each model minimizes classification error during training.
2. **Approximation Error**: How well each model captures the underlying data distribution.
3. **Learning Dynamics**: Exploring the trade-offs between convergence speed, risk minimization, and distributional accuracy, particularly for the VAE-based approach.

By investigating these models on both image and synthetic datasets, we aim to reveal insights into their strengths, limitations, and generalization capabilities.

---

## Objectives
### 1. Empirical Risk Comparison
- Analyze the empirical risk evolution of VGG, ResNet, U-Net, and the VAE + clustering model on a binary classification task.
- Compare:
  - **Convergence speed**: How quickly the models reduce empirical risk.
  - **Final empirical risk**: The minimum error each model achieves.

### 2. Approximation Error Evaluation
- Evaluate each model's ability to approximate the true data distribution when it is explicitly known (using synthetic data).
- For VAE: Compute the KL divergence between the true and learned distributions.
- For CNNs: Compare predicted class distributions against the ground truth distribution.

### 3. Trade-off Analysis
- Investigate the relationship between empirical risk minimization and approximation error.
- Determine whether the VAE + clustering model provides a favorable balance compared to CNNs.

---

## Methodology
### 1. Model Selection and Training
- Implement and train **VGG, ResNet, U-Net**, and a **VAE-based model**.
- Use a **binary classification task** derived from the MNIST dataset (e.g., distinguish between the digits `0` and `1`).
- Configure all models with comparable sizes and train them using the same optimizer and hyperparameters for consistency.

### 2. Empirical Risk Tracking
- Track classification loss (empirical risk) over training epochs for each model.
- Visualize convergence behavior to compare training stability and speed across architectures.

### 3. Synthetic Data for Approximation Error
- Generate 2D Gaussian mixture datasets with explicitly known distributions.
- Use these datasets to:
  - Measure each model’s approximation error relative to the true distribution.
  - Compute the **Bayesian risk (R(h*))** and evaluate how close the models get to it.

### 4. Trade-off Analysis
- Compare empirical risk and approximation error across models.
- Examine the implications for generalization and robustness in different data settings.

---

## What We Will Implement Ourselves

### 1. Custom Model Implementations
- **VGG, ResNet, and U-Net**:
  - Develop **nested architectures** to mimic the historical progression of CNN complexity.
  - Implement shared methods for:
    - Extracting and visualizing feature spaces to study model behavior.
    - Comparing feature representations across models.
    - Handling flexible input shapes and dataset types.
- **VAE**:
  - Implement a VAE with two modes:
    - **Generative Mode**: Train to learn \( P(X) \) using reconstruction loss.
    - **Classification Mode**: Extract latent space features and train an SVM on them for supervised classification.

### 2. Flexible Data Loading Framework
- Develop a data loader to handle:
  - **Image Datasets**: Load binary classification tasks from MNIST or other image datasets.
  - **Synthetic Datasets**: Generate 2D Gaussian mixtures or other customizable synthetic datasets.
- Introduce controlled distortions to the data:
  - Class imbalances.
  - Noise (e.g., additive Gaussian noise, label noise).
  - Increased complexity (e.g., overlapping class distributions).

### 3. Study of Estimation and Approximation Errors
- **Synthetic Data Generation**:
  - Create datasets with known distributions and define the **Bayes classifier** as a theoretical baseline.
- **Approximation of \( \inf_h R(h) \)**:
  - Train models on the test set to approximate \( \inf_h R(h) \), providing a reference for estimation error calculations.
- **Estimation of \( R(h) \)**:
  - Train models on the training set and compute their empirical risk.
- **Error Analysis**:
  - Quantify:
    - **Estimation Error**: \( R(h) - \inf_h R(h) \).
    - **Approximation Error**: \( \inf_h R(h) - R(h^*) \), where \( R(h^*) \) is the Bayesian risk.

---

## Expected Contributions
1. A **comparative understanding** of VGG, ResNet, U-Net, and VAE + clustering models in:
   - Classification performance.
   - Convergence speed.
   - Generalization capabilities.
2. Insights into the trade-offs between **empirical risk minimization** and **approximation accuracy** across different architectures.
3. A deeper understanding of how model complexity (e.g., nested CNN architectures) influences:
   - Feature extraction.
   - Learning dynamics.

---

## References
- **VAE**: Kingma, D. P., & Welling, M. *Auto-Encoding Variational Bayes*. [arXiv:1312.6114](https://doi.org/10.48550/arXiv.1312.6114).
- **VGG**: Simonyan, K., & Zisserman, A. *Very Deep Convolutional Networks for Large-Scale Image Recognition*. [arXiv:1409.1556](https://doi.org/10.48550/arXiv.1409.1556).
- **ResNet**: He, K., et al. *Deep Residual Learning for Image Recognition*. [arXiv:1512.03385](https://doi.org/10.48550/arXiv.1512.03385).
- **U-Net**: Ronneberger, O., et al. *U-Net: Convolutional Networks for Biomedical Image Segmentation*. [arXiv:1505.04597](https://doi.org/10.48550/arXiv.1505.04597).
- **MNIST Dataset**: [MNIST on Kaggle](https://www.kaggle.com/datasets/hojjatk/mnist-dataset).
