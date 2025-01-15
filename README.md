# Wasserstein GAN with Gradient Penalty (WGAN-GP)

This repository implements a Wasserstein GAN with Gradient Penalty (WGAN-GP) for the conditional generation of MNIST numbers. The WGAN-GP model leverages the advantages of Wasserstein loss and gradient penalty for stable training, producing high-quality samples of MNIST digits.

## Setup Instructions

To run this project, you will need to set up a Python environment with the required dependencies. This repository provides a `environment.yaml` file for easy environment setup using Conda.

#### Creating the Environment

1. **Install Conda** (if not already installed):
    - You can download and install Conda from [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html).

2. **Create the environment**:
    ```bash
    conda env create -f _config/environment.yaml -y
    ```
    *Note*: This process might take some time as it installs all necessary dependencies, including GPU-compatible libraries if available.

3. **Activate the environment**:
    ```bash
    conda activate keras-gan-gpu
    ```

4. **To remove the environment** (if needed):
    - Deactivate the environment:
        ```bash
        conda deactivate
        ```
    - Remove the environment:
        ```bash
        conda env remove --name keras-gan-gpu
        ```

## TODO

- Detailed instructions for running the training script.
- Sample results and visualizations.
- Tips for customizing the model and dataset.
