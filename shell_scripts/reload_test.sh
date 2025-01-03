# Script to test out the training and reloading of the WGAN-GP model with a specific checkpoint
#!/bin/bash

# Activate the Conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate keras-gan-gpu

echo running the model from a specific checkpoint | tee shell_scripts/training_logs.txt
# no need to specify a save frequency since the model is always saved on_train_end
python train_WGAN_GP.py \
    --reload_path /home/nicklalo/coding-projects/Keras-3.0-WGAN-GP-Conditional/wgan_gp_mnist_training_runs/0023__2025-01-02__10-05-53 \
    --dataset_subset_percentage 0.01 \
    --epochs 1
echo done running the model from a specific checkpoint | tee -a shell_scripts/training_logs.txt
echo | tee -a shell_scripts/training_logs.txt