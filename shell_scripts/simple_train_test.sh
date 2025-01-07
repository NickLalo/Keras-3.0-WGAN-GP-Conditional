# Script to test out training of the WGAN-GP model
#!/bin/bash

# Activate the Conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate keras-gan-gpu

# Run the model with a minimal dataset and epochs for a quick test
echo running a simple model training | tee shell_scripts/training_logs.txt
python train_WGAN_GP.py --fresh_start --dataset_subset_percentage 0.01 --epochs 1
echo done running the model | tee -a shell_scripts/training_logs.txt
echo | tee -a shell_scripts/training_logs.txt
