# Script to test out the training and reloading of the WGAN-GP model with a specific checkpoint. This can be modified to help pick up training
# from a failed run. However, some of the epochs may need to be deleted so that the last epoch in the model_checkpoints directory has a 
# model_save directory available for reloading.
# NOTE: run from the base directory of the project
#!/bin/bash

# if the parent dir of the current path is shell_scripts, then navigate one up to the base dir of the project
if [[ $(basename $(dirname $(pwd))) == "shell_scripts" ]]; then
    cd ..
fi

# Activate the Conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate keras-gan-gpu

echo running the model from a specific checkpoint | tee shell_scripts/training_logs.txt
# no need to specify a save frequency since the model is always saved on_train_end
python train_WGAN_GP.py \
    --reload_path wgan_gp_mnist_training_runs/2025-01-03__10-51-05__0060 \
    --dataset_subset_percentage 0.1 \
    --epochs 5
echo done running the model from a specific checkpoint | tee -a shell_scripts/training_logs.txt
echo | tee -a shell_scripts/training_logs.txt