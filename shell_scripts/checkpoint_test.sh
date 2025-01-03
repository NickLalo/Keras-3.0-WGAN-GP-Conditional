# Script to test out the training and reloading of the WGAN-GP model
#!/bin/bash

# Activate the Conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate keras-gan-gpu

# Run the model one time to create a checkpoint
echo running the model to create a checkpoint | tee shell_scripts/training_logs.txt
# no need to specify a save frequency since the model is always saved on_train_end
python train_WGAN_GP.py --fresh_start --dataset_subset_percentage 0.01 --epochs 1
echo done running the model to create a checkpoint | tee -a shell_scripts/training_logs.txt
echo | tee -a shell_scripts/training_logs.txt

# Run the model again to continue training from the checkpoint
echo running the model to continue training from the checkpoint | tee -a shell_scripts/training_logs.txt
python train_WGAN_GP.py --reload_last_trained_model --dataset_subset_percentage 0.01 --epochs 1
echo done running the model to continue training from the checkpoint | tee -a shell_scripts/training_logs.txt
echo | tee -a shell_scripts/training_logs.txt

# Run the model a second time to confirm that loading a model from a checkpoint works with no compile or build errors
echo running the model to confirm that saving a model from a checkpoint works | tee -a shell_scripts/training_logs.txt
python train_WGAN_GP.py --reload_last_trained_model --dataset_subset_percentage 0.01 --epochs 1
echo done running the model to confirm that saving a model from a checkpoint works | tee -a shell_scripts/training_logs.txt
echo | tee -a shell_scripts/training_logs.txt