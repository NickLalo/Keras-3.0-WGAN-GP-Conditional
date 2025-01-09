# Script to test out training of the WGAN-GP model with a minimal dataset and a long training time
# NOTE: run from the base directory of the project
#!/bin/bash

# if the parent dir of the current path is shell_scripts, then navigate one up to the base dir of the project
if [[ $(basename $(dirname $(pwd))) == "shell_scripts" ]]; then
    cd ..
fi

# Activate the Conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate keras-gan-gpu

# Run the model with a minimal dataset and epochs for a quick test
echo running a simple model training for a long time | tee shell_scripts/training_logs.txt
python train_WGAN_GP.py --fresh_start \
    --dataset_subset_percentage 0.01 \
    --epochs 1000 \
    --model_save_frequency 0 \
    --video_of_validation_frequency 0
echo done running the model | tee -a shell_scripts/training_logs.txt
echo | tee -a shell_scripts/training_logs.txt
