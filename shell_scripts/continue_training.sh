# Script to continue training on the last trained model with the default parameters
# NOTE: run from the base directory of the project
#!/bin/bash

# if the parent dir of the current path is shell_scripts, then navigate one up to the base dir of the project
if [[ $(basename $(dirname $(pwd))) == "shell_scripts" ]]; then
    cd ..
fi

# Activate the Conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate keras-gan-gpu

echo continuing to train the last trained model | tee shell_scripts/training_logs.txt
python train_WGAN_GP.py --reload_last_trained_model
echo done continuing training the last trained model | tee -a shell_scripts/training_logs.txt
echo | tee -a shell_scripts/training_logs.txt