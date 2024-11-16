"""
File to hold some utility functions not directly related to the WGAN_GP model
"""


import os
import time


def print_fresh_start_warning_message():
    """
    function to print out a big warning message that takes around ~10 seconds giving the user a small amount of time to cancel the operation if they
    want to.
    
    Parameters:
        None
    
    Returns:
        None
    """
    print(f"\n\n{'#'*150}")
    for num in range(1, 25):
        sleep_time = round(num * 0.03, 2)
        print(f"{'#'*(num+1)} FRESH START. TRAINING STARTING OVER, model_training_output WILL BE DELETED! {'#'*(num+1)}".center(150))
        time.sleep(sleep_time)
    print(f"{'#'*150}\n\n")
    return


def get_last_checkpoint_dir_and_file(model_checkpoints_dir):
    """
    function to get the path for the last model checkpoint dir and last model checkpoint directory and file
    
    Parameters:
        model_checkpoints_dir (Path): the path to the model_checkpoints directory
    
    Returns:
        last_checkpoint_dir_path (Path): the path to the last checkpoint directory
        last_model_checkpoint_path (Path): the path to the last model checkpoint file
    """
    sorted_checkpoint_dirs = sorted(os.listdir(model_checkpoints_dir))
    last_checkpoint_dir = sorted_checkpoint_dirs[-1]
    
    # list the files in the last checkpoint directory
    last_checkpoint_dir_path = model_checkpoints_dir.joinpath(last_checkpoint_dir)
    last_checkpoint_files = os.listdir(last_checkpoint_dir_path)
    # get the filename that ends with .keras
    last_model_checkpoint_filename = [file for file in last_checkpoint_files if file.endswith(".keras")][0]
    last_model_checkpoint_path = last_checkpoint_dir_path.joinpath(last_model_checkpoint_filename)
    return last_checkpoint_dir_path, last_model_checkpoint_path