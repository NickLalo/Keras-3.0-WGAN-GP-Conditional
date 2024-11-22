"""
File to hold some utility functions not directly related to the WGAN_GP model
"""


import os
import time
import argparse
import psutil
import subprocess


def parse_arguments():
    """
    Parse command-line arguments for the script.
    Returns:
        args (Namespace): Parsed arguments with fresh_start and debug_run as attributes.
    """
    parser = argparse.ArgumentParser(description="Train a WGAN-GP model on MNIST data.")
    # fresh start
    parser.add_argument(
        "--fresh_start",
        action="store_true",
        default=False,
        help="If set, deletes previous checkpoints and starts training from scratch.",
        )
    # debug run
    parser.add_argument(
        "--debug_run",
        action="store_true",
        default=False,
        help="If set, runs in debug mode with a reduced dataset and fewer epochs.",
        )
    # percentage between 0 and 1
    parser.add_argument(
        "--dataset_subset_percentage",
        type=float,
        default=1.0,
        help="The percentage of the dataset to use for training in small subset mode.",
        )
    # batch size
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="The batch size for training the model.",
        )
    # noise dimension
    parser.add_argument(
        "--noise_dim",
        type=int,
        default=128,
        help="The dimension of the noise vector for the generator.",
        )
    # epochs
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="The number of epochs to train the model.",
        )
    # parse the arguments and return
    return parser.parse_args()


def print_fresh_start_warning_message():
    """
    function to print out a big warning message that takes around ~10 seconds giving the user a small amount of time to cancel the operation if they
    want to.
    
    Parameters:
        None
    
    Returns:
        None
    """
    print(f"\n\n{'#'*160}\n")
    # iterate up to 25 making a bigger warning message each time
    count_up_list = list(range(1, 30))
    # iterate down to 1 so we can print out a timer
    count_down_list = count_up_list[::-1]
    for count_up, count_down in zip(count_up_list, count_down_list):
        sleep_time = round(count_up * 0.03, 2)
        print(
            f"{'#' * (count_up + 1)} | {count_down:>2} | "
            f"FRESH START. TRAINING STARTING OVER, model_training_output WILL BE DELETED!"
            f" | {count_down:>2} | {'#' * (count_up + 1)}".center(160)
            )
        time.sleep(sleep_time)
    print(f"\n{'#'*160}\n\n")
    time.sleep(0.5)  # give a little bit of extra time
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


def get_memory_usage():
    """
    function to get the current memory usage of the process
    """
    # get the current process and its memory usage
    process = psutil.Process(os.getpid())
    total_memory_usage = process.memory_info().rss
    memory_mb = total_memory_usage / 1024 ** 2
    memory_gb = memory_mb / 1024
    
    return memory_mb, memory_gb


def get_gpu_memory_usage():
    """
    function to get the GPU memory usage
    """
    try:
        # Run nvidia-smi and parse the output
        result = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,nounits,noheader'],
            encoding='utf-8'
        )
        gpu_memory_info = result.strip().split("\n")
        total_memory_used_mb = 0
        for i, gpu in enumerate(gpu_memory_info):
            used, total = gpu.split(", ")
            # print(f"GPU {i}: {used} MB / {total} MB")
            total_memory_used_mb += int(used)
        memory_mb = total_memory_used_mb
        memory_gb = memory_mb / 1024
        
    except FileNotFoundError:
        print("nvidia-smi not found. Ensure that NVIDIA drivers and tools are installed.")
        memory_mb = 9999  # numbers that are obviously not memory usage to indicate an error
        memory_gb = 9999  # numbers that are obviously not memory usage to indicate an error
    except Exception as e:
        print(f"Error querying GPU memory usage: {e}")
        memory_mb = 9999  # numbers that are obviously not memory usage to indicate an error
        memory_gb = 9999  # numbers that are obviously not memory usage to indicate an error
    
    return memory_mb, memory_gb
