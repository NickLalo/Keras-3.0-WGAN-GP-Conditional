"""
File to hold some utility functions not directly related to the WGAN_GP model
"""


import os
import sys
import io
import traceback
from pathlib import Path
import argparse
import psutil
import subprocess
from datetime import datetime
import pytz


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
        "--noise_shape",
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


def get_last_checkpoint_paths_for_reload(all_model_training_output_dir: Path):
    """
    function to find the last model training run started and return the path to the model and last checkpoint directories so that the model can be 
    reloaded to continue training, or used for inference.
    
    Parameters:
        all_model_training_output_dir (Path): the path to the directory containing all the model training output directories
    
    Returns:
        last_model_training_run_dir (Path): the path to the last model training run directory
        model_checkpoints_dir (Path): the path to the checkpoint directories in the last model training directory
        last_checkpoint_dir_path (Path): the path to the last checkpoint directory in the last model training directory
        model_save_file_path (Path): the path to the last model save file in the last checkpoint directory
    """
    # sort the directories in the model_training_output directory and get the path to the last model training started
    sorted_model_training_dirs = sorted(os.listdir(all_model_training_output_dir))
    last_model_created = sorted_model_training_dirs[-1]
    last_model_training_run_dir = all_model_training_output_dir.joinpath(last_model_created)
    
    # get the path to the checkpoint directories in the last model training started, sort them, and get the last checkpoint directory
    checkpoint_dir = last_model_training_run_dir.joinpath("model_checkpoints")
    sorted_checkpoint_dirs = sorted(os.listdir(checkpoint_dir))
    last_checkpoint_dir = sorted_checkpoint_dirs[-1]
    last_checkpoint_dir_path = checkpoint_dir.joinpath(last_checkpoint_dir)
    
    # find the .keras file in the last checkpoint directory
    last_checkpoint_dir_files = os.listdir(last_checkpoint_dir_path)
    model_save_file = [file for file in last_checkpoint_dir_files if file.endswith(".keras")][0]
    model_save_file_path = last_checkpoint_dir_path.joinpath(model_save_file)
    
    # get the model checkpoints dir from the model_training_output_dir
    model_checkpoints_dir = last_model_training_run_dir.joinpath("model_checkpoints")
    
    return last_model_training_run_dir, model_checkpoints_dir, last_checkpoint_dir_path, model_save_file_path 


def get_specific_checkpoint_paths_for_reload(reload_path: Path):
    """
    function to return the path to the model and last checkpoint directories from a specified model training run directory so that the model can be
    reloaded to continue training, or used for inference.
    
    Parameters:
        reload_path (Path): the path to the directory containing the model specified for reloading
    
    Returns:
        model_checkpoints_dir (Path): the path to the checkpoint directories in the model training directory
        last_checkpoint_dir_path (Path): the path to the last checkpoint directory in the model training directory
        last_model_save_file_path (Path): the path to the last model save file in the last checkpoint directory
    """
    # get the path to the checkpoint directories in the last model training started, sort them, and get the last checkpoint directory
    model_checkpoints_dir = reload_path.joinpath("model_checkpoints")
    sorted_checkpoint_dirs = sorted(os.listdir(model_checkpoints_dir))
    last_checkpoint_dir = sorted_checkpoint_dirs[-1]
    last_checkpoint_dir_path = model_checkpoints_dir.joinpath(last_checkpoint_dir)
    
    # find the .keras file in the last checkpoint directory
    last_checkpoint_dir_files = os.listdir(last_checkpoint_dir_path)
    model_save_file = [file for file in last_checkpoint_dir_files if file.endswith(".keras")][0]
    last_model_save_file_path = last_checkpoint_dir_path.joinpath(model_save_file)
    
    return model_checkpoints_dir, last_checkpoint_dir_path, last_model_save_file_path


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


def get_timestamp():
    """
    function to get a formatted timestamp in the US Central Time Zone for logging purposes that can also be used as a unique identifier
    """
    # Get the current time in the US Central Time Zone with custom formatting
    central_time = datetime.now(pytz.timezone('US/Central'))
    formatted_time = central_time.strftime("%Y-%m-%d__%H:%M:%S")
    return formatted_time


def get_readable_time_string(seconds_input: float):
    """
    function to convert seconds (float) to a readable time string in the format HHH:MM:SS.ss
    
    """
    # convert to a readable format (HH:MM:SS.ss)
    hours = int(seconds_input // 3600)
    minutes = int((seconds_input % 3600) // 60)
    seconds = seconds_input % 60
    readable_time_string = f"{hours:03}:{minutes:02}:{seconds:05.2f}"
    return readable_time_string


class Terminal_Logger:
    """
    Custom class to redirect the standard output and standard error to a log file while still printing to the terminal. Useful for running
    scripts on remote servers that might close the terminal session and lose the output or for viewing the output of older scripts.
    """
    def __init__(self, log_path):
        """
        log_path: PurePath object representing the full path of the log file
        """
        self.terminal = sys.stdout
        self.log = io.open(log_path, 'a', encoding='utf-8')
        self._setup_logging()
        # print out a header to the log file to indicate that the terminal output is being logged
        print(f"\n\n{'='*150}\n{'='*150}\nLogging terminal output to: {log_path} starting at {get_timestamp()}\n{'='*150}\n{'='*150}\n\n")
        return
    
    def _setup_logging(self):
        sys.stdout = self
        sys.stderr = self
        sys.excepthook = self.custom_exception_hook
        return
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()  # Ensure messages are written out immediately
        return
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()
        return
    
    def custom_exception_hook(self, exctype, value, tb):
        """
        custom exception hook to print the error traceback to stderr, which is now redirected to Logger. When the script errors out, the
        logs will contain the error traceback.
        """
        # Print the error traceback to stderr, which is now redirected to Logger
        print(f"\n\nERROR OCCURRED\n{'-'*40}", file=sys.stderr)
        traceback.print_exception(exctype, value, tb)
        return
    
    def reconnect_to_log_file(self):
        """
        Re-establish logging to the log file. This is useful if sys.stdout was redirected temporarily.
        """
        self._setup_logging()
        return
