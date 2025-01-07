"""
File to hold some utility functions not directly related to the WGAN_GP model
"""


import os
import sys
import io
import time
import shutil
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
    
    NOTE: When reloading the model, the learning rate arguments are loaded from the model configuration file instead of the command line arguments.
    
    Parameters:
        None
    
    Returns:
        model_configurations (dict): Dictionary of parsed model configurations.
    """
    parser = argparse.ArgumentParser(description="Train a WGAN-GP model on MNIST data.")
    
    ################################################################### debug run ####################################################################
    parser.add_argument("--debug_run", action="store_true", default=False,
        help="If set, runs in debug mode with a reduced dataset and fewer epochs. Cannot be set with a dataset_subset_percentage other than 1.0.")
    
    ########################################################### loading specific arguments ###########################################################
    # Create a mutually exclusive group for fresh start or reload options
    group = parser.add_mutually_exclusive_group()
    # fresh_start
    group.add_argument(
        "--fresh_start", action="store_true",
        help="Starts a new model training run. (default if reload_last_trained_model or reload_path are not set)."
    )
    # reload_last_trained_model
    group.add_argument(
        "--reload_last_trained_model", action="store_true", 
        help="Reloads the last trained model to continue training."
    )
    # reload_path
    group.add_argument(
        "--reload_path", type=Path,
        help="The path to the model training run directory to reload the model from."
    )
    
    ############################################################### dataset arguments ################################################################
    # dataset_subset_percentage
    parser.add_argument("--dataset_subset_percentage", type=float, default=1.0,
        help="The percentage of the dataset to use for training in small subset mode. Cannot be set with debug_run.")
    # random_shift_frequency
    parser.add_argument("--random_rotate_frequency", type=float, default=0.0,
        help="The frequency of applying random rotations to the training images. e.g. 0.5 means 50% of the time.")
    # random_translate_frequency
    parser.add_argument("--random_translate_frequency", type=float, default=0.0,
        help="The frequency of applying random translations to the training images. e.g. 0.5 means 50% of the time.")
    # random_zoom_frequency
    parser.add_argument("--random_zoom_frequency", type=float, default=0.0,
        help="The frequency of applying random zooms to the training images. e.g. 0.5 means 50% of the time.")
    
    ########################################################## model and training arguments ##########################################################
    # noise_shape
    parser.add_argument("--noise_shape", type=int, default=128,
        help="The dimension of the noise vector for the generator.")
    # batch_size
    parser.add_argument("--batch_size", type=int, default=512,
        help="The batch size for training the model.")
    # epochs
    parser.add_argument("--epochs", type=int, default=600,
        help="The number of epochs to train the model on the current run. On reload, the model will train this many more epochs.")
    # critic_to_generator_training_ratio
    parser.add_argument("--critic_to_generator_training_ratio", type=int, default=5,
        help="The number of times the critic is trained for every time the generator is trained.")
    # gradient_penalty_weight
    parser.add_argument("--gradient_penalty_weight", type=float, default=10.0,
        help="The weight of the gradient penalty in the loss function.")
    # initial_critic_learning_rate  (set to 60% of the generator learning rate)
    parser.add_argument("--initial_critic_learning_rate", type=float, default=6e-5,
        help="The initial learning rate for the model training.")
    # initial_generator_learning_rate
    parser.add_argument("--initial_generator_learning_rate", type=float, default=1e-4,
        help="The initial learning rate for the model training.")
    # learning_rate_warmup_epochs
    parser.add_argument("--learning_rate_warmup_epochs", type=int, default=50,
        help="The number of epochs to warm up the learning rate.")
    # learning_rate_decay
    parser.add_argument("--learning_rate_decay", type=float, default=0.99,
        help="The decay factor for the learning rate.")
    # min_critic_learning_rate
    parser.add_argument("--min_critic_learning_rate", type=float, default=2.25e-5,
        help="The minimum learning rate for the critic.")
    # min_generator_learning_rate
    parser.add_argument("--min_generator_learning_rate", type=float, default=3.75e-5,
        help="The minimum learning rate for the generator.")
    
    ##################################################### model, video, and FID score frequency ######################################################
    # model_save_frequency
    parser.add_argument("--model_save_frequency", type=int, default=10,
        help="The frequency of saving the model.")
    # video_of_validation_frequency
    parser.add_argument("--video_of_validation_frequency", type=int, default=1,
        help="The frequency of creating a video of the validation samples.")
    # FID_score_frequency
    parser.add_argument("--FID_score_frequency", type=int, default=1,
        help="The frequency of calculating the FID score. Defaults to 0 (no FID score calculation) because it is computationally expensive.")
    
    ############################################################### validate arguments ###############################################################
    # Parse arguments
    args = parser.parse_args()
    # Default to fresh_start if neither reload_last_trained_model nor reload_path are set
    if not args.reload_last_trained_model and args.reload_path is None:
        args.fresh_start = True
    
    # ensure that dataset_subset_percentage is greater than 0 and less than or equal to 1.0
    if args.dataset_subset_percentage <= 0 or args.dataset_subset_percentage > 1.0:
        raise ValueError("dataset_subset_percentage must be greater than 0 and less than or equal to 1.0.")
    
    # ensure that debug_run and a non 1.0 dataset_subset_percentage are not set together
    if args.debug_run and args.dataset_subset_percentage != 1.0:
        raise ValueError("debug_run and a dataset_subset_percentage other than 1.0 cannot be set together.")
    
    # ensure that the random augmentations are between 0 and 1.0
    if args.random_rotate_frequency < 0 or args.random_rotate_frequency > 1.0:
        raise ValueError("random_rotate_frequency must be between 0 and 1.0.")
    if args.random_translate_frequency < 0 or args.random_translate_frequency > 1.0:
        raise ValueError("random_translate_frequency must be between 0 and 1.0.")
    if args.random_zoom_frequency < 0 or args.random_zoom_frequency > 1.0:
        raise ValueError("random_zoom_frequency must be between 0 and 1.0.")
    
    # Convert args namespace to dictionary
    training_params = vars(args)
    
    # if the model is being run in debug mode, set the number of epochs to a small number
    if training_params["debug_run"]:
        print("\nDEBUG MODE: Running with a small number of epochs.\n")
        training_params["epochs"] = 5
    
    return training_params


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
    model_checkpoints_dir = last_model_training_run_dir.joinpath("model_checkpoints")
    sorted_checkpoint_dirs = sorted(os.listdir(model_checkpoints_dir))
    last_checkpoint_dir = sorted_checkpoint_dirs[-1]
    last_checkpoint_dir_path = model_checkpoints_dir.joinpath(last_checkpoint_dir)
    
    # get the path to the model save directory in the last checkpoint directory
    model_save_file_path = last_checkpoint_dir_path.joinpath("model_save")
    
    # error out if the model save file does not exist or is empty
    if not model_save_file_path.exists() or model_save_file_path.stat().st_size == 0:
        print(f"\nERROR: The model save file does not exist or is empty: {model_save_file_path}")
        print(f"Please check the model training run directory: {last_model_training_run_dir}")
        print(f"The model is only saved every X epochs, so the last checkpoint directory may not have a model save file and may need to be adjusted")
        print(f"Try deleting some checkpoints so that the last checkpoint directory has a model save file")
        exit()
    
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
    
    # get the path to the model save directory in the last checkpoint directory
    model_save_file_path = last_checkpoint_dir_path.joinpath("model_save")
    
    # error out if the model save file does not exist or is empty
    if not model_save_file_path.exists() or model_save_file_path.stat().st_size == 0:
        print(f"\nERROR: The model save file does not exist or is empty: {model_save_file_path}")
        print(f"Please check the model training run directory: {reload_path}")
        print(f"The model is only saved every X epochs, so the last checkpoint directory may not have a model save file and may need to be adjusted")
        print(f"Try deleting some checkpoints so that the last checkpoint directory has a model save file")
        exit()
    
    return model_checkpoints_dir, last_checkpoint_dir_path, model_save_file_path


def get_last_model_save_dir_path(model_checkpoints_dir):
    """
    helper script for the end of the training script to get the path to the last model save directory
    
    Parameters:
        model_checkpoints_dir (Path): the path to the directory containing the model checkpoint directories
    
    Returns:
        model_save_file_path (Path): the path to the last model save directory
    """
    # get a sorted list of all the model checkpoint directories
    sorted_checkpoint_dirs = sorted(os.listdir(model_checkpoints_dir))
    # only look at the last checkpoint directory
    last_checkpoint_dir_path = model_checkpoints_dir.joinpath(sorted_checkpoint_dirs[-1])
    # get the path to the model save directory in the last checkpoint directory
    model_save_file_path = last_checkpoint_dir_path.joinpath("model_save")
    return model_save_file_path


def backup_model_code(model_training_output_dir: Path):
    """
    extra mechanism to backup the code used for a specific model training run assuming a hardcoded file name is used and everything is run from the
    base directory. This is not best practice, but is a quick and hacky way to backup the code used for a specific model training run.
    
    Parameters:
        model_training_output_dir: Path object, the directory where the model training output is stored
    
    Returns:
        None
    """
    model_code_source = Path("critic_and_generator_models.py")
    model_code_destination = model_training_output_dir.joinpath("model_architecture_and_summary").joinpath(model_code_source)
    if model_code_source.exists():
        shutil.copy(model_code_source, model_code_destination)
        print(f"Model source code {model_code_source} backed up successfully.")
    else:
        print(f"Model source code {model_code_source} does not exist. Skipping backup of code file.")
    return


def print_and_save_training_parameters(training_parameters, model_training_output_dir):
    """
    Prints and saves the model configuration to a file.
    This function takes a dictionary of model configurations and a directory path where the model training output is stored.
    It prints the model configurations in a formatted string and saves this string to a file with a timestamp in the specified directory.
    
    Parameters:
        training_parameters (dict): A dictionary containing the model configuration and training parameters
        model_training_output_dir (Path): A pathlib.Path object representing the directory where the model training output is stored.
    
    Returns:
        None
    """
    model_config_string = (f"\n{'#'*64} TRAINING PARAMETERS {'#'*65}\n")
    for key, Value in training_parameters.items():
        model_config_string += f"{key}: {Value}\n"
    model_config_string += (
        f"NOTE: When reloading the model, the learning rate arguments are loaded from the model configuration file instead of the command line arguments.\n"
        f"NOTE: A new model configuration file is created for each run that is started or reloaded to continue training.\n"
    )
    model_config_string += (f"{'#'*150}\n")
    print(model_config_string)
    
    # To save to a file with a timestamp so that we can keep track of the initial model configuration as well as any reloaded models
    config_save_path = model_training_output_dir.joinpath(f"_model_configuration_{get_timestamp()}.txt")
    with open(config_save_path, "w") as file:
        file.write(model_config_string)
    return


def get_memory_usage():
    """
    function to get the current memory usage of the process
    
    Parameters:
        None
    
    Returns:
        memory_mb (float): the memory usage in MB
        memory_gb (float): the memory usage in GB
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
    
    Parameters:
        None
    
    Returns:
        memory_mb (int): the memory usage in MB
        memory_gb (float): the memory usage in GB
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
    function to get a formatted timestamp in the US Central Time Zone for logging purposes that can also be used as a unique identifier in file names
    
    Parameters:
        None
    
    Returns:
        formatted_time (str): the formatted timestamp string
    """
    # Get the current time in the US Central Time Zone with custom formatting
    central_time = datetime.now(pytz.timezone('US/Central'))
    formatted_time = central_time.strftime("%Y-%m-%d__%H-%M-%S")
    return formatted_time


def get_experiment_number(experiment_save_dir):
    """
    function to get a 4-digit experiment number to put at the beginning of the experiment save directory name
    
    Parameters:
        experiment_save_dir (Path): the path to the directory where the experiments are saved
    
    Returns:
        formatted_experiment_number (str): the 4-digit experiment number
    """
    # get the experiment numbers of the directories in the experiment save directory
    experiment_dirs = os.listdir(experiment_save_dir)
    current_experiment_count = len(experiment_dirs)
    this_experiment_number = current_experiment_count + 1
    formatted_experiment_number = f"{this_experiment_number:04}"
    return formatted_experiment_number


def get_readable_time_string(seconds_input: float):
    """
    function to convert seconds (float) to a readable time string in the format HHH:MM:SS.ss
    
    Parameters:
        seconds_input (float): the time in seconds to convert to a readable time string
    
    Returns:
        readable_time_string (str): the readable time string in the format HHH:MM:SS.ss
    """
    # convert to a readable format (HH:MM:SS.ss)
    hours = int(seconds_input // 3600)
    minutes = int((seconds_input % 3600) // 60)
    seconds = seconds_input % 60
    readable_time_string = f"{hours:03}:{minutes:02}:{seconds:05.2f}"
    return readable_time_string


class Terminal_Logger:
    """
    Custom class to redirect the standard output and standard error to a log file while still printing to the terminal. Useful for running scripts on
    remote servers that might close the terminal session and lose the output or for viewing the output of older scripts. If no log file exists, then 
    a new one is created. If a log file exists, then the terminal output is appended to the existing log file. A header is printed to the log file to
    more easily identify when a new terminal output is being logged.
    """
    def __init__(self, log_path):
        """
        log_path: PurePath object representing the full path of the log file
        """
        self.terminal = sys.stdout
        self.log = io.open(log_path, 'a', encoding='utf-8')
        self._setup_logging()
        # print out a header to the log file to indicate that the terminal output is being logged
        print(f"\n\n{'='*150}\n{'='*150}\nLogging terminal output to: {log_path}\nstarting at {get_timestamp()}\n{'='*150}\n{'='*150}\n\n")
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


def print_script_execution_time(script_start_time):
    """
    print the total execution time of the script
    
    Parameters:
        script_start_time (float): the time the script started
    
    Returns:
        None
    """
    script_end_time = time.time()
    time_hours = int((script_end_time - script_start_time) // 3600)
    time_minutes = int(((script_end_time - script_start_time) % 3600) // 60)
    time_seconds = int(((script_end_time - script_start_time) % 3600) % 60)
    print(f"\nTotal execution time: {time_hours:02d}:{time_minutes:02d}:{time_seconds:05.2f}\n")
    return


if __name__ == "__main__":
    for num in range(10):
        print("This script is a utility script and is not meant to be executed")
        time.sleep(0.1)
