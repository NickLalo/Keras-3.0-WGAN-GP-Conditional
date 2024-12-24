"""
Implementation of a WGAN-GP with MNIST from this tutorial:
https://keras.io/examples/generative/wgan_gp/
"""


import os
import time
from pathlib import Path

os.environ["KERAS_BACKEND"] = "tensorflow"

import keras
import tensorflow as tf

# Check if the GPU is available and being used
gpus = tf.config.list_physical_devices('GPU')
print("Is GPU available:", gpus)
print("GPU being used:", tf.test.gpu_device_name())
# Set memory growth on the GPU (so it doesn't use all the memory at once)
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Set the mixed precision policy globally
keras.mixed_precision.set_global_policy("mixed_float16")
print("Mixed precision policy:", keras.mixed_precision.global_policy())

from model_and_callbacks import get_critic_model, get_generator_model, Training_Monitor, WGAN_GP
from utils import parse_arguments, get_timestamp, Terminal_Logger, get_last_checkpoint_paths_for_reload, get_specific_checkpoint_paths_for_reload, \
    print_and_save_training_parameters, print_script_execution_time
from load_data import load_mnist_data_for_gan, visualize_training_samples


# set a random seed for reproducibility
seed = 112
keras.utils.set_random_seed(seed)  # sets random seed for Tensorflow, Numpy, and Python
tf.config.experimental.enable_op_determinism()  # sets the graph-level deterministic operations for Tensorflow (can cause a training slowdown)

# Hardcoded path to the directory containing all model training runs
WGAN_GP_MNIST_MODELS_DIR = Path("wgan_gp_mnist_training_runs")


def load_model_and_data(training_params, WGAN_GP_MNIST_MODELS_DIR):
    """
    Load the model and data for training the WGAN_GP model. The model is either created new one or loaded from a previous run's checkpoint.
    
    Parameters:
        training_params (dict): A dictionary containing the training parameters for the current run.
        WGAN_GP_MNIST_MODELS_DIR (Path): The directory containing all the model training runs.
    
    Returns:
        wgan_gp (WGAN_GP): The WGAN_GP model for training.
        train_dataset (tf.data.Dataset): A tf.data.Dataset object containing the training images and labels.
        model_training_output_dir (Path): The directory where the model and metrics will be saved.
        model_checkpoints_dir (Path): The directory where the model checkpoints will be saved.
        last_checkpoint_dir_path (Path): The directory path of the last checkpoint saved.
        num_classes (int): The number of classes in the dataset.
        samples_per_epoch (int): The number of samples per epoch in the dataset.
    """
    if training_params["fresh_start"]:
        this_training_run_dirname = get_timestamp()  # Get the current time in the US Central Time Zone
        model_training_output_dir = WGAN_GP_MNIST_MODELS_DIR.joinpath(this_training_run_dirname)
        model_checkpoints_dir = model_training_output_dir.joinpath("model_checkpoints")
        os.makedirs(model_checkpoints_dir, exist_ok=True)  # includes the other directories in the path
        
        # create a log file to save the terminal output in the model and metrics directory
        terminal_output_log_filename = model_training_output_dir.joinpath("_terminal_output_logs.txt")
        terminal_logger = Terminal_Logger(terminal_output_log_filename)
        
        # last checkpoint dir set to None to indicate that we are not loading a model from a checkpoint
        last_checkpoint_dir_path = None
        
        # Load the MNIST dataset for training a GAN
        train_dataset, img_shape, num_classes, samples_per_epoch = load_mnist_data_for_gan(training_params["debug_run"], 
            training_params["dataset_subset_percentage"], training_params["batch_size"], training_params["random_shift_frequency"])
        
        # visualize a collection of training samples
        visualize_training_samples(train_dataset, model_training_output_dir)
        
        # initialize the critic and generator models
        critic_model = get_critic_model(img_shape, num_classes, model_training_output_dir)
        generator_model = get_generator_model(training_params["noise_shape"], num_classes, model_training_output_dir)
        
        # Initialize the optimizers
        generator_optimizer = keras.optimizers.Adam(learning_rate=training_params["initial_learning_rate"], beta_1=0.5, beta_2=0.9)
        critic_optimizer = keras.optimizers.Adam(learning_rate=training_params["initial_learning_rate"], beta_1=0.5, beta_2=0.9)
        
        # Get the wgan_gp model
        wgan_gp = WGAN_GP(
            critic=critic_model,
            generator=generator_model,
            num_classes=num_classes,
            latent_dim=training_params["noise_shape"],
            critic_input_shape=img_shape,
            learning_rate=training_params["initial_learning_rate"],
            learning_rate_warmup_epochs=training_params["learning_rate_warmup_epochs"],
            learning_rate_decay=training_params["learning_rate_decay"],
            critic_extra_steps=training_params["critic_to_generator_training_ratio"],
            gp_weight=training_params["gradient_penalty_weight"],
            )
        
        # Compile the wgan_gp model
        wgan_gp.compile(
            critic_optimizer=critic_optimizer,
            gen_optimizer=generator_optimizer,
            )
    elif training_params["reload_last_trained_model"]:
        # find the last model trained on to continue training
        last_model_training_run_dir, model_checkpoints_dir, last_checkpoint_dir_path, \
            model_save_file_path = get_last_checkpoint_paths_for_reload(WGAN_GP_MNIST_MODELS_DIR)
        
        # set the model training output directory to the last model training run directory
        model_training_output_dir = last_model_training_run_dir
        
        # create a log file to save the terminal output in the model and metrics directory
        terminal_output_log_filename = last_model_training_run_dir.joinpath("_terminal_output_logs.txt")
        terminal_logger = Terminal_Logger(terminal_output_log_filename)
        
        # print out info about the model being reloaded
        print(f"loading model from last checkpoint: {last_checkpoint_dir_path}")
        print(f"Model save file path: {model_save_file_path}")
        
        # Load the MNIST dataset for training a GAN
        train_dataset, img_shape, num_classes, samples_per_epoch = load_mnist_data_for_gan(training_params["debug_run"], 
            training_params["dataset_subset_percentage"], training_params["batch_size"])
        
        # load the model from the last checkpoint
        wgan_gp = keras.models.load_model(model_save_file_path)
    elif training_params["reload_path"] is not None:
        # set the model training output directory to the reload path
        model_training_output_dir = training_params["reload_path"]
        
        # find the specific model trained on to continue training
        model_checkpoints_dir, last_checkpoint_dir_path, last_model_save_file_path = \
            get_specific_checkpoint_paths_for_reload(training_params["reload_path"])
        
        # create a log file to save the terminal output in the model and metrics directory
        terminal_output_log_filename = model_training_output_dir.joinpath("_terminal_output_logs.txt")
        terminal_logger = Terminal_Logger(terminal_output_log_filename)
        
        # print out info about the model being reloaded
        print(f"loading model from last checkpoint: {last_checkpoint_dir_path}")
        print(f"Model save file path: {last_model_save_file_path}")
        
        # Load the MNIST dataset for training a GAN
        train_dataset, img_shape, num_classes, samples_per_epoch = load_mnist_data_for_gan(training_params["debug_run"], 
            training_params["dataset_subset_percentage"], training_params["batch_size"])
        
        # load the model from the specific checkpoint
        wgan_gp = keras.models.load_model(last_model_save_file_path)
    else:
        raise ValueError("One of fresh_start, reload_last_trained_model, or reload_path must be set to train the model.")
    
    return wgan_gp, train_dataset, model_training_output_dir, model_checkpoints_dir, last_checkpoint_dir_path, num_classes, samples_per_epoch


if __name__ == "__main__":
    script_start_time = time.time()
    
    # Parse arguments and get a training parameters dictionary for the current run
    training_params = parse_arguments()
    
    # ####################################################################################################
    # # DEBUG: a quick way to test out the model with hardcoded parameters that should be removed later
    # hardcoded_params = True
    # if hardcoded_params:
    #     print("\nUsing HARDCODED custom parameters for the model run.\n")
    #     training_params["fresh_start"] = True
    #     training_params["reload_last_trained_model"] = False
    #     training_params["reload_path"] = None
    #     # model_configurations["reload_path"] = Path("wgan_gp_mnist_training_runs/2024-12-21__06:13:18")
        
    #     # training_params["dataset_subset_percentage"] = 1.0
    #     # training_params["epochs"] = 9999
    # # DEBUG: a quick way to test out the model with hardcoded parameters that should be removed later
    # ####################################################################################################
    
    # get the wgan_gp and training data. Model is either created new one or loaded from a previous run's checkpoint
    wgan_gp, train_dataset, model_training_output_dir, model_checkpoints_dir, last_checkpoint_dir_path, num_classes, samples_per_epoch = \
        load_model_and_data(training_params, WGAN_GP_MNIST_MODELS_DIR)
    
    # Print the training parameters to the terminal
    print_and_save_training_parameters(training_params, model_training_output_dir)
    
    # Initialize a custom training monitor callback to log info about the training process, save checkpoints, and generate validation samples
    training_monitor_callback = Training_Monitor(
        model_training_output_dir,
        model_checkpoints_dir,
        latent_dim=training_params["noise_shape"],
        samples_per_epoch=samples_per_epoch,
        gif_and_model_save_frequency=training_params["gif_and_model_save_frequency"],
        last_checkpoint_dir_path=last_checkpoint_dir_path
        )
    
    # Start training
    wgan_gp.fit(
        train_dataset,
        batch_size=training_params["batch_size"],
        epochs=training_params["epochs"],
        callbacks=[
            training_monitor_callback,
            ]
        )
    
    print_script_execution_time(script_start_time)
