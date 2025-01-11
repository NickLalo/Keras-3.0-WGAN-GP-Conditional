"""
Implementation of a WGAN-GP with MNIST from this tutorial:
https://keras.io/examples/generative/wgan_gp/
"""


import os
import time
import numpy as np

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

from critic_and_generator_models import get_critic_model, get_generator_model
from load_data import load_mnist_data_for_gan, visualize_training_samples
from wgan_gp_model import WGAN_GP
from training_monitor_callback import Training_Monitor
from utils import parse_arguments, get_all_model_runs_dir, get_timestamp, Terminal_Logger, get_last_checkpoint_paths_for_reload, \
    get_specific_checkpoint_paths_for_reload, print_and_save_training_parameters, print_script_execution_time, get_experiment_number, \
    get_last_model_save_dir_path, backup_model_code


# set a random seed for reproducibility for Tensorflow, Numpy, and Python
seed = 112
keras.utils.set_random_seed(seed)
# sets the graph-level deterministic operations for Tensorflow (can cause a training slowdown)
tf.config.experimental.enable_op_determinism()
# Create a global TensorFlow random generator with a fixed seed for the DataAugmentor class
generator = tf.random.Generator.from_seed(seed)
tf.random.set_global_generator(generator)


def load_model_and_data(training_params):
    """
    Load the model and data for training the WGAN_GP model. The model is either created new one or loaded from a previous run's checkpoint.
    
    Parameters:
        training_params (dict): A dictionary containing the training parameters for the current run.
    
    Returns:
        wgan_gp (WGAN_GP): The WGAN_GP model for training.
        train_dataset (tf.data.Dataset): A tf.data.Dataset object containing the training images and labels.
        model_training_output_dir (Path): The directory where the model and metrics will be saved.
        model_checkpoints_dir (Path): The directory where the model checkpoints will be saved.
        last_checkpoint_dir_path (Path): The directory path of the last checkpoint saved.
        num_classes (int): The number of classes in the dataset.
    """
    # get the path to the directory containing all the model training runs. Setups a default directory if it doesn't exist
    all_model_runs_dir = get_all_model_runs_dir()
    
    if training_params["fresh_start"]:
        run_timestamp = get_timestamp()  # Get the current time in the US Central Time Zone
        experiment_number = get_experiment_number(all_model_runs_dir)
        this_training_run_dirname = f"{run_timestamp}__{experiment_number}"
        model_training_output_dir = all_model_runs_dir.joinpath(this_training_run_dirname)
        model_checkpoints_dir = model_training_output_dir.joinpath("model_checkpoints")
        os.makedirs(model_checkpoints_dir, exist_ok=True)  # includes the other directories in the path
        
        # create a log file to save the terminal output in the model and metrics directory
        terminal_output_log_filename = model_training_output_dir.joinpath("_terminal_output_logs.txt")
        _ = Terminal_Logger(terminal_output_log_filename)  # don't need to save the logger object
        
        # last checkpoint dir set to None to indicate that we are not loading a model from a checkpoint
        last_checkpoint_dir_path = None
        
        # Load the MNIST dataset for training a WGAN
        train_dataset, img_shape, num_classes, samples_per_epoch = load_mnist_data_for_gan(training_params["debug_run"], 
            training_params["dataset_subset_percentage"], training_params["batch_size"], training_params["random_rotate_frequency"],
            training_params["random_translate_frequency"], training_params["random_zoom_frequency"])
        
        # visualize a collection of training samples
        visualize_training_samples(train_dataset, model_training_output_dir)
        
        # initialize the critic and generator models
        critic_model = get_critic_model(img_shape, num_classes, model_training_output_dir)
        generator_model = get_generator_model(training_params["noise_shape"], num_classes, model_training_output_dir)
        
        # save copy of the critic and generator code to the model training output directory
        backup_model_code(model_training_output_dir)
        
        # Initialize the optimizers
        generator_optimizer = keras.optimizers.Adam(learning_rate=training_params["initial_generator_learning_rate"], beta_1=0.5, beta_2=0.9)
        critic_optimizer = keras.optimizers.Adam(learning_rate=training_params["initial_critic_learning_rate"], beta_1=0.5, beta_2=0.9)
        
        # initialize wgan_gp model which will train the critic and generator models
        wgan_gp = WGAN_GP(
            critic=critic_model,
            generator=generator_model,
            latent_dim=training_params["noise_shape"],
            critic_learning_rate=training_params["initial_critic_learning_rate"],
            generator_learning_rate=training_params["initial_generator_learning_rate"],
            learning_rate_warmup_epochs=training_params["learning_rate_warmup_epochs"],
            learning_rate_decay=training_params["learning_rate_decay"],
            min_critic_learning_rate=training_params["min_critic_learning_rate"],
            min_generator_learning_rate=training_params["min_generator_learning_rate"],
            critic_extra_steps=training_params["critic_to_generator_training_ratio"],
            gp_weight=training_params["gradient_penalty_weight"],
            )
        
        # Compile the wgan_gp model (adds the optimizers to the wgan_gp model)
        wgan_gp.compile(
            critic_optimizer=critic_optimizer,
            gen_optimizer=generator_optimizer,
            )
    else:
        if training_params["reload_last_trained_model"]:
            # find the last model trained on to continue training
            last_model_training_run_dir, model_checkpoints_dir, last_checkpoint_dir_path, \
                model_save_file_path = get_last_checkpoint_paths_for_reload(all_model_runs_dir)
            
            # set the model training output directory to the last model training run directory
            model_training_output_dir = last_model_training_run_dir
        elif training_params["reload_path"] is not None:
            # set the model training output directory to the reload path
            model_training_output_dir = training_params["reload_path"]
            
            # find the specific model trained on to continue training
            model_checkpoints_dir, last_checkpoint_dir_path, model_save_file_path = \
                get_specific_checkpoint_paths_for_reload(training_params["reload_path"])
        
        # create a log file to save the terminal output in the model training output directory
        terminal_output_log_filename = model_training_output_dir.joinpath("_terminal_output_logs.txt")
        _ = Terminal_Logger(terminal_output_log_filename)  # don't need to save the logger object
        
        # Load the MNIST dataset for training a WGAN
        train_dataset, img_shape, num_classes, samples_per_epoch = load_mnist_data_for_gan(training_params["debug_run"], 
            training_params["dataset_subset_percentage"], training_params["batch_size"], training_params["random_rotate_frequency"], 
            training_params["random_translate_frequency"], training_params["random_zoom_frequency"])
        
        # print out info about the model being reloaded
        print(f"loading model from checkpoint: {last_checkpoint_dir_path}")
        print(f"Model save file path: {model_save_file_path}")
        
        # load the model from the last checkpoint
        wgan_gp = keras.models.load_model(model_save_file_path)
    
    # Print the training parameters to the terminal and save them to a file
    print_and_save_training_parameters(training_params, model_training_output_dir)
    
    return wgan_gp, train_dataset, model_training_output_dir, model_checkpoints_dir, last_checkpoint_dir_path, num_classes, samples_per_epoch


def model_reload_test(wgan_gp, model_checkpoints_dir):
    """
    Tests the consistency of the WGAN-GP model by comparing the outputs of the 
    generator before and after reloading the model from the last checkpoint.
    Parameters:
        wgan_gp: The trained WGAN-GP model instance.
        model_checkpoints_dir (str): Directory path where model checkpoints are saved.
    Returns:
        None. Prints whether the model reload test passed or failed based on 
        the comparison of generated images.
    """
    # generate some images with the trained model
    noise = tf.random.normal([10, training_params["noise_shape"]])
    labels = tf.convert_to_tensor([x for x in range(10)], dtype=tf.int32)
    generated_image = wgan_gp.generator.predict([noise, labels])
    
    # reload the model and regenerate the images
    last_model_save_dir_path = get_last_model_save_dir_path(model_checkpoints_dir)
    if not last_model_save_dir_path.exists():
        print(f"{'#'*54} Save-Reload Output Integrity Test Failed {'#'*54}")
        print(f"    {'  ERROR  '*16}")
        print(f"please investigate why the path to the last model save directory does not exist.\n")
        return
    
    reloaded_wgan_gp = keras.models.load_model(last_model_save_dir_path)
    reloaded_generated_image = reloaded_wgan_gp.generator.predict([noise, labels])
    
    # compare the generated images between the trained model and the reloaded model with a numpy all close check
    all_close_results = np.allclose(generated_image, reloaded_generated_image, atol=1e-6)
    if all_close_results:
        print(f"{'#'*53} Save-Reload Output Integrity Test Passed! {'#'*54}\n")
    else:
        print(f"{'#'*54} Save-Reload Output Integrity Test Failed {'#'*54}")
        print(f"    {'  ERROR  '*16}")
        print(f"please investigate why the reloaded model did not pass the np.allclose test before continuing to use the model.\n")
    return


if __name__ == "__main__":
    script_start_time = time.time()
    
    # Parse arguments and get a training parameters dictionary for the current run
    training_params = parse_arguments()
    
    # get the wgan_gp and training data. Model is either created new one or loaded from a previous run's checkpoint
    wgan_gp, train_dataset, model_training_output_dir, model_checkpoints_dir, last_checkpoint_dir_path, num_classes, samples_per_epoch \
        = load_model_and_data(training_params)
    
    # Initialize a custom training monitor callback to log info about the training process, save checkpoints, and generate validation samples
    training_monitor_callback = Training_Monitor(
        model_training_output_dir,
        model_checkpoints_dir,
        noise_dim=training_params["noise_shape"],
        model_save_frequency=training_params["model_save_frequency"],
        video_of_validation_frequency=training_params["video_of_validation_frequency"],
        FID_score_frequency=training_params["FID_score_frequency"],
        train_dataset=train_dataset,
        samples_per_epoch=samples_per_epoch,
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
    
    # Test that the model can be reloaded and generate the same output
    model_reload_test(wgan_gp, model_checkpoints_dir)
    
    print_script_execution_time(script_start_time)
