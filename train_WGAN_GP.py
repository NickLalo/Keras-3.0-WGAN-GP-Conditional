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

# set a random seed for reproducibility
seed = 112
keras.utils.set_random_seed(seed)  # sets random seed for Tensorflow, Numpy, and Python
tf.config.experimental.enable_op_determinism()  # sets the graph-level deterministic operations for Tensorflow (can cause a training slowdown)

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

from model_and_callbacks import get_critic_model, get_generator_model, Training_Monitor, WGAN_GP
from utils import get_last_checkpoint_dir_and_file, parse_arguments, get_timestamp
from load_data import load_mnist_data_for_gan


# Hardcoded path to the directory containing all model training runs
WGAN_GP_MNIST_MODELS_DIR = Path("wgan_gp_mnist_training_runs")


if __name__ == "__main__":
    script_start_time = time.time()
    
    # Parse arguments to get the hyperparameters for the model run
    args = parse_arguments()
    batch_size = args.batch_size
    noise_shape = args.noise_shape
    debug_run = args.debug_run
    fresh_start = args.fresh_start
    dataset_subset_percentage = args.dataset_subset_percentage
    
    # Set the number of epochs for the model run
    if debug_run:
        print("\nDEBUG MODE: Running with a small number of epochs.\n")
        epochs = 3
    else:
        epochs = args.epochs
    
    # TODO: move these hardcoded values to the argument parser
    fresh_start = True
    dataset_subset_percentage = 0.5
    epochs = 50
    batch_size = 64
    # the number of times the critic is trained for every time the generator is trained
    critic_to_generator_training_ratio = 5
    initial_learning_rate = 0.00001
    learning_rate_warmup_epochs=85
    learning_rate_decay=0.998
    
    # Load the MNIST dataset for training a GAN
    train_dataset, img_shape, num_classes, samples_per_epoch = load_mnist_data_for_gan(debug_run, dataset_subset_percentage, batch_size)
    
    # get the wgan_gp model by either creating a new one or loading the last model checkpoint
    if fresh_start:
        this_training_run_dirname = get_timestamp()  # Get the current time in the US Central Time Zone
        model_training_output_dir = WGAN_GP_MNIST_MODELS_DIR.joinpath(this_training_run_dirname)
        model_checkpoints_dir = model_training_output_dir.joinpath("model_checkpoints")
        os.makedirs(model_checkpoints_dir, exist_ok=True)  # includes the other directories in the path
        
        # last checkpoint dir set to None to indicate that we are not loading a model from a checkpoint
        last_checkpoint_dir_path = None
        
        # initialize the critic and generator models
        critic_model = get_critic_model(img_shape, num_classes, model_training_output_dir)
        generator_model = get_generator_model(noise_shape, num_classes, model_training_output_dir)
        
        # Initialize the optimizers
        """
        Currently working on implementing this. The learning rate is set here for a fresh run
        - I will be updating this value in the Training_Monitor class at the end of each epoch where I will directly modify the value of optimizer's 
            learning rate
                - at the end of each epoch check if we are past some set warmup_step or warmup_epoch count
                - if no, do nothing
                - if yes, calculate a new learning rate for both models, which will be the same value, and update a WGAN_GP attribute as well as 
                    both of the optimizers learning rates
        - I need to keep track of this information in the WGAN_GP class as well AND will have to change how the model loading works
            - the learning rate is an attribute of the WGAN_GP
            - when saving the model save this attribute along with the optimizers
            - when loading the model load this attribute along with the optimizers
                - When testing out the code check that these two values are the same when loading
        """
        generator_optimizer = keras.optimizers.Adam(learning_rate=initial_learning_rate, beta_1=0.5, beta_2=0.9)
        critic_optimizer = keras.optimizers.Adam(learning_rate=initial_learning_rate, beta_1=0.5, beta_2=0.9)
        
        # Get the wgan_gp model
        wgan_gp = WGAN_GP(
            critic=critic_model,
            generator=generator_model,
            num_classes=num_classes,
            latent_dim=noise_shape,
            critic_input_shape=img_shape,
            learning_rate=initial_learning_rate,
            learning_rate_warmup_epochs=learning_rate_warmup_epochs,
            learning_rate_decay=learning_rate_decay,
            critic_extra_steps=5,
            gp_weight=10.0,
            )
        
        # Compile the wgan_gp model
        wgan_gp.compile(
            critic_optimizer=critic_optimizer,
            gen_optimizer=generator_optimizer,
            )
    else:
        # TODO: add in the ability to load from a specific model/checkpoint to continue training
        # find the last model trained on and the last checkpoint saved to load the model to continue training
        model_training_output_dir, last_checkpoint_dir_path = get_last_checkpoint_dir_and_file(WGAN_GP_MNIST_MODELS_DIR)
        print(f"loading model from last checkpoint: {last_checkpoint_dir_path}")
        
        # find the .keras file in the last checkpoint directory
        last_checkpoint_dir_files = os.listdir(last_checkpoint_dir_path)
        model_save_file = [file for file in last_checkpoint_dir_files if file.endswith(".keras")][0]
        model_save_file_path = last_checkpoint_dir_path.joinpath(model_save_file)
        print(f"Model save file path: {model_save_file_path}")
        
        # get the model checkpoints dir from the model_training_output_dir
        model_checkpoints_dir = model_training_output_dir.joinpath("model_checkpoints")
        
        # load the last model checkpoint from the last training session
        wgan_gp = keras.models.load_model(model_save_file_path)
    
    # Initialize a custom training monitor callback to log info about the training process, save checkpoints, and generate validation samples
    training_monitor_callback = Training_Monitor(
        model_training_output_dir,
        model_checkpoints_dir,
        num_classes=num_classes,
        num_img=20,
        latent_dim=noise_shape,
        grid_size=(4, 5),
        samples_per_epoch=samples_per_epoch,
        last_checkpoint_dir_path=last_checkpoint_dir_path
        )
    
    # Start training
    wgan_gp.fit(
        train_dataset,
        batch_size=batch_size,
        epochs=epochs, 
        callbacks=[
            training_monitor_callback,
            ]
        )
    
    script_end_time = time.time()
    time_hours = int((script_end_time - script_start_time) // 3600)
    time_minutes = int(((script_end_time - script_start_time) % 3600) // 60)
    time_seconds = int(((script_end_time - script_start_time) % 3600) % 60)
    print(f"\nTotal execution time: {time_hours:02d}:{time_minutes:02d}:{time_seconds:02d}")
