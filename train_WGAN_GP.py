"""
Implementation of a WGAN-GP with MNIST from this tutorial:
https://keras.io/examples/generative/wgan_gp/
"""


import os
import time
import shutil
from pathlib import Path

os.environ["KERAS_BACKEND"] = "tensorflow"

import keras
import tensorflow as tf

# Set memory growth on the GPU (so it doesn't use all the memory at once)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

from model_and_callbacks import get_critic_model, get_generator_model, Training_Monitor, WGAN_GP
from utils import print_fresh_start_warning_message, get_last_checkpoint_dir_and_file, parse_arguments
from load_data import load_mnist_data_for_gan


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
    
    # TODO: remove these hardcoded values after testing code
    fresh_start = True
    dataset_subset_percentage = 0.2
    epochs = 500
    
    # ensure the model_training_output and model_checkpoints directories exist
    model_training_output_dir = Path("model_training_output")
    model_checkpoints_dir = model_training_output_dir.joinpath("model_checkpoints")
    os.makedirs(model_training_output_dir, exist_ok=True)
    os.makedirs(model_checkpoints_dir, exist_ok=True)
    
    # delete ALL files in the model_training_output directory for a fresh start
    if fresh_start:
        # print out a big warning message that gives some time to cancel before the files are deleted
        # print_fresh_start_warning_message()  # TODO: put this back after testing code
        
        # reset the model_training_output directory
        shutil.rmtree(model_training_output_dir)
        os.makedirs(model_training_output_dir, exist_ok=True)
        os.makedirs(model_checkpoints_dir, exist_ok=True)
        last_checkpoint_dir_path = None
        last_model_checkpoint_path = None
        print("Fresh start: Deleted all files in the model_training_output directory.")
    else:
        # populate the last checkpoint directory and file paths for model reloading
        last_checkpoint_dir_path, last_model_checkpoint_path = get_last_checkpoint_dir_and_file(model_checkpoints_dir)
    
    # Load the MNIST dataset for training a GAN
    train_dataset, img_shape, num_classes, samples_per_epoch = load_mnist_data_for_gan(debug_run, dataset_subset_percentage, batch_size)
    
    # initialize the critic and generator models
    critic_model = get_critic_model(img_shape, num_classes, model_training_output_dir)
    generator_model = get_generator_model(noise_shape, num_classes, model_training_output_dir)
    
    # REVIEW: there are better hardcoded values for the betas that I should try out first
    # REVIEW: consider making the learning rate higher for the critic (I think that is the one? look in the notes and in the book)
    # Instantiate the optimizers
    generator_optimizer = keras.optimizers.Adam(learning_rate=0.00005, beta_1=0.5, beta_2=0.9)
    critic_optimizer = keras.optimizers.Adam(learning_rate=0.00005, beta_1=0.5, beta_2=0.9)
    
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
    
    # get the wgan_gp model by either creating a new one or loading the last model checkpoint
    if fresh_start:
        # Get the wgan_gp model
        wgan_gp = WGAN_GP(
            critic=critic_model,
            generator=generator_model ,
            num_classes=num_classes,
            latent_dim=noise_shape,
            critic_input_shape=img_shape,
            critic_extra_steps=5,
            gp_weight=10.0,
            )
        
        # Compile the wgan_gp model
        wgan_gp.compile(
            critic_optimizer=critic_optimizer,
            gen_optimizer=generator_optimizer,
            )
    else:
        # load the last model checkpoint from the last training session
        wgan_gp = keras.models.load_model(last_model_checkpoint_path)
    
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
