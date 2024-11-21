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

from model_and_callbacks import get_discriminator_model, get_generator_model, Training_Monitor, WGAN_GP, discriminator_loss, generator_loss
from utils import print_fresh_start_warning_message, get_last_checkpoint_dir_and_file, parse_arguments
from load_data import load_mnist_data_for_gan


if __name__ == "__main__":
    script_start_time = time.time()
    
    # Parse arguments to get the hyperparameters for the model run
    args = parse_arguments()
    batch_size = args.batch_size
    noise_dim = args.noise_dim
    debug_run = args.debug_run
    fresh_start = args.fresh_start
    small_subset = args.small_subset
    
    # TODO: remove this after testing out some code
    fresh_start = True
    debug_run = True
    
    # Set the number of epochs for the model run
    if debug_run:
        print("DEBUG MODE: Running with a small number of epochs.")
        epochs = 3
    else:
        epochs = args.epochs
    
    # ensure the model_training_output and model_checkpoints directories exist
    model_training_output_dir = Path("model_training_output")
    model_checkpoints_dir = model_training_output_dir.joinpath("model_checkpoints")
    os.makedirs(model_training_output_dir, exist_ok=True)
    os.makedirs(model_checkpoints_dir, exist_ok=True)
    
    # delete ALL files in the model_training_output directory for a fresh start
    if fresh_start:
        # print out a big warning message that gives some time to cancel the operation
        # TODO: uncomment this after testing out some code
        # print_fresh_start_warning_message()
        
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
    train_dataset, img_shape = load_mnist_data_for_gan(debug_run=debug_run, small_subset=small_subset, batch_size=batch_size, verbose=True)
    
    disc_model = get_discriminator_model(img_shape)
    # disc_model.summary()
    
    gen_model = get_generator_model(noise_dim)
    # gen_model.summary()
    
    # Initialize a custom training monitor callback to log info about the training process, save checkpoints, and generate validation samples
    training_monitor_callback = Training_Monitor(
        model_training_output_dir,
        model_checkpoints_dir,
        num_img=20,
        latent_dim=noise_dim,
        grid_size=(4, 5),
        samples_per_epoch=len(train_dataset),
        last_checkpoint_dir_path=last_checkpoint_dir_path
        )
    
    # get the wgan_gp model by either creating a new one or loading the last model checkpoint
    if fresh_start:
        # Get the wgan_gp model
        wgan_gp = WGAN_GP(
            discriminator=disc_model,
            generator=gen_model,
            latent_dim=noise_dim,
            discriminator_input_shape=img_shape,
            discriminator_extra_steps=5,
            )
        
        # Instantiate the optimizer for both networks
        generator_optimizer = keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9)
        discriminator_optimizer = keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9)
        
        # Compile the wgan_gp model
        wgan_gp.compile(
            disc_optimizer=discriminator_optimizer,
            gen_optimizer=generator_optimizer,
            disc_loss_fn=discriminator_loss,
            gen_loss_fn=generator_loss,
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
