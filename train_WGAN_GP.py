"""
Implementing a WGAN-GP from this tutorial: https://keras.io/examples/generative/wgan_gp/ with a different model and MNIST instead of fashion MNIST.
"""


import os
import time
import shutil
from pathlib import Path

os.environ["KERAS_BACKEND"] = "tensorflow"

import numpy as np
import keras
import tensorflow as tf
from keras import layers
import matplotlib.pyplot as plt

from model_and_callbacks import get_discriminator_model, get_generator_model, Training_Monitor, WGAN_GP, discriminator_loss, generator_loss, \
get_discriminator_optimizer, get_generator_optimizer
from utils import print_fresh_start_warning_message, get_last_checkpoint_dir_and_file


if __name__ == "__main__":
    script_start_time = time.time()
    #-------------------------------
    ### HYPERPARAMETERS
    #-------------------------------
    batch_size = 256
    # Size of the noise vector
    noise_dim = 128
    
    # TODO: add a debug_run flag to set a small number of epochs and a small percentage of the data to train on
    # Set the number of epochs for training.
    epochs = 5  # REVIEW: for now, train for a very small number of epochs
    
    # ensure the model_training_output and model_checkpoints directories exist
    model_training_output_dir = Path("model_training_output")
    model_checkpoints_dir = model_training_output_dir.joinpath("model_checkpoints")
    os.makedirs(model_training_output_dir, exist_ok=True)
    os.makedirs(model_checkpoints_dir, exist_ok=True)
    
    fresh_start = True
    # delete ALL files in the model_training_output directory for a fresh start
    if fresh_start:
        # print out a big warning message that gives some time to cancel the operation
        print_fresh_start_warning_message()
        
        # reset the model_training_output directory
        shutil.rmtree(model_training_output_dir)
        os.makedirs(model_training_output_dir, exist_ok=True)
        os.makedirs(model_checkpoints_dir, exist_ok=True)
        last_checkpoint_dir_path = None
    
    # Load the MNIST dataset
    (train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()
    # determine the shape of a single image sample
    img_shape = train_images.shape[1:]
    
    # if the img shape is 2 dimensional, add a singleton layer to the end
    if len(img_shape) == 2:
        img_shape = img_shape + (1,)
    # MNIST image shape should be: (28, 28, 1)
    
    # add the test data to the train data
    train_images = np.concatenate((train_images, test_images), axis=0)
    train_labels = np.concatenate((train_labels, test_labels), axis=0)
    
    # Scale the images to the range [-1, 1]
    # Original pixel values are in the range [0, 255].
    # To scale them to [-1, 1], we subtract 127.5 (which centers the values around 0)
    # and then divide by 127.5 (which scales the values to be between -1 and 1).
    train_images = (train_images.astype(np.float32) - 127.5) / 127.5
    
    # add a singleton layer to the end of the train images
    train_images = np.expand_dims(train_images, axis=-1)
    
    # TODO: add a debug_run flag to set a small number of epochs and a small percentage of the data to train on
    # REVIEW: for now, train using only 10% of the data
    percentage = 0.1
    train_images = train_images[:int(percentage * len(train_images))]
    train_labels = train_labels[:int(percentage * len(train_labels))]
    
    # Print the shapes to verify
    print("Shape of train_images:", train_images.shape)
    print("Shape of train_labels:", train_labels.shape)
    
    # Check that all labels are the same
    unique_labels = np.unique(train_labels)
    print("Unique labels in the dataset:", unique_labels)
    
    disc_model = get_discriminator_model(img_shape)
    # disc_model.summary()
    
    gen_model = get_generator_model(noise_dim)
    # gen_model.summary()
    
    # REVIEW: putting these inside of functions so that I can serialize them with the model?
    # # Instantiate the optimizer for both networks
    # generator_optimizer = keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9)
    # discriminator_optimizer = keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9)
    
    # get the optimizers for both networks
    discriminator_optimizer = get_discriminator_optimizer()
    generator_optimizer = get_generator_optimizer()
    
    # Initialize a custom training monitor callback to log info about the training process, save checkpoints, and generate validation samples
    training_monitor_callback = Training_Monitor(
        model_training_output_dir, 
        model_checkpoints_dir, 
        num_img=20, 
        latent_dim=noise_dim, 
        grid_size=(4, 5), 
        samples_per_epoch=train_images.shape[0],
        last_checkpoint_dir_path=last_checkpoint_dir_path
    )
    
    # Get the wgan_gp model
    wgan_gp = WGAN_GP(
        discriminator=disc_model,
        generator=gen_model,
        latent_dim=noise_dim,
        discriminator_extra_steps=5,
        disc_optimizer=discriminator_optimizer,
        gen_optimizer=generator_optimizer,
        disc_loss_fn=discriminator_loss,
        gen_loss_fn=generator_loss,
        )
    # Compile the wgan_gp model
    wgan_gp.compile()
    
    # Start training
    wgan_gp.fit(
        train_images, 
        batch_size=batch_size,
        epochs=epochs, 
        callbacks=[
            training_monitor_callback,
            ]
        )
    
    ############################################################ First Model Reload ############################################################
    # # naive approach: load the model using the keras load method
    last_checkpoint_dir_path, last_model_checkpoint_path = get_last_checkpoint_dir_and_file(model_checkpoints_dir)
    loaded_wgan_gp = keras.models.load_model(last_model_checkpoint_path)
    loaded_wgan_gp.compile()
    
    # Initialize a custom training monitor callback to log info about the training process, save checkpoints, and generate validation samples
    training_monitor_callback = Training_Monitor(
        model_training_output_dir, 
        model_checkpoints_dir, 
        num_img=20, 
        latent_dim=noise_dim, 
        grid_size=(4, 5), 
        samples_per_epoch=train_images.shape[0],
        last_checkpoint_dir_path=last_checkpoint_dir_path
    )
    
    # Start training
    loaded_wgan_gp.fit(
        train_images,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[
            training_monitor_callback,
            ]
        )
    
    ############################################################ Second Model Reload ############################################################
    # # naive approach: load the model using the keras load method
    last_checkpoint_dir_path, last_model_checkpoint_path = get_last_checkpoint_dir_and_file(model_checkpoints_dir)
    second_loaded_wgan_gp = keras.models.load_model(last_model_checkpoint_path)
    second_loaded_wgan_gp.compile()
    
    # Initialize a custom training monitor callback to log info about the training process, save checkpoints, and generate validation samples
    training_monitor_callback = Training_Monitor(
        model_training_output_dir, 
        model_checkpoints_dir, 
        num_img=20, 
        latent_dim=noise_dim, 
        grid_size=(4, 5), 
        samples_per_epoch=train_images.shape[0],
        last_checkpoint_dir_path=last_checkpoint_dir_path
    )
    
    # Start training
    second_loaded_wgan_gp.fit(
        train_images,
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
