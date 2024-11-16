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
        # TODO: have a fun !FRESH START! message print out to give me a few seconds to stop it if I don't want to delete everything
        #       currently working on this in prototype_ideas_01.py
        
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
    
    # make the data folder if it doesn't exist
    if not os.path.exists("data"):
        os.makedirs("data")
    # delete all of the files in the data folder
    for file in os.listdir("data"):
        os.remove(f"data/{file}")
    
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
        discriminator_extra_steps=5,  # was set to 3, but I think 5 is recommended
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
    
    # print("\n######## saving model...")
    # # save the model (This will throw a warning saying that our model is not built, but that's ok because our model has two sub-models that are built)
    # wgan_gp.save("wgan_gp_model.keras")
    
    #################################################################################################################################################
    ######################### write a function to get the path for the last model checkpoint dir and last model checkpoint file
    print("\n######## loading model to pick up training from where we left off...")
    # TODO: write a function for finding the latest model checkpoint and loading the model from that checkpoint
    sorted_checkpoint_dirs = sorted(os.listdir(model_checkpoints_dir))
    last_checkpoint_dir = sorted_checkpoint_dirs[-1]
    
    # list the files in the last checkpoint directory
    last_checkpoint_dir_path = model_checkpoints_dir.joinpath(last_checkpoint_dir)
    last_checkpoint_files = os.listdir(last_checkpoint_dir_path)
    # get the filename that ends with .keras
    last_model_checkpoint_filename = [file for file in last_checkpoint_files if file.endswith(".keras")][0]
    last_model_checkpoint_path = last_checkpoint_dir_path.joinpath(last_model_checkpoint_filename)
    # RETURN last_checkpoint_dir_path, last_model_checkpoint_path
    #################################################################################################################################################
    # # naive approach: load the model using the keras load method
    loaded_wgan_gp = keras.models.load_model(last_model_checkpoint_path)
    loaded_wgan_gp.compile()
    
    # check that the models are the same
    random_latent_vectors = tf.random.normal(shape=(20, noise_dim))
    assert np.allclose(wgan_gp.generator(random_latent_vectors, training=False), loaded_wgan_gp.generator(random_latent_vectors, training=False))
    print("Model save/load all close check passed!")
    
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
    
    #################################################################################################################################################
    ######################### write a function to get the path for the last model checkpoint dir and last model checkpoint file
    print("\n######## loading model to pick up training from where we left off...")
    # TODO: write a function for finding the latest model checkpoint and loading the model from that checkpoint
    sorted_checkpoint_dirs = sorted(os.listdir(model_checkpoints_dir))
    last_checkpoint_dir = sorted_checkpoint_dirs[-1]
    
    # list the files in the last checkpoint directory
    last_checkpoint_dir_path = model_checkpoints_dir.joinpath(last_checkpoint_dir)
    last_checkpoint_files = os.listdir(last_checkpoint_dir_path)
    # get the filename that ends with .keras
    last_model_checkpoint_filename = [file for file in last_checkpoint_files if file.endswith(".keras")][0]
    last_model_checkpoint_path = last_checkpoint_dir_path.joinpath(last_model_checkpoint_filename)
    # RETURN last_checkpoint_dir_path, last_model_checkpoint_path
    #################################################################################################################################################
    # # naive approach: load the model using the keras load method
    second_loaded_wgan_gp = keras.models.load_model(last_model_checkpoint_path)
    second_loaded_wgan_gp.compile()
    
    # check that the models are the same
    random_latent_vectors = tf.random.normal(shape=(20, noise_dim))
    assert np.allclose(second_loaded_wgan_gp.generator(random_latent_vectors, training=False), loaded_wgan_gp.generator(random_latent_vectors, training=False))
    print("Model save/load all close check passed!")
    
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
    
    # # save the model
    # loaded_wgan_gp.save("wgan_gp_model.keras")
    
    # print("\n######## loading model a second time to pick up training from where we left off...")#
    
    # # naive approach: load the model using the keras load method
    # second_loaded_wgan_gp = keras.models.load_model("wgan_gp_model.keras")
    # second_loaded_wgan_gp.compile()
    
    # # Initialize a custom LossLogger callback to log metrics at the end of every epoch
    # training_monitor_callback = Training_Monitor(num_img=20, latent_dim=noise_dim, grid_size=(4, 5), samples_per_epoch=train_images.shape[0])
    
    # # Start training
    # second_loaded_wgan_gp.fit(
    #     train_images, 
    #     batch_size=batch_size,
    #     epochs=epochs, 
    #     callbacks=[
    #         training_monitor_callback,
    #         ]
    #     )
    
    # # save the model
    # second_loaded_wgan_gp.save("wgan_gp_model.keras")
    
    script_end_time = time.time()
    time_hours = int((script_end_time - script_start_time) // 3600)
    time_minutes = int(((script_end_time - script_start_time) % 3600) // 60)
    time_seconds = int(((script_end_time - script_start_time) % 3600) % 60)
    print(f"\nTotal execution time: {time_hours:02d}:{time_minutes:02d}:{time_seconds:02d}")
