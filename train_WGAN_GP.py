"""
Implementing a WGAN-GP from this tutorial: https://keras.io/examples/generative/wgan_gp/ with a different model and MNIST instead of fashion MNIST.
"""


import os
import time

os.environ["KERAS_BACKEND"] = "tensorflow"

import numpy as np
import keras
import tensorflow as tf
from keras import layers
import matplotlib.pyplot as plt

from model_and_callbacks import get_discriminator_model, get_generator_model, GANMonitor, WGAN


if __name__ == "__main__":
    script_start_time = time.time()
    #-------------------------------
    ### HYPERPARAMETERS
    #-------------------------------
    batch_size = 256
    # Size of the noise vector
    noise_dim = 128
    # Set the number of epochs for training.
    epochs = 2000
    
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
    
    # Print the shapes to verify
    print("Shape of train_images_7:", train_images.shape)
    print("Shape of train_labels_7:", train_labels.shape)
    
    # Check that all labels are the same
    unique_labels = np.unique(train_labels)
    print("Unique labels in the dataset:", unique_labels)
    
    # make the data folder if it doesn't exist
    if not os.path.exists("data"):
        os.makedirs("data")
    # delete all of the files in the data folder
    for file in os.listdir("data"):
        os.remove(f"data/{file}")
    
    d_model = get_discriminator_model(img_shape)
    # d_model.summary()
    
    g_model = get_generator_model(noise_dim)
    # g_model.summary()
    
    # Instantiate the optimizer for both networks
    # (learning_rate=0.0002, beta_1=0.5 are recommended)
    generator_optimizer = keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9)
    discriminator_optimizer = keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9)
    
    # Define the loss functions for the discriminator,
    # which should be (fake_loss - real_loss).
    # We will add the gradient penalty later to this loss function.
    def discriminator_loss(real_img, fake_img):
        real_loss = tf.reduce_mean(real_img)
        fake_loss = tf.reduce_mean(fake_img)
        return fake_loss - real_loss
    
    # Define the loss functions for the generator.
    def generator_loss(fake_img):
        return -tf.reduce_mean(fake_img)
    
    # Instantiate the customer `GANMonitor` Keras callback.
    callback = GANMonitor(num_img=20, latent_dim=noise_dim, grid_size=(4, 5))
    
    # Get the wgan model
    wgan = WGAN(
        discriminator=d_model,
        generator=g_model,
        latent_dim=noise_dim,
        discriminator_extra_steps=5,  # was set to 3, but I think 5 is recommended
        batch_size=batch_size
        )
    
    # Compile the wgan model
    wgan.compile(
        d_optimizer=discriminator_optimizer,
        g_optimizer=generator_optimizer,
        g_loss_fn=generator_loss,
        d_loss_fn=discriminator_loss,
        )
    
    # Start training
    wgan.fit(
        train_images, 
        batch_size=batch_size, 
        epochs=epochs, 
        callbacks=[callback]
        )
    
    script_end_time = time.time()
    time_hours = int((script_end_time - script_start_time) // 3600)
    time_minutes = int(((script_end_time - script_start_time) % 3600) // 60)
    time_seconds = int(((script_end_time - script_start_time) % 3600) % 60)
    print(f"Total execution time: {time_hours:02d}:{time_minutes:02d}:{time_seconds:02d}")
