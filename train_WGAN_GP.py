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

from model_and_callbacks import get_discriminator_model, get_generator_model, GANMonitor, LossLogger, WGAN_GP


if __name__ == "__main__":
    script_start_time = time.time()
    #-------------------------------
    ### HYPERPARAMETERS
    #-------------------------------
    batch_size = 256
    # Size of the noise vector
    noise_dim = 128
    # Set the number of epochs for training.
    epochs = 5  # REVIEW: for now, train for a very small number of epochs
    
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
    # d_model.summary()
    
    gen_model = get_generator_model(noise_dim)
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
    
    # Initialize the customer `GANMonitor` Keras callback to view generated images at the end of every epoch
    gan_monitor_callback = GANMonitor(num_img=20, latent_dim=noise_dim, grid_size=(4, 5))
    # Initialize a custom LossLogger callback to log metrics at the end of every epoch
    loss_logger_callback = LossLogger(samples_per_epoch=train_images.shape[0])
    
    # Get the wgan_gp model
    wgan_gp = WGAN_GP(
        discriminator=disc_model,
        generator=gen_model,
        latent_dim=noise_dim,
        discriminator_extra_steps=5,  # was set to 3, but I think 5 is recommended
        )
    
    # Compile the wgan_gp model
    wgan_gp.compile(
        disc_optimizer=discriminator_optimizer,
        gen_optimizer=generator_optimizer,
        disc_loss_fn=discriminator_loss,
        gen_loss_fn=generator_loss,
        )
    
    # REVIEW: model testing
    # save the model using custom save method
    wgan_gp.save_model("wgan_gp_model")
    
    # TODO: I have to compile the model after loading it in
    # TODO: do I need to save the optimizer state as well?
    # load the model using custom load method
    loaded_wgan_gp = WGAN_GP.load_model("wgan_gp_model")
    
    # REVIEW: if I am reinitializing the model, would rebuilding the optimizers cause the checkpoint to train differently and not pick back up correctly?
    #   The optimizer state should be saved and loaded as well.
    #   OR I should investigate if I can reload the model in a way that doesn't require recompiling the model.
    loaded_wgan_gp.compile(
        disc_optimizer=discriminator_optimizer,
        gen_optimizer=generator_optimizer,
        disc_loss_fn=discriminator_loss,
        gen_loss_fn=generator_loss,
        )
    
    random_latent_vectors = tf.random.normal(shape=(20, noise_dim))
    assert np.allclose(wgan_gp.generator(random_latent_vectors, training=False), loaded_wgan_gp.generator(random_latent_vectors, training=False))
    
    # REVIEW: remove this code after testing model save is complete
    print("initial assertion for saving/load new model passed!\n")
    
    # Start training the loaded model to see if we run into any errors with reloading the model
    wgan_gp.fit(
        train_images, 
        batch_size=batch_size,
        epochs=epochs, 
        callbacks=[
            gan_monitor_callback,
            loss_logger_callback]
        )
    
    # # Start training
    # wgan_gp.fit(
    #     train_images, 
    #     batch_size=batch_size,
    #     epochs=epochs, 
    #     callbacks=[
    #         gan_monitor_callback,
    #         loss_logger_callback]
    #     )
    
    script_end_time = time.time()
    time_hours = int((script_end_time - script_start_time) // 3600)
    time_minutes = int(((script_end_time - script_start_time) % 3600) // 60)
    time_seconds = int(((script_end_time - script_start_time) % 3600) % 60)
    print(f"Total execution time: {time_hours:02d}:{time_minutes:02d}:{time_seconds:02d}")
