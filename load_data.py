"""
Script to hold functions to load data for the WGAN_GP model.
"""


import numpy as np
import keras


def load_mnist_dataset(debug_run=False):
    """
    Load the MNIST dataset and return the training images and labels. A format that is tailored for training a GAN.
    """
    # Load the MNIST dataset
    (train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()
    # determine the shape of a single image sample
    img_shape = train_images.shape[1:]
    
    # if the img shape is 2 dimensional, add a singleton layer to the end
    # MNIST image shape should be: (28, 28, 1)
    if len(img_shape) == 2:
        img_shape = img_shape + (1,)
    
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
    
    if debug_run:
        print("DEBUG MODE: Running with a minimal subset of the training data.")
        percentage = 0.05
        train_images = train_images[:int(percentage * len(train_images))]
        train_labels = train_labels[:int(percentage * len(train_labels))]
    
    # Print the shapes to verify that the data was loaded correctly
    print("Shape of train_images:", train_images.shape)
    print("Shape of train_labels:", train_labels.shape)
    
    # Check that all labels are the same
    unique_labels = np.unique(train_labels)
    print("Unique labels in the dataset:", unique_labels)
    
    return train_images, train_labels, img_shape
