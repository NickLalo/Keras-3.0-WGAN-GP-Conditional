"""
Script to hold functions to load data for the WGAN_GP model.
"""


import numpy as np
import keras
import tensorflow as tf


def load_mnist_data_for_gan(debug_run=False, dataset_subset_percentage=1.0, batch_size=16, verbose=True):
    """
    Load the MNIST dataset and return the training images and labels. A format that is tailored for training a GAN.
    
    Parameters:
        debug_run (bool): If True, run the model with a minimal subset of the training data.
        small_subset (bool): If True, run the model with a small subset of the training data.
        batch_size (int): The batch size for the training data.
        verbose (bool): If True, print out the shapes of the training data.
    
    Returns:
        train_dataset (tf.data.Dataset): A tf.data.Dataset object containing the training images and labels.
        img_shape (tuple): The shape of a single image sample
    """
    if debug_run and dataset_subset_percentage != 1.0:
        raise ValueError("Both debug_run and a dataset_subset_percentage cannot be set at the same time.")
    
    # Load the MNIST dataset
    (train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()
    
    # add the test data to the train data to increase the size of the dataset
    train_images = np.concatenate((train_images, test_images), axis=0)
    train_labels = np.concatenate((train_labels, test_labels), axis=0)
    
    if debug_run:
        print("\nDEBUG MODE: Running with a minimal subset of the training data.\n")
        # a number that is a power of 2 so it should work well with the batch size and be greater than the batch size
        subset_size = 1024
        train_images = train_images[:subset_size]
        train_labels = train_labels[:subset_size]
    elif dataset_subset_percentage < 1.0:
        print(f"\nDATASET SUBSET PERCENTAGE: Running with {dataset_subset_percentage}% of the training data.\n")
        train_images = train_images[:int(dataset_subset_percentage * len(train_images))]
        train_labels = train_labels[:int(dataset_subset_percentage * len(train_labels))]
    
    # Ensure that the batch size is not greater than the dataset size
    if batch_size > len(train_images):
        raise ValueError(f"Batch size ({batch_size}) cannot be greater than the dataset size ({len(train_images)}).")
    
    # Scale the images to the range [-1, 1]
    # Original pixel values are in the range [0, 255].
    # To scale them to [-1, 1], we subtract 127.5 (which centers the values around 0)
    # and then divide by 127.5 (which scales the values to be between -1 and 1).
    train_images = (train_images.astype(np.float32) - 127.5) / 127.5
    
    # MNIST images are (28, 28) (W x H) but our model needs a channel dimension (28, 28, 1) (W x H x C) so we add a singleton layer to the end
    train_images = np.expand_dims(train_images, axis=-1)
    
    # get the number of samples in the dataset (as a batched dataset makes it hard to get the exact number of samples)
    samples_per_epoch = len(train_images)
    
    # Create a tf.data.Dataset object for the training data
    buffer_size = min(len(train_images), 1024)
    train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))  # works seamlessly with numpy arrays
    train_dataset = train_dataset.shuffle(buffer_size).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    
    # determine the shape of a single image sample
    img_shape = train_dataset.element_spec[0].shape[1:]
    
    # get the number of classes in the dataset
    unique_labels = np.unique(train_labels)
    num_classes = len(unique_labels)
    
    if verbose:
        print(f"\nnumber of samples in the train_dataset: {samples_per_epoch}")
        print(f"Type of train_dataset: {type(train_dataset)}")
        print(f"Shape of train_dataset: {train_dataset.element_spec}")
        print(f"Shape of a single image: {img_shape}")
        print(f"Unique labels: {unique_labels}")
        print(f"Number of classes: {num_classes}\n")
    
    return train_dataset, img_shape, num_classes, samples_per_epoch


if __name__ == "__main__":
    train_dataset, img_shape, number_of_classes, samples_per_epoch = load_mnist_data_for_gan()
    
    number_of_batches = len(train_dataset)
    real_images, real_labels = next(iter(train_dataset))
    
    print(f"Number of batches in the train_dataset: {number_of_batches}")
    print(f"Shape of the real_images: {real_images.shape}")
    print(f"Shape of the real_labels: {real_labels.shape}")
