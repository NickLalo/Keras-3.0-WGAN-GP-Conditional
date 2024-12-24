"""
Script to hold functions to load data for the WGAN_GP model.
"""


from pathlib import Path
import random
import numpy as np
import matplotlib.pyplot as plt
import keras
import tensorflow as tf


# Set a random number for shuffling the dataset before the random seed so that the visualized training samples generated are different each run. It 
# is important to take a good look at the training samples to really understand the data the model is attempting to generate.
# NOTE: the random seed must be set after any thing is imported from this file
RANDOM_NUMBER_FOR_DATASET_SHUFFLE = random.randint(0, 1000)


def load_mnist_data_for_gan(debug_run=False, dataset_subset_percentage=1.0, batch_size=512, random_shift_frequency=0.5, verbose=True):
    """
    Load the MNIST dataset and return the training images and labels. A format that is tailored for training a GAN.
    
    Parameters:
        debug_run (bool): If True, run the model with a minimal subset of the training data.
        small_subset (bool): If True, run the model with a small subset of the training data.
        batch_size (int): The batch size for the training data.
        random_shift_frequency (float): The frequency of applying random shifts to the training images.
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
    elif dataset_subset_percentage > 1.0:
        raise ValueError(f"Dataset subset percentage ({dataset_subset_percentage}) cannot be greater than 1.0.")
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
    
    # Create a tf.data.Dataset object for the training data
    buffer_size = len(train_images)  # ensures the dataset is fully shuffled
    # use this line below if memory becomes an issue. This could help
    # buffer_size = min(len(train_images), 2048)  # ensure that the buffer size is not greater than the dataset size when running with a testing subset
    train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))  # works seamlessly with numpy arrays
    
    if random_shift_frequency > 0:
        # Define a function that does the random shift
        def add_random_shift(image, label):
            # Decide if we apply shift or not
            random_decision = tf.random.uniform([], 0, 1) < random_shift_frequency
            # Random direction: 0=Up, 1=Down, 2=Left, 3=Right
            random_direction = tf.random.uniform([], 0, 4, dtype=tf.int32)
            return map_data_shifter(image, label, random_decision, random_direction)
        
        # Apply the shifting to each element
        train_dataset = train_dataset.map(
            add_random_shift,
            num_parallel_calls=tf.data.AUTOTUNE
        )
    
    # Make an infinite dataset
    train_dataset = train_dataset.shuffle(buffer_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    # get some info about the dataset
    samples_per_epoch = len(train_images)
    img_shape = train_dataset.element_spec[0].shape[1:] # single image shape
    unique_labels = np.unique(train_labels)
    num_classes = len(unique_labels)
    
    if verbose:
        print(f"\n{'#'*60} DATASET LOADED SUCCESSFULLY {'#'*61}")
        print(f"Number of training samples: {len(train_images)}")
        print(f"range of values in the training dataset: [{np.min(train_images)}, {np.max(train_images)}]")
        print(f"range of values in the training labels: [{np.min(train_labels)}, {np.max(train_labels)}]")
        print(f"Type of train_dataset: {type(train_dataset)}")
        print(f"Shape of train_dataset: {train_dataset.element_spec}")
        print(f"Shape of a single image: {img_shape}")
        print(f"Unique labels: {unique_labels}")
        print(f"Number of classes: {num_classes}")
        print(f"random_shift_frequency to apply a 2px shift in a random direction: {random_shift_frequency}")
        print(f"{'#'*150}\n")
    
    return train_dataset, img_shape, num_classes, samples_per_epoch


def data_shifter(image, direction, apply_shift):
    """
    Shift the image in the specified direction if apply_shift is True by 2 pixel.
    
    Args:
        image (tf.Tensor): Input image tensor [height, width, channels].
        direction (int): 0 = Up, 1 = Down, 2 = Left, 3 = Right.
        apply_shift (bool): Whether to apply the shift.
    
    Returns:
        tf.Tensor: The shifted image.
    """
    if not apply_shift:
        return image
    
    shift_amount = 2  # pixels
    height, width = tf.shape(image)[0], tf.shape(image)[1]
    
    if direction == 0:  # Up
        # Pad bottom and crop from offset_height = shift_amount
        padded_image = tf.pad(
            image,
            paddings=[[0, shift_amount], [0, 0], [0, 0]],
            mode="CONSTANT",
            constant_values=-1,
        )
        offset_height, offset_width = shift_amount, 0
    
    elif direction == 1:  # Down
        # Pad top and crop from offset_height = 0
        padded_image = tf.pad(
            image,
            paddings=[[shift_amount, 0], [0, 0], [0, 0]],
            mode="CONSTANT",
            constant_values=-1,
        )
        offset_height, offset_width = 0, 0
    
    elif direction == 2:  # Left
        # Pad right and crop from offset_width = shift_amount
        padded_image = tf.pad(
            image,
            paddings=[[0, 0], [0, shift_amount], [0, 0]],
            mode="CONSTANT",
            constant_values=-1,
        )
        offset_height, offset_width = 0, shift_amount
    
    else:  # direction == 3 -> Right
        # Pad left and crop from offset_width = 0
        padded_image = tf.pad(
            image,
            paddings=[[0, 0], [shift_amount, 0], [0, 0]],
            mode="CONSTANT",
            constant_values=-1,
        )
        offset_height, offset_width = 0, 0
    
    shifted_image = tf.image.crop_to_bounding_box(
        padded_image,
        offset_height,
        offset_width,
        height,
        width
    )
    
    return shifted_image


def map_data_shifter(image, label, random_decision, direction):
    """
    Applies a random shift to the image if random_decision is True.
    
    Parameters:
        image (tf.Tensor): The input image tensor [height, width, channels].
        label (tf.Tensor): The label tensor.
        random_decision (tf.Tensor): A boolean tensor indicating whether to apply the shift.
        direction (tf.Tensor): The direction to shift the image.
    
    Returns:
        tf.Tensor: The shifted image.
    """
    shifted_image = data_shifter(image, direction, random_decision)
    return shifted_image, label


def test_shift_rate(frequency, num_samples=1000):
    """
    Test the shift logic and observe shift rates.
    
    Parameters:
        frequency (float): The frequency of applying random shifts.
        num_samples (int): The number of samples to test.
    
    Returns:
        None
    """
    dummy_image = tf.zeros((28, 28, 1), dtype=tf.float32)
    
    # Generate random decisions and directions for each image
    random_decisions = np.random.rand(num_samples) < frequency
    random_directions = np.random.randint(0, 4, size=num_samples)
    
    shifted_count = 0
    for i in range(num_samples):
        shifted_image, _ = map_data_shifter(dummy_image,
                                            tf.constant(0),
                                            random_decisions[i],
                                            random_directions[i])
        # Check if image was modified
        if not tf.reduce_all(tf.equal(shifted_image, dummy_image)):
            shifted_count += 1
    
    observed_rate = shifted_count / num_samples
    print(f"Frequency: {frequency}, Observed Shift Rate: {observed_rate:.2f}")
    return


def visualize_training_samples(train_dataset: tf.data.Dataset, model_training_output_dir: Path):
    """
    Saves a grid of training images provided by the train_dataset
    
    Parameters:
        None
    
    Returns:
        None
    """
    print(f"Visualizing training samples...")
    
    # number of rows and columns of training samples to visualize
    training_sample_grid = (10, 15)  
    
    # Retrieve 15 training images from each class.
    training_images_dict = {label: [] for label in range(10)}
    training_sample_collection_complete = False
    
    # get the GLOBAL variable RANDOM_NUMBER_FOR_DATASET_SHUFFLE to shuffle the dataset so that each time we run this code we get different samples
    global RANDOM_NUMBER_FOR_DATASET_SHUFFLE
    train_dataset = train_dataset.shuffle(buffer_size=1024, seed=RANDOM_NUMBER_FOR_DATASET_SHUFFLE)
    
    # Loop through the dataset to collect samples from each class
    while not training_sample_collection_complete:
        # Use .take(1) to extract one batch of data without consuming the entire dataset
        for real_images, real_labels in train_dataset.shuffle(buffer_size=1024).take(1):
            # Loop through the batch of images and labels and save the first 15 samples from each class
            for image, label in zip(real_images, real_labels):
                label = label.numpy()  # Convert label tensor to a numpy scalar
                if len(training_images_dict[label]) < 15:
                    training_images_dict[label].append(image)
        
        # Check if we have collected 15 samples from each class
        training_sample_collection_complete = all(len(images) == 15 for images in training_images_dict.values())
    
    # Stack the images for each class together
    training_images = []
    for label, images_list in training_images_dict.items():
        training_images.append(np.stack(images_list, axis=0))
    # convert to a numpy array
    training_images = np.stack(training_images, axis=0)
    
    # Plot the generated MNIST images
    fig, ax = plt.subplots(
        training_sample_grid[0], training_sample_grid[1],
        figsize=(training_sample_grid[1], training_sample_grid[0]),
        gridspec_kw={
            'wspace': -0.795,  # left and right
            'hspace': 0.05  # up and down
            }
    )
    
    # Add labels for each row and column
    for row in range(training_sample_grid[0]):
        for col in range(training_sample_grid[1]):
            # Plot the image
            ax[row, col].imshow(training_images[row, col, :, :, 0])
            ax[row, col].axis('off')
    
        # Add class label on the left
        ax[row, 0].annotate(f'{row}',
                            xy=(-0.42, 0.5), xycoords='axes fraction',
                            va='center', ha='center', fontsize=25, rotation=0)
    
    # Add column labels on the top
    for col in range(training_sample_grid[1]):
        ax[0, col].annotate(f'Training\nSample {col+1:>2}',
                            xy=(0.5, 1.05), xycoords='axes fraction',
                            va='bottom', ha='center', fontsize=9.5, rotation=0)
    
    # Add a title to the plot
    fig.text(x=0.0625, y=0.9625, s=f"Training Samples", ha='left', fontsize=24)
    
    # remove all axes from the plot
    plt.axis('off')
    # remove all box lines from the plot
    plt.box(False)
    
    # Adjust figure to remove extra padding and ensure spacing consistency
    plt.subplots_adjust(left=-0.015, right=1.113, top=0.90, bottom=0.01)
    
    # Save the grid of images to the current epoch checkpoint directory
    fig_save_path = model_training_output_dir.joinpath(f"training_samples.png")
    plt.savefig(fig_save_path, dpi=300, bbox_inches='tight')
    
    # Clear the current figure and close the plot to avoid memory leaks
    plt.clf()
    plt.close()
    
    print(f"Visualization of training samples saved to: {fig_save_path}")
    return


if __name__ == "__main__":
    # Load the MNIST dataset for training the WGAN_GP model
    train_dataset, img_shape, number_of_classes, samples_per_epoch = load_mnist_data_for_gan(random_shift_frequency=0.5)
    # generate a grid of training samples for visualization
    visualize_training_samples(train_dataset, Path.cwd())