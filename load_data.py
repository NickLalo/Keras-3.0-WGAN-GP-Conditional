"""
Script to hold functions to load data for the WGAN_GP model with KerasCV for random rotation, random translation, and random zoom augmentations. This
code can be run in standalone mode to visualize a random selection of training samples with a higher frequency of random augmentations than is likely
to be used in practice.

NOTE: Due to the small image size of MNIST, these augmentations cause modified images to look significantly different from the original images.
Because of this limitation, these augmentations shouldn't be used for MNIST, but can be explored when working with larger image datasets.

One workaround worth exploring in the future to utilize the KerasCV augmentations is to adopt the method used in the Training Generative Adversarial 
Networks with Limited Data paper by Karras et. al. (https://arxiv.org/abs/2006.06676). The method used excels at training a GAN with a small dataset
and an adaptive augmentation strategy that prevents the critic from overfitting to the training data.
"""

from pathlib import Path
import random
import numpy as np
import matplotlib.pyplot as plt
import keras
import tensorflow as tf
import keras_cv as kcv


# Set a random number for shuffling the dataset before the global random seed for reproducibility so that the visualized training samples are 
# different each run. It is important to take a good look at the training samples to really understand the data the model is attempting to generate.
# NOTE: the global random seed for reproducibility must be set after anything is imported from this file.
RANDOM_NUMBER_FOR_DATASET_SHUFFLE = random.randint(0, 1000)


class DataAugmentor:
    """
    A class to encapsulate KerasCV augmentation layers and provide methods to apply them conditionally.
    NOTE: uses hardcoded values for augmentation parameters for simplicity, but these could be modified to be passed as arguments.
    """
    def __init__(self):
        """
        Initialize the augmentation layers.
        """
        # decided fill mode and value for all augmentations
        self.fill_mode = "constant"
        self.fill_value = -1.0  # fill values with -1.0 (images are in [-1,1]). Useful when there is a mostly black background in the images.
        # NOTE: choosing fill_mode of constant and fill_value of 1.0 can be a good way to very obviously see the applied augmentations when running
        # this script in standalone mode.
        
        ########################################################### Rotation Augmentation ############################################################
        # decide the range of rotation in degrees. For example, if a max_clockwise_rotation of 15 degrees and a max_counter_clockwise_rotation of 15
        # degrees is chosen, the image is randomly rotated some amount up to either 15 degrees clockwise or 15 degrees counterclockwise. Specifying
        # these values separately can be helpful if rotating in one direction is more likely than the other.
        self.max_clockwise_rotation = 15 # degrees
        self.max_counter_clockwise_rotation = 15 # degrees
        
        # Convert the rotation values to fractions of 2π radians for the KerasCV layer
        # Degrees → Radians → Fraction of 2π
        self.max_clockwise_rotation = -1 * self.max_clockwise_rotation / 360.0  # Negative for clockwise
        self.max_counter_clockwise_rotation = self.max_counter_clockwise_rotation / 360.0  # Positive for counterclockwise
        
        # instantiate the KerasCV layer for random rotation
        self.random_rotation_layer = kcv.layers.RandomRotation(
            factor=(self.max_clockwise_rotation, self.max_counter_clockwise_rotation),
            fill_mode=self.fill_mode,
            fill_value=self.fill_value,
        )
        
        ########################################################## Translation Augmentation ##########################################################
        # decide the range of translation factors in both directions as a percentage of the image size. For example, if 10% for vertical and 10% for
        # horizontal is chosen, the image is randomly shifted some amount up to 10% of the image height and also up to 10% of the image width. The
        # randomly chosen values are not the same for both directions resulting in non-uniform shifts.
        self.vertical_shift_factor = 0.1  # percentage
        self.horizontal_shift_factor = 0.1   # percentages
        
        # instantiate the KerasCV layer for random translation
        self.random_translation_layer = kcv.layers.RandomTranslation(
            height_factor=self.vertical_shift_factor,
            width_factor=self.horizontal_shift_factor,
            fill_mode=self.fill_mode,
            fill_value=self.fill_value,
        )
        
        ############################################################# Zoom Augmentation ##############################################################
        # decide the range of zoom in/out factors in both directions as a percentage of the image size. For example, if 20% for vertical and 20% for
        # horizontal is chosen, the image is randomly zoomed in or out some amount up to 20% of the image height and also up to 20% of the image width.
        # The randomly chosen values are not the same for both directions resulting in non-uniform zooms which can make some images look stretched if 
        # the zoom percentage is too high.
        self.vertical_in_out_zoom = 0.1  # percentage
        self.horizontal_in_out_zoom = 0.1  # percentage
        
        self.random_zoom_layer = kcv.layers.RandomZoom(
            height_factor=self.vertical_in_out_zoom,
            width_factor=self.horizontal_in_out_zoom,
            fill_mode=self.fill_mode,
            fill_value=self.fill_value,
        )
        return
    
    def apply_keras_cv_layer(self, image, layer):
        """
        Internal helper to apply a KerasCV layer to a single image (H, W, C).
        KerasCV expects (batch, H, W, C), and might produce float16 under a
        mixed_float16 policy. The result back to float32 to avoid errors.
        
        Parameters: 
            image: tf.Tensor, shape (H, W, C)
            layer: KerasCV layer object
        
        Returns:
            Augmented image as tf.Tensor, shape (H, W, C), float32
        """
        # Cast input to float32 for augmentation
        image_32 = tf.cast(image, tf.float32)
        
        # Expand dims to (1, H, W, C)
        image_batched = tf.expand_dims(image_32, axis=0)
        
        # Apply augmentation
        augmented_batched = layer(image_batched, training=True)
        
        # Squeeze back to (H, W, C)
        augmented = tf.squeeze(augmented_batched, axis=0)
        
        # Force final output to float32
        return tf.cast(augmented, tf.float32)
    
    def map_data_random_rotation(self, image, label, random_decision):
        """
        Apply random rotation via KerasCV if random_decision is True.
        
        Parameters:
            image: tf.Tensor, shape (H, W, C)
            label: tf.Tensor, shape ()
            random_decision: tf.Tensor, shape ()
        
        Returns:
            Augmented image as tf.Tensor, shape (H, W, C), float32
            label: tf.Tensor, shape ()
        """
        def do_aug():
            aug_image = self.apply_keras_cv_layer(image, self.random_rotation_layer)
            return tf.cast(aug_image, tf.float32)
        
        def no_aug():
            return tf.cast(image, tf.float32)
        
        image_final = tf.cond(random_decision, do_aug, no_aug)
        return image_final, label
    
    def map_data_random_translation(self, image, label, random_decision):
        """
        Apply random translation via KerasCV if random_decision is True.
        
        Parameters:
            image: tf.Tensor, shape (H, W, C)
            label: tf.Tensor, shape ()
            random_decision: tf.Tensor, shape ()
        
        Returns:
            Augmented image as tf.Tensor, shape (H, W, C), float32
            label: tf.Tensor, shape ()
        """
        def do_aug():
            aug_image = self.apply_keras_cv_layer(image, self.random_translation_layer)
            return tf.cast(aug_image, tf.float32)
        
        def no_aug():
            return tf.cast(image, tf.float32)
        
        image_final = tf.cond(random_decision, do_aug, no_aug)
        return image_final, label
    
    def map_data_random_zoom(self, image, label, random_decision):
        """
        Apply random zoom via KerasCV if random_decision is True.
        
        Parameters:
            image: tf.Tensor, shape (H, W, C)
            label: tf.Tensor, shape ()
            random_decision: tf.Tensor, shape ()
        
        Returns:
            Augmented image as tf.Tensor, shape (H, W, C), float32
            label: tf.Tensor, shape ()
        """
        def do_aug():
            aug_image = self.apply_keras_cv_layer(image, self.random_zoom_layer)
            return tf.cast(aug_image, tf.float32)
        
        def no_aug():
            return tf.cast(image, tf.float32)
        
        image_final = tf.cond(random_decision, do_aug, no_aug)
        return image_final, label


def load_mnist_data_for_gan(debug_run: bool=False,
                            dataset_subset_percentage: float=1.0,
                            batch_size: int=512,
                            random_rotate_frequency: float=0.0,
                            random_translate_frequency: float=0.0,
                            random_zoom_frequency: float=0.0,
                            verbose: bool=True):
    """
    Load the MNIST dataset and return the training images and labels in a format tailored for training a GAN. Augmentations (rotation, translation, 
    zoom) can each be toggled via frequency arguments. However, due to the small image size of MNIST, these augmentations cause modified images to
    look significantly different from the original images. Due to this limitation, these augmentations shouldn't be used for MNIST, but can be explored
    when working with larger image datasets.
    
    Parameters:
        augmentor: DataAugmentor, instance containing augmentation layers and methods
        debug_run: bool, whether to run in debug mode with minimal data
        dataset_subset_percentage: float, percentage of the dataset to use (0.0 to 1.0)
        batch_size: int, batch size for training
        random_rotate_frequency: float, frequency of random rotation augmentations
        random_translate_frequency: float, frequency of random translation augmentations
        random_zoom_frequency: float, frequency of random zoom augmentations
        verbose: bool, whether to print dataset information
    
    Returns:
        train_dataset: tf.data.Dataset, training dataset for the GAN
        img_shape: tuple, shape of a single image
        num_classes: int, number of classes in the dataset
        samples_per_epoch: int, number of samples in the training
    """
    if debug_run and dataset_subset_percentage != 1.0:
        raise ValueError("Both debug_run and dataset_subset_percentage cannot be set.")
    
    # 1) Load MNIST data
    (train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()
    
    # 2) Combine train+test
    train_images = np.concatenate((train_images, test_images), axis=0)
    train_labels = np.concatenate((train_labels, test_labels), axis=0)
    
    # 3) Reduce dataset size when either debug_mode or dataset_subset_percentage is set
    if debug_run:
        if verbose:
            print("\nDEBUG MODE: Using minimal subset of training data.\n")
        subset_size = 1024
        train_images = train_images[:subset_size]
        train_labels = train_labels[:subset_size]
    elif dataset_subset_percentage > 1.0:
        raise ValueError(f"Dataset subset percentage ({dataset_subset_percentage}) > 1.0.")
    elif dataset_subset_percentage < 1.0:
        if verbose:
            print(f"\nDATASET SUBSET PERCENTAGE: Using {dataset_subset_percentage * 100:.0f}% of data.\n")
        subset_size = int(dataset_subset_percentage * len(train_images))
        train_images = train_images[:subset_size]
        train_labels = train_labels[:subset_size]
    
    if batch_size > len(train_images):
        raise ValueError(f"Batch size ({batch_size}) > dataset size ({len(train_images)}).")
    
    # 4) Scale to [-1, 1] and expand dims by adding channel dimension
    train_images = (train_images.astype(np.float32) - 127.5) / 127.5
    train_images = np.expand_dims(train_images, axis=-1)
    
    # 5) Create Dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    
    # 6) Add Augmentations to train_dataset for random augmentations during loading in the training loop
    augmentor = DataAugmentor()  # instantiate the DataAugmentor class
    if random_rotate_frequency > 0:
        def add_random_rotation(image, label):
            random_decision = tf.random.uniform([], 0, 1) < random_rotate_frequency
            return augmentor.map_data_random_rotation(image, label, random_decision)
        train_dataset = train_dataset.map(add_random_rotation, num_parallel_calls=tf.data.AUTOTUNE)
    
    if random_translate_frequency > 0:
        def add_random_translation(image, label):
            random_decision = tf.random.uniform([], 0, 1) < random_translate_frequency
            return augmentor.map_data_random_translation(image, label, random_decision)
        train_dataset = train_dataset.map(add_random_translation, num_parallel_calls=tf.data.AUTOTUNE)
    
    if random_zoom_frequency > 0:
        def add_random_zoom(image, label):
            random_decision = tf.random.uniform([], 0, 1) < random_zoom_frequency
            return augmentor.map_data_random_zoom(image, label, random_decision)
        train_dataset = train_dataset.map(add_random_zoom, num_parallel_calls=tf.data.AUTOTUNE)
    
    # 7) Shuffle, batch, prefetch, cache
    # NOTE: the buffer_size can be set to the length of the dataset, but this can cause significant overhead for visualizing the training samples
    # as well as training the model at the start of each epoch. A smaller buffer size was chosen to reduce this time which results in a non-perfectly
    # shuffled dataset that is still good enough for training.
    buffer_size = 4096
    if buffer_size > len(train_images):
        buffer_size = len(train_images)
    train_dataset = (
        train_dataset
        .shuffle(buffer_size)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
        .cache()  # can only be used if the dataset fits in memory
    )
    
    # 8) Get dataset information and print to console
    samples_per_epoch = len(train_images)
    img_shape = train_dataset.element_spec[0].shape[1:]
    img_dtype = train_dataset.element_spec[0].dtype
    label_dtype = train_dataset.element_spec[1].dtype
    unique_labels = np.unique(train_labels)
    num_classes = len(unique_labels)
    # get the probability that at least one of the augmentations will be applied to a single image
    random_augmentation_probability = (1 - (1 - random_rotate_frequency) * (1 - random_translate_frequency) * (1 - random_zoom_frequency)) * 100
    # get the probability that all of the augmentations will be applied to a single image
    random_augmentation_probability_all = (random_rotate_frequency * random_translate_frequency * random_zoom_frequency * 100)
    
    if verbose:
        print(f"\n{'#'*60} DATASET LOADED SUCCESSFULLY {'#'*61}")
        print(f"Number of training samples: {samples_per_epoch}")
        print(f"Value range in training dataset: [{np.min(train_images)}, {np.max(train_images)}]")
        print(f"Labels range: [{np.min(train_labels)}, {np.max(train_labels)}]")
        print(f"Shape of a single image: {img_shape}")
        print(f"Image dtype: {img_dtype}")
        print(f"Label dtype: {label_dtype}")
        print(f"Unique labels: {unique_labels}")
        print(f"Number of classes: {num_classes}")
        print(f"Batch size: {batch_size}")
        print(f"random_rotate_frequency: {random_rotate_frequency}")
        print(f"random_translate_frequency: {random_translate_frequency}")
        print(f"random_zoom_frequency: {random_zoom_frequency}")
        print(f"probability of at least one augmentation per sample: {random_augmentation_probability:.4f}")
        print(f"probability of all augmentations per sample: {random_augmentation_probability_all:.4f}")
        print(f"{'#'*150}\n")
    
    return train_dataset, img_shape, num_classes, samples_per_epoch


########################################################## Visualize Training Data Function ##########################################################
def visualize_training_samples(train_dataset: tf.data.Dataset, model_training_output_dir: Path):
    """
    Saves a grid of training images provided by the train_dataset
    
    Parameters:
        train_dataset: tf.data.Dataset, training dataset for the GAN
        model_training_output_dir: Path, directory to save the visualization
    
    Returns:
        None
    """
    print(f"Visualizing training samples...")
    
    # number of rows (classes) and columns (samples) of training samples to visualize
    training_sample_grid = (10, 15)
    
    # collect 15 samples for each class
    training_images_dict = {label: [] for label in range(training_sample_grid[0])}
    training_sample_collection_complete = False
    
    # get the GLOBAL RANDOM_NUMBER_FOR_DATASET_SHUFFLE which is a true random number for extracting different training samples each run
    global RANDOM_NUMBER_FOR_DATASET_SHUFFLE
    train_dataset = train_dataset.shuffle(buffer_size=32, seed=RANDOM_NUMBER_FOR_DATASET_SHUFFLE)
    
    # loop through the dataset to collect 15 samples for each class
    while not training_sample_collection_complete:
        # Use .take(1) to extract one batch of data without consuming the entire dataset
        for real_images, real_labels in train_dataset.take(1):
            # Loop through the batch of images and labels and save the first 15 samples from each class
            for image, label in zip(real_images, real_labels):
                label = label.numpy()
                if len(training_images_dict[label]) < 15:
                    training_images_dict[label].append(image)
        
        # Check if we have collected 15 samples from each class and are ready to visualize
        training_sample_collection_complete = all(len(lst) == 15 for lst in training_images_dict.values())
    
    # Stack the images for each class together
    training_images = []
    for label, images_list in training_images_dict.items():
        training_images.append(np.stack(images_list, axis=0))
    # convert list of arrays to numpy array
    training_images = np.stack(training_images, axis=0)  # shape: (10, 15, 28, 28, 1)
    
    # Plot the training samples
    fig, ax = plt.subplots(
        training_sample_grid[0],
        training_sample_grid[1],
        figsize=(training_sample_grid[1], training_sample_grid[0]),
        gridspec_kw={
            'wspace': -0.795,  # left and right spacing between subplots
            'hspace': 0.05  # top and bottom spacing between subplots
            }
    )
    
    # Loop through the training samples and plot them
    for row in range(training_sample_grid[0]):
        for col in range(training_sample_grid[1]):
            # Plot the image
            ax[row, col].imshow(training_images[row, col, :, :, 0])
            ax[row, col].axis('off')
        # Annotate the row number
        ax[row, 0].annotate(
            f'{row}',
            xy=(-0.42, 0.5), xycoords='axes fraction',
            va='center', ha='center', fontsize=25
        )
    
    # Annotate the columns with the training sample number 
    for col in range(training_sample_grid[1]):
        ax[0, col].annotate(
            f'Training\nSample {col+1:>2}',
            xy=(0.5, 1.05), xycoords='axes fraction',
            va='bottom', ha='center', fontsize=9.5
        )
    
    # add title and save the figure
    fig.text(x=0.0625, y=0.9625, s="Training Samples", ha='left', fontsize=24)
    plt.axis('off')
    plt.box(False)
    plt.subplots_adjust(left=-0.015, right=1.113, top=0.90, bottom=0.01)
    
    fig_save_path = model_training_output_dir.joinpath("training_samples.png")
    plt.savefig(fig_save_path, dpi=150, bbox_inches='tight')
    plt.clf()
    plt.close()
    print(f"Visualization of training samples saved to: {fig_save_path}")
    return


if __name__ == "__main__":
    # test to visualize a random selection of training samples with a higher frequency of random augmentations than is likely to be used in practice
    # NOTE: for MNIST, these augmentations cause modified images to look significantly different from the original images. Because of this limitation,
    # these augmentations shouldn't be used for MNIST, but can be explored when working with larger image datasets.
    
    # load the training dataset
    train_dataset, img_shape, num_classes, samples_per_epoch = load_mnist_data_for_gan(
        debug_run=False,
        dataset_subset_percentage=1.0,
        batch_size=512,
        random_rotate_frequency=0.5,
        random_translate_frequency=0.5,
        random_zoom_frequency=0.5,
        verbose=True
    )
    
    # Visualize training samples
    visualize_training_samples(train_dataset, Path.cwd())
