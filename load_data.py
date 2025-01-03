"""
Script to hold functions to load data for the WGAN_GP model, now using KerasCV
for random rotation, random translation, and random zoom augmentations, and
always casting augmentation outputs to float32 to avoid tf.cond dtype mismatches.

This script implements data augmentation, but due to the small image size of MNIST it
causes the image to have overly smoothed edges which is detrimental for this example.
These options can be better explored in a larger image dataset.
"""

from pathlib import Path
import random
import numpy as np
import matplotlib.pyplot as plt
import keras
import tensorflow as tf
import keras_cv as kcv


################################################################################
# GLOBALS
################################################################################

RANDOM_NUMBER_FOR_DATASET_SHUFFLE = random.randint(0, 1000)

################################################################################
# KerasCV AUGMENTATION LAYERS
################################################################################

random_rotation_layer = kcv.layers.RandomRotation(
    factor=(5 / 180.0, 10 / 180.0),  # rotate between 5° to 10°
    fill_mode="constant",
    fill_value=-1.0,  # fill corners with -1.0 (images are in [-1,1])
)

random_translation_layer = kcv.layers.RandomTranslation(
    height_factor=0.1,  # up to 10% shift vertically
    width_factor=0.1,   # up to 10% shift horizontally
    fill_mode="constant",
    fill_value=-1.0,
)

random_zoom_layer = kcv.layers.RandomZoom(
    height_factor=(0.0, 0.2),  # up to 20% zoom in/out vertically
    width_factor=(0.0, 0.2),   # up to 20% zoom in/out horizontally
    fill_mode="constant",
    fill_value=-1.0,
)

################################################################################
# KERAS-CV AUGMENTATION HELPERS (Always cast final output to float32)
################################################################################

def apply_keras_cv_layer(image, layer):
    """
    Internal helper to apply a KerasCV layer to a single image (H, W, C).
    KerasCV expects (batch, H, W, C), and might produce float16 under a
    mixed_float16 policy. We'll cast the result back to float32.
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

def map_data_random_rotation(image, label, random_decision):
    """
    Apply random rotation (5-10 degrees) via KerasCV if random_decision is True.
    Both branches return float32.
    """
    def do_aug():
        aug_image = apply_keras_cv_layer(image, random_rotation_layer)
        return tf.cast(aug_image, tf.float32)

    def no_aug():
        # Even if the original image might be float16 or float32,
        # ensure the output is float32 here as well.
        return tf.cast(image, tf.float32)

    image_final = tf.cond(random_decision, do_aug, no_aug)
    return image_final, label

def map_data_random_translation(image, label, random_decision):
    """
    Apply random translation (~±10%) via KerasCV if random_decision is True.
    Both branches return float32.
    """
    def do_aug():
        aug_image = apply_keras_cv_layer(image, random_translation_layer)
        return tf.cast(aug_image, tf.float32)

    def no_aug():
        return tf.cast(image, tf.float32)

    image_final = tf.cond(random_decision, do_aug, no_aug)
    return image_final, label

def map_data_random_zoom(image, label, random_decision):
    """
    Apply random zoom (0-20%) via KerasCV if random_decision is True.
    Both branches return float32.
    """
    def do_aug():
        aug_image = apply_keras_cv_layer(image, random_zoom_layer)
        return tf.cast(aug_image, tf.float32)

    def no_aug():
        return tf.cast(image, tf.float32)

    image_final = tf.cond(random_decision, do_aug, no_aug)
    return image_final, label

################################################################################
# DATA LOADING FUNCTION
################################################################################

def load_mnist_data_for_gan(debug_run=False,
                            dataset_subset_percentage=1.0,
                            batch_size=512,
                            random_rotate_frequency=0.0,
                            random_translate_frequency=0.0,
                            random_zoom_frequency=0.0,
                            verbose=True):
    """
    Load the MNIST dataset and return the training images and labels in a format
    tailored for training a GAN. Augmentations (rotation, translation, zoom)
    can each be toggled via frequency arguments. Final dataset remains float32.
    """
    if debug_run and dataset_subset_percentage != 1.0:
        raise ValueError("Both debug_run and dataset_subset_percentage cannot be set.")

    # 1) Load MNIST data
    (train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()

    # 2) Combine train+test
    train_images = np.concatenate((train_images, test_images), axis=0)
    train_labels = np.concatenate((train_labels, test_labels), axis=0)

    # 3) Possibly reduce dataset size
    if debug_run:
        print("\nDEBUG MODE: Using minimal subset of training data.\n")
        subset_size = 1024
        train_images = train_images[:subset_size]
        train_labels = train_labels[:subset_size]
    elif dataset_subset_percentage > 1.0:
        raise ValueError(f"Dataset subset percentage ({dataset_subset_percentage}) > 1.0.")
    elif dataset_subset_percentage < 1.0:
        print(f"\nDATASET SUBSET PERCENTAGE: Using {dataset_subset_percentage * 100:.0f}% of data.\n")
        subset_size = int(dataset_subset_percentage * len(train_images))
        train_images = train_images[:subset_size]
        train_labels = train_labels[:subset_size]

    if batch_size > len(train_images):
        raise ValueError(f"Batch size ({batch_size}) > dataset size ({len(train_images)}).")

    # 4) Scale to [-1, 1], expand dims => float32
    train_images = (train_images.astype(np.float32) - 127.5) / 127.5
    train_images = np.expand_dims(train_images, axis=-1)

    # 5) Create Dataset
    buffer_size = 4096
    if buffer_size > len(train_images):
        buffer_size = len(train_images)
    train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))

    # 6) Random ROTATION
    if random_rotate_frequency > 0:
        def add_random_rotation(image, label):
            random_decision = tf.random.uniform([], 0, 1) < random_rotate_frequency
            return map_data_random_rotation(image, label, random_decision)
        train_dataset = train_dataset.map(add_random_rotation, num_parallel_calls=tf.data.AUTOTUNE)

    # 7) Random TRANSLATION
    if random_translate_frequency > 0:
        def add_random_translation(image, label):
            random_decision = tf.random.uniform([], 0, 1) < random_translate_frequency
            return map_data_random_translation(image, label, random_decision)
        train_dataset = train_dataset.map(add_random_translation, num_parallel_calls=tf.data.AUTOTUNE)

    # 8) Random ZOOM
    if random_zoom_frequency > 0:
        def add_random_zoom(image, label):
            random_decision = tf.random.uniform([], 0, 1) < random_zoom_frequency
            return map_data_random_zoom(image, label, random_decision)
        train_dataset = train_dataset.map(add_random_zoom, num_parallel_calls=tf.data.AUTOTUNE)

    # 9) Shuffle, batch, prefetch, cache
    train_dataset = (
        train_dataset
        .shuffle(buffer_size)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
        .cache()  # can only be used if the dataset fits in memory
    )

    samples_per_epoch = len(train_images)
    img_shape = train_dataset.element_spec[0].shape[1:]
    unique_labels = np.unique(train_labels)
    num_classes = len(unique_labels)

    if verbose:
        print(f"\n{'#'*60} DATASET LOADED SUCCESSFULLY {'#'*61}")
        print(f"Number of training samples: {samples_per_epoch}")
        print(f"Value range in training dataset: [{np.min(train_images)}, {np.max(train_images)}]")
        print(f"Labels range: [{np.min(train_labels)}, {np.max(train_labels)}]")
        print(f"Shape of a single image: {img_shape}")
        print(f"Unique labels: {unique_labels}")
        print(f"Number of classes: {num_classes}")
        print(f"random_rotate_frequency: {random_rotate_frequency}")
        print(f"random_translate_frequency: {random_translate_frequency}")
        print(f"random_zoom_frequency: {random_zoom_frequency}")
        print(f"{'#'*150}\n")

    return train_dataset, img_shape, num_classes, samples_per_epoch

################################################################################
# VISUALIZATION FUNCTION
################################################################################

def visualize_training_samples(train_dataset: tf.data.Dataset, model_training_output_dir: Path):
    """
    Saves a grid of training images provided by the train_dataset
    """
    print(f"Visualizing training samples...")

    training_sample_grid = (10, 15)
    training_images_dict = {label: [] for label in range(10)}
    training_sample_collection_complete = False

    global RANDOM_NUMBER_FOR_DATASET_SHUFFLE
    train_dataset = train_dataset.shuffle(buffer_size=32, seed=RANDOM_NUMBER_FOR_DATASET_SHUFFLE)

    while not training_sample_collection_complete:
        for real_images, real_labels in train_dataset.take(1):
            for image, label in zip(real_images, real_labels):
                label = label.numpy()
                if len(training_images_dict[label]) < 15:
                    training_images_dict[label].append(image)
        training_sample_collection_complete = all(len(lst) == 15 for lst in training_images_dict.values())

    training_images = []
    for label, images_list in training_images_dict.items():
        training_images.append(np.stack(images_list, axis=0))
    training_images = np.stack(training_images, axis=0)  # shape: (10, 15, 28, 28, 1)

    fig, ax = plt.subplots(
        training_sample_grid[0],
        training_sample_grid[1],
        figsize=(training_sample_grid[1], training_sample_grid[0]),
        gridspec_kw={'wspace': -0.795, 'hspace': 0.05}
    )

    for row in range(training_sample_grid[0]):
        for col in range(training_sample_grid[1]):
            ax[row, col].imshow(training_images[row, col, :, :, 0])
            ax[row, col].axis('off')
        ax[row, 0].annotate(
            f'{row}',
            xy=(-0.42, 0.5), xycoords='axes fraction',
            va='center', ha='center', fontsize=25
        )

    for col in range(training_sample_grid[1]):
        ax[0, col].annotate(
            f'Training\nSample {col+1:>2}',
            xy=(0.5, 1.05), xycoords='axes fraction',
            va='bottom', ha='center', fontsize=9.5
        )

    fig.text(x=0.0625, y=0.9625, s="Training Samples", ha='left', fontsize=24)
    plt.axis('off')
    plt.box(False)
    plt.subplots_adjust(left=-0.015, right=1.113, top=0.90, bottom=0.01)

    fig_save_path = model_training_output_dir.joinpath("training_samples.png")
    plt.savefig(fig_save_path, dpi=150, bbox_inches='tight')
    plt.clf()
    plt.close()
    print(f"Visualization of training samples saved to: {fig_save_path}")

################################################################################
# MAIN
################################################################################

if __name__ == "__main__":
    train_dataset, img_shape, num_classes, samples_per_epoch = load_mnist_data_for_gan(
        random_rotate_frequency=0.0,
        random_translate_frequency=0.0,
        random_zoom_frequency=0.0
    )
    visualize_training_samples(train_dataset, Path.cwd())
