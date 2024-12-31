"""
critic and generator model definitions and callbacks
"""


import os
from pathlib import Path
import keras
from keras import layers
import tensorflow as tf


# Clear all previously registered custom objects. Necessary addition for loading models with custom objects.
keras.saving.get_custom_objects().clear()


@keras.saving.register_keras_serializable(package="critic_model", name="critic_model")
def get_critic_model(img_shape: tuple, num_classes: int, model_training_output_dir: Path):
    """
    initialize the critic using the Keras Functional API. The critic model takes as input an image tensor and a label tensor and outputs a single
    value. The label tensor is converted to a one-hot tensor, passed through a fully connected layer to reshape it to the same shape as the image
    and then concatenated with the image tensor. This combined tensor is passed through the convolutional layers and lastly through a dense layer
    without an activation function to output the critic's score for the input image-label pair.
    
    I removed dropout from the critic model, but could add it back if the model is overfitting.
    
    This function also:
        1. saves a copy of the model summary to a text file in the model_training_output directory
        2. visualizes the model architecture with keras.utils.plot_model
        3. saves a copy of the model for visualization with netron.app
    
    Parameters:
        img_shape: tuple, the shape of the input image tensor (height, width, channels)
        num_classes: int, the number of classes in the dataset
        model_training_output_dir: Path, the directory to save outputs from the model training
    Returns:
        critic_model: keras.Model, the compiled critic model
    """
    # -----------------------------
    #   Inputs
    # -----------------------------
    img_input = layers.Input(shape=img_shape, name="image_input")
    class_input = layers.Input(shape=(1,), dtype=tf.int32, name="class_input")

    # -----------------------------
    #   Label Embedding & Map
    # -----------------------------
    # 1) Convert label to an embedding
    label_embed = layers.Embedding(input_dim=num_classes, output_dim=10)(class_input)
    label_embed = layers.Flatten()(label_embed)
    
    # 2) Create a 28×28 "label map"
    label_map = layers.Dense(units=28 * 28, use_bias=True)(label_embed)
    label_map = layers.Reshape((28, 28, 1))(label_map)
    
    # 3) Concatenate label_map with the image → (28,28,2)
    x = layers.Concatenate()([img_input, label_map])
    
    # -----------------------------
    #   (28×28) Conv + Residual
    # -----------------------------
    # Initial conv (stride=1) to blend channels
    x = layers.Conv2D(filters=64, kernel_size=3, strides=1, padding="same")(x)
    x = layers.LeakyReLU(alpha=0.2)(x)

    # Residual block at 28×28
    shortcut = x
    x = layers.Conv2D(filters=64, kernel_size=3, strides=1, padding="same")(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Conv2D(filters=64, kernel_size=3, strides=1, padding="same")(x)
    x = layers.Add()([shortcut, x])
    x = layers.LeakyReLU(alpha=0.2)(x)

    # -----------------------------
    #   Downsample (28×28 → 14×14)
    # -----------------------------
    x = layers.Conv2D(filters=128, kernel_size=4, strides=2, padding="same")(x)
    x = layers.LeakyReLU(alpha=0.2)(x)

    # Residual block at 14×14
    shortcut = x
    x = layers.Conv2D(filters=128, kernel_size=3, strides=1, padding="same")(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Conv2D(filters=128, kernel_size=3, strides=1, padding="same")(x)
    x = layers.Add()([shortcut, x])
    x = layers.LeakyReLU(alpha=0.2)(x)

    # -----------------------------
    #   Downsample (14×14 → 7×7)
    # -----------------------------
    x = layers.Conv2D(filters=128, kernel_size=4, strides=2, padding="same")(x)
    x = layers.LeakyReLU(alpha=0.2)(x)

    # Residual block at 7×7
    shortcut = x
    x = layers.Conv2D(filters=128, kernel_size=3, strides=1, padding="same")(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Conv2D(filters=128, kernel_size=3, strides=1, padding="same")(x)
    x = layers.Add()([shortcut, x])
    x = layers.LeakyReLU(alpha=0.2)(x)

    # -----------------------------
    #   Downsample (7×7 → 4×4)
    # -----------------------------
    x = layers.Conv2D(filters=256, kernel_size=4, strides=2, padding="same")(x)
    x = layers.LeakyReLU(alpha=0.2)(x)

    # Residual block at 4×4
    shortcut = x
    x = layers.Conv2D(filters=256, kernel_size=3, strides=1, padding="same")(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Conv2D(filters=256, kernel_size=3, strides=1, padding="same")(x)
    x = layers.Add()([shortcut, x])
    x = layers.LeakyReLU(alpha=0.2)(x)

    # -----------------------------
    #   Final Classification
    # -----------------------------
    x = layers.Flatten()(x)
    x = layers.Dense(1, use_bias=True)(x)

    # Build the critic model
    critic_model = tf.keras.models.Model(
        [img_input, class_input],
        x,
        name="critic"
    )
    
    ######################################################## Model Summary and Visualization ########################################################
    # get the filepath to the model_architecture_and_summary directory
    model_architecture_and_summary_dir = model_training_output_dir.joinpath("model_architecture_and_summary")
    # create the directory if it doesn't exist
    os.makedirs(model_architecture_and_summary_dir, exist_ok=True)
    
    # save a copy of the model summary to a text file
    model_summary_path = model_architecture_and_summary_dir.joinpath("critic_model_summary.txt")
    with open(model_summary_path, "w") as file:
        critic_model.summary(line_length=150, print_fn=lambda x: file.write(x + "\n"))
    
    # visualize the model architecture with keras.utils.plot_model
    model_visualization_path = model_architecture_and_summary_dir.joinpath("critic_model_architecture.png")
    keras.utils.plot_model(critic_model, to_file=model_visualization_path, show_shapes=True, show_layer_names=True)
    
    # save a copy of the model for visualization with netron.app
    model_netron_path = model_architecture_and_summary_dir.joinpath("critic_model_for_netron_app.keras")
    critic_model.save(model_netron_path)
    
    print(f"\nDone initializing the critic model.")
    print(f"Model summary saved to: {model_summary_path}")
    print(f"Visualization of architecture saved to: {model_visualization_path}")
    print(f"Model saved for visualization with netron.app to: {model_netron_path}\n")
    return critic_model


@keras.saving.register_keras_serializable(package="generator_model", name="generator_model")
def get_generator_model(noise_dim: int, num_classes: int, model_training_output_dir: Path):
    """
    Builds and returns a generator model for a Wasserstein GAN with Gradient Penalty (WGAN-GP).
    Parameters:
        noise_dim (int): Dimension of the noise vector input.
        num_classes (int): Number of classes for the class conditional input.
        model_training_output_dir (Path): Directory path for saving model training outputs.
    Returns:
        keras.models.Model: A Keras Model representing the generator.
    """
    ############################################################### Input Preparation ###############################################################
    # arguments
    embed_dim = 10
    # -----------------------------
    #   Inputs
    # -----------------------------
    noise_input = layers.Input(shape=(noise_dim,))
    class_input = layers.Input(shape=(1,), dtype=tf.int32)

    # Class embedding → Flatten
    class_embed = layers.Embedding(num_classes, embed_dim)(class_input)
    class_embed = layers.Flatten()(class_embed)

    # Concatenate noise + embedded class
    x = layers.Concatenate()([noise_input, class_embed])
    ############################################################# Generator Architecture ############################################################
    # -----------------------------
    #   Initial Dense + Reshape
    # -----------------------------
    x = layers.Dense(7 * 7 * 256, use_bias=False)(x)  
    x = layers.BatchNormalization()(x)  
    x = layers.LeakyReLU(alpha=0.2)(x)

    x = layers.Reshape((7, 7, 256))(x)

    # -----------------------------
    #   Residual Block at 7×7
    # -----------------------------
    shortcut = x
    x = layers.Conv2D(256, (3,3), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.Conv2D(256, (3,3), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Add()([x, shortcut])  # residual add
    x = layers.LeakyReLU(0.2)(x)

    # -----------------------------
    #   Upsample to 14×14 (256→128 filters)
    # -----------------------------
    x = layers.Conv2DTranspose(
        128, (5,5), strides=(2,2), 
        padding='same', use_bias=False
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)

    # -----------------------------
    #   Residual Block at 14×14
    # -----------------------------
    shortcut = x
    x = layers.Conv2D(128, (3,3), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.Conv2D(128, (3,3), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Add()([x, shortcut])  # residual add
    x = layers.LeakyReLU(0.2)(x)

    # -----------------------------
    #   Upsample to 28×28 (128→64 filters)
    # -----------------------------
    x = layers.Conv2DTranspose(
        64, (5,5), strides=(2,2), 
        padding='same', use_bias=False
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)

    # -----------------------------
    #   Residual Block at 28×28
    # -----------------------------
    shortcut = x
    x = layers.Conv2D(64, (3,3), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.Conv2D(64, (3,3), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Add()([x, shortcut])  
    x = layers.LeakyReLU(0.2)(x)

    # -----------------------------
    #   Final Conv → 1 channel
    # -----------------------------
    x = layers.Conv2D(
        1, (5,5), padding="same", 
        use_bias=False
    )(x)
    x = layers.Activation("tanh")(x)

    # Define the model
    generator_model = keras.models.Model(
        [noise_input, class_input], x, 
        name="better_generator"
    )
    
    ######################################################## Model Summary and Visualization ########################################################
    # get the filepath to the model_architecture_and_summary directory
    model_architecture_and_summary_dir = model_training_output_dir.joinpath("model_architecture_and_summary")
    # create the directory if it doesn't exist
    os.makedirs(model_architecture_and_summary_dir, exist_ok=True)
    
    # save a copy of the model summary to a text file
    model_summary_path = model_architecture_and_summary_dir.joinpath("generator_model_summary.txt")
    with open(model_summary_path, "w") as file:
        generator_model.summary(line_length=150, print_fn=lambda x: file.write(x + "\n"))
    
    # visualize the model architecture with keras.utils.plot_model
    model_visualization_path = model_architecture_and_summary_dir.joinpath("generator_model_architecture.png")
    keras.utils.plot_model(generator_model, to_file=model_visualization_path, show_shapes=True, show_layer_names=True)
    
    # save a copy of the model for visualization with netron.app
    model_netron_path = model_architecture_and_summary_dir.joinpath("generator_model_for_netron_app.keras")
    generator_model.save(model_netron_path)
    
    print(f"\nDone initializing the generator model.")
    print(f"Model summary saved to: {model_summary_path}")
    print(f"Visualization of architecture saved to: {model_visualization_path}")
    print(f"Model saved for visualization with netron.app to: {model_netron_path}\n")
    return generator_model