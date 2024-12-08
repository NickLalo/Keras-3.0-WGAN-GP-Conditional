"""
model definitions and callbacks

Implementation modified from following this tutorial:
https://keras.io/examples/generative/wgan_gp/
"""


import os
from datetime import datetime
import shutil
import pytz
from pathlib import Path
import pandas as pd
import numpy as np
import keras
from keras import layers
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")  # set to use the "Agg" to avoid tkinter error
from PIL import Image
import warnings

from utils import get_memory_usage, get_gpu_memory_usage, get_timestamp, get_readable_time_string


# choose to ignore the specific warning for loading optimizers as our WGAN_GP class handles this in the compile_from_config method. This method 
# may not perfectly recreate the optimizers, but it doesn't seem to cause any issues in testing so far.
warnings.filterwarnings(
    "ignore", 
    message=r"Skipping variable loading for optimizer.*",
    category=UserWarning
)

# Clear all previously registered custom objects. Necessary addition for loading models with custom objects.
keras.saving.get_custom_objects().clear()


def conv_block(
        x,
        filters,
        activation,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="same",
        use_bias=True,
        use_bn=False,
        use_dropout=False,
        drop_value=0.5,
    ):
    """
    convolutional block used in the critic model with optional batch normalization and dropout.
    """
    x = layers.Conv2D(
        filters, kernel_size, strides=strides, padding=padding, use_bias=use_bias
    )(x)
    if use_bn:
        x = layers.BatchNormalization()(x)
    x = activation(x)
    if use_dropout:
        x = layers.Dropout(drop_value)(x)
    return x


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
    # Image input
    img_input = layers.Input(shape=img_shape)
    
    # class input
    class_input = layers.Input(shape=(1,), dtype=tf.int32)  # Assuming labels are integers
    class_input_one_hot = layers.Embedding(num_classes, num_classes)(class_input)  # Convert to one-hot
    class_input_one_hot = layers.Flatten()(class_input_one_hot)  # Flatten embedding output
    
    # Expand labels to match the image dimensions
    # Fully connected layer to reshape one-hot labels to a tensor the same shape as a flattened image
    label_tensor = layers.Dense(np.prod(img_shape))(class_input_one_hot)
    # Reshape to match image shape (height, width, channels)
    label_tensor = layers.Reshape(img_shape)(label_tensor)
    
    # Concatenate image and label tensors
    combined_input = layers.Concatenate()([img_input, label_tensor])
    
    # critic architecture
    x = conv_block(
        combined_input,
        64,
        activation=layers.LeakyReLU(0.2),
        kernel_size=(5, 5),
        strides=(2, 2),
        use_bn=False,
        use_bias=True,
        use_dropout=False,
    )
    x = conv_block(
        x,
        128,
        activation=layers.LeakyReLU(0.2),
        kernel_size=(5, 5),
        strides=(2, 2),
        use_bn=False,
        use_bias=True,
        use_dropout=False,
        # dropout (0.3) removed
    )
    x = conv_block(
        x,
        256,
        activation=layers.LeakyReLU(0.2),
        kernel_size=(5, 5),
        strides=(2, 2),
        use_bn=False,
        use_bias=True,
        use_dropout=False,
        # dropout (0.3) removed
    )
    x = conv_block(
        x,
        512,
        activation=layers.LeakyReLU(0.2),
        kernel_size=(5, 5),
        strides=(2, 2),
        use_bn=False,
        use_bias=True,
        use_dropout=False,
        # dropout (0.3) removed
    )
    
    # Flatten the output and add a dense layer with no activation for a single critic score
    x = layers.Flatten()(x)
    # dropout (0.3) removed
    x = layers.Dense(1)(x)
    
    # Define the model with both inputs
    critic_model = keras.models.Model([img_input, class_input], x, name="critic")
    
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


def upsample_block(
        x,
        filters,
        activation,
        kernel_size=(3, 3),
        strides=(1, 1),
        up_size=(2, 2),
        padding="same",
        use_bn=False,
        use_bias=True,
        use_dropout=False,
        drop_value=0.3,
    ):
    """
    Creates an upsampling block used by the generator with optional batch normalization and dropout.
    Parameters:
    x (tensor): Input tensor.
    filters (int): Number of filters for the Conv2D layer.
    activation (function): Activation function to use.
    kernel_size (tuple): Size of the convolution kernel. Default is (3, 3).
    strides (tuple): Strides for the convolution. Default is (1, 1).
    up_size (tuple): Size for the upsampling. Default is (2, 2).
    padding (str): Padding type for the Conv2D layer. Default is "same".
    use_bn (bool): Whether to use batch normalization. Default is False.
    use_bias (bool): Whether the Conv2D layer uses a bias vector. Default is True.
    use_dropout (bool): Whether to use dropout. Default is False.
    drop_value (float): Dropout rate. Default is 0.3.
    Returns:
    tensor: Output tensor after applying upsampling, convolution, optional batch normalization, activation, and optional dropout.
    """
    x = layers.UpSampling2D(up_size)(x)
    x = layers.Conv2D(
        filters, kernel_size, strides=strides, padding=padding, use_bias=use_bias
    )(x)
    
    if use_bn:
        x = layers.BatchNormalization()(x)
    
    if activation:
        x = activation(x)
    if use_dropout:
        x = layers.Dropout(drop_value)(x)
    return x


@keras.saving.register_keras_serializable(package="generator_model", name="generator_model")
def get_generator_model(noise_dim_shape: int, num_classes: int, model_training_output_dir: Path):
    """
    Builds and returns a generator model for a Wasserstein GAN with Gradient Penalty (WGAN-GP).
    Parameters:
        noise_dim_shape (int): Dimension of the noise vector input.
        num_classes (int): Number of classes for the class conditional input.
        model_training_output_dir (Path): Directory path for saving model training outputs.
    Returns:
        keras.models.Model: A Keras Model representing the generator.
    """
    # model inputs will look like this: [noise_input, class_input]
    # Noise input
    noise_input = layers.Input(shape=(noise_dim_shape,))
    
    # class input (conditional input)
    class_input = layers.Input(shape=(1,), dtype=tf.int32)  # Assuming class_inputs are integers
    class_input_one_hot = layers.Embedding(num_classes, num_classes)(class_input)  # Convert to one-hot
    class_input_one_hot = layers.Flatten()(class_input_one_hot)  # Flatten embedding output
    
    # Concatenate noise input and class input
    x = layers.Concatenate()([noise_input, class_input_one_hot])
    
    # Generator architecture
    x = layers.Dense(4 * 4 * 256, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    
    x = layers.Reshape((4, 4, 256))(x)
    x = upsample_block(
        x,
        256,
        layers.LeakyReLU(0.2),
        strides=(1, 1),
        use_bias=False,
        use_bn=True,
        padding="same",
        use_dropout=False,
    )
    x = upsample_block(
        x,
        128,
        layers.LeakyReLU(0.2),
        strides=(1, 1),
        use_bias=False,
        use_bn=True,
        padding="same",
        use_dropout=False,
    )
    x = upsample_block(
        x, 
        1, 
        layers.Activation("tanh"), 
        strides=(1, 1), 
        use_bias=False, 
        use_bn=True,
    )
    # At this point, we have an output which has a shape of (32, 32, 1) but we want our output to be (28, 28, 1).
    # We will use a Cropping2D layer to make it (28, 28, 1).
    x = layers.Cropping2D((2, 2))(x)
    
    # Define the model with both inputs
    generator_model = keras.models.Model([noise_input, class_input], x, name="generator")
    
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


@keras.saving.register_keras_serializable(package="custom_wgan_gp", name="custom_wgan_gp")
class WGAN_GP(keras.Model):
    """
    WGAN-GP model
    """
    def __init__(
        self,
        critic,
        generator,
        num_classes,
        latent_dim,
        learning_rate,
        learning_rate_warmup_epochs,
        learning_rate_decay,
        critic_extra_steps=5,
        critic_input_shape=None,
        gp_weight=10.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.critic = critic
        self.generator = generator
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        # REVIEW: learning rate value that is updated by the learning rate scheduler. Used to set the initial values of the optimizers, but I'm not 
        # sure if keeping track of it here is actually necessary because we save/load the optimizers directly so the value should remain the same.
        self.learning_rate = learning_rate
        # REVIEW: learning rate warmup epochs value that is used by the learning rate scheduler. This is the number of epochs the model trains for
        #         before updating the learning rate.
        self.learning_rate_warmup_epochs = learning_rate_warmup_epochs
        self.learning_rate_decay = learning_rate_decay
        # in a WGAN-GP, the critic trains for a number of steps and then generator trains for one step
        self.num_critic_steps = critic_extra_steps
        self.critic_input_shape = critic_input_shape
        self.gp_weight = gp_weight
        
        # set dummy value for self.optimizer avoid errors at the end of .fit() call
        self.optimizer = None
        
        # build the model
        self.build()
        return
    
    def compile(self, critic_optimizer=None, gen_optimizer=None):
        super(WGAN_GP, self).compile()
        self.critic_optimizer = critic_optimizer
        self.gen_optimizer = gen_optimizer
        return
    
    def gradient_penalty(self, batch_size, real_images, fake_images, image_one_hot_labels):
        alpha = tf.random.normal([batch_size, 1, 1, 1], 0.0, 1.0)
        diff = fake_images - real_images
        interpolated = real_images + alpha * diff
        
        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            pred = self.critic(
                [interpolated, image_one_hot_labels], training=True
            )
        
        grads = gp_tape.gradient(pred, [interpolated])[0]
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp
    
    def train_step(self, batch_data):
        # unpack the batch data into images and labels
        real_images, real_labels = batch_data
        # Get the batch size from the data as sometimes the last batch can be smaller
        batch_size = tf.shape(real_images)[0]
        
        # For each batch, we are going to perform the
        # following steps as laid out in the original paper:
        # 1. Train the generator and get the generator loss
        # 2. Train the critic and get the critic loss
        # 3. Calculate the gradient penalty
        # 4. Multiply this gradient penalty with a constant weight factor
        # 5. Add the gradient penalty to the critic loss
        # 6. Return the generator and critic losses as a loss dictionary
        
        # # Initialize lists to track critic scores on real and fake images
        critic_real_scores = []
        critic_fake_scores = []
        gradient_penalties = []
        critic_losses = []
        
        ####################################### Train the critic #######################################
        for _ in range(self.num_critic_steps):
            # Get a batch of random latent vectors
            random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
            
            with tf.GradientTape() as tape:
                # Generate fake images using the real labels for training the critic
                fake_images = self.generator([random_latent_vectors, real_labels], training=True)
                
                # Get the critic scores (logits) for the fake and real images
                fake_logits = self.critic([fake_images, real_labels], training=True)
                real_logits = self.critic([real_images, real_labels], training=True)
                
                # Append the mean scores to track them
                critic_real_scores.append(tf.reduce_mean(real_logits))
                critic_fake_scores.append(tf.reduce_mean(fake_logits))
                
                # Calculate the critic loss
                # critic_cost = self.critic_loss_fn(real_img=real_logits, fake_img=fake_logits)
                # Calculate the critic loss using the Wasserstein loss
                critic_wasserstein_loss = tf.reduce_mean(fake_logits) - tf.reduce_mean(real_logits)
                
                # Calculate the gradient penalty
                gp = self.gradient_penalty(batch_size, real_images, fake_images, real_labels)
                
                # Add the gradient penalty to the critic loss
                critic_loss = critic_wasserstein_loss + gp * self.gp_weight
                
                # Append the gradient penalty and critic loss to track them
                gradient_penalties.append(gp)
                critic_losses.append(critic_loss)
            
            # Get the gradients for the critic loss
            critic_gradient = tape.gradient(critic_loss, self.critic.trainable_variables)
            # Update the critic's weights
            self.critic_optimizer.apply_gradients(zip(critic_gradient, self.critic.trainable_variables))
            
            # REVIEW: consider resampling the dataset to get a new batch of real images and labels here to avoid training on the same batch
            #         multiple times. This could be achieved by making a copy of the dataset a class attribute.
            # Example:
            # real_images, real_labels = next(iter(self.train_dataset))
        
        ######################################### Train the generator #########################################
        # Generate a new batch of random latent vectors
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        
        with tf.GradientTape() as tape:
            # Generate fake images using real labels for training the generator
            generated_images = self.generator([random_latent_vectors, real_labels], training=True)
            
            # Get the critic scores (logits) for the generated images
            gen_img_logits = self.critic([generated_images, real_labels], training=True)
            
            # Calculate the generator loss
            gen_loss = -tf.reduce_mean(gen_img_logits)
        
        # Get the gradients for the generator loss
        gen_gradient = tape.gradient(gen_loss, self.generator.trainable_variables)
        # Update the generator's weights
        self.gen_optimizer.apply_gradients(zip(gen_gradient, self.generator.trainable_variables))
        
        # # Calculate the average scores for real and fake images for this batch
        avg_real_score = tf.reduce_mean(critic_real_scores)
        avg_fake_score = tf.reduce_mean(critic_fake_scores)
        avg_gradient_penalty = tf.reduce_mean(gradient_penalties)
        avg_critic_loss = tf.reduce_mean(critic_losses)
        
        return_dict = {
            "critic_loss": avg_critic_loss,
            "gen_loss": gen_loss,
            "critic_loss_real": avg_real_score,
            "critic_loss_fake": avg_fake_score,
            "gradient_penalty": avg_gradient_penalty
            }
        
        return return_dict
    
    def build(self, input_shape=None):
        # Explicitly build the generator and critic models
        self.generator.build(input_shape=(None, self.latent_dim))
        self.critic.build(input_shape=self.critic_input_shape)
        super().build(input_shape)
        return
    
    def get_config(self):
        """
        Custom saving step 1: get_config()
        More info on custom loading and saving:
        https://keras.io/guides/serialization_and_saving/
        Runs as part of the saving process to serialize the model when calling model.save()
        Runs first to get the default configuration for the model
        """
        # get the default configuration for the model
        base_config = super().get_config()
        
        # TODO: instead of saving the critic and generator models to the custom config, instead save them as separate files here.
        # get extra configurations for custom model attributes (not Python objects like ints, strings, etc.)
        custom_config = {
            "critic": keras.saving.serialize_keras_object(self.critic),
            "generator": keras.saving.serialize_keras_object(self.generator),
            "num_classes": self.num_classes,
            "latent_dim": self.latent_dim,
            "critic_extra_steps": self.num_critic_steps,
            "gp_weight": self.gp_weight,
            "learning_rate": self.learning_rate,
            "learning_rate_warmup_epochs": self.learning_rate_warmup_epochs,
            "learning_rate_decay": self.learning_rate_decay,
        }
        
        # combine the two configurations
        config = {**base_config, **custom_config}
        return config
    
    def get_compile_config(self):
        """
        Custom saving step 2: get_compile_config()
        More info on custom loading and saving:
        https://keras.io/guides/customizing_saving_and_serialization/#getcompileconfig-and-compilefromconfig
        Runs as part of the saving process to serialize the compiled parameters when calling model.save()
        Runs after get_config() to serialize the compiled parameters
        """
        # These parameters will be serialized at saving time
        config = {
            "critic_optimizer": self.critic_optimizer.get_config(),
            "critic_optimizer_state": [v.numpy() for v in self.critic_optimizer.variables],
            "gen_optimizer": self.gen_optimizer.get_config(),
            "gen_optimizer_state": [v.numpy() for v in self.gen_optimizer.variables],
        }
        return config
    
    @classmethod
    def from_config(cls, config, custom_objects=None):
        """
        Custom loading step 1: from_config()
        More info on custom loading and saving:
        https://keras.io/guides/serialization_and_saving/
        Runs as part of the loading process to deserialize the model when calling keras.models.load_model()
        Runs first to deserialize the model configuration
        """
        # TODO: instead of loading the critic and generator models from the config, load them from separate files here.
        #       but, I wonder how I will get the paths to the files to load them...
        
        # get the critic and generator models from the config
        critic = keras.saving.deserialize_keras_object(config.pop("critic"), custom_objects=custom_objects)
        generator = keras.saving.deserialize_keras_object(config.pop("generator"), custom_objects=custom_objects)
        num_classes = config.pop("num_classes")
        latent_dim = config.pop("latent_dim")
        learning_rate = config.pop("learning_rate")
        learning_rate_warmup_epochs = config.pop("learning_rate_warmup_epochs")
        learning_rate_decay = config.pop("learning_rate_decay")
        critic_extra_steps = config.pop("critic_extra_steps")
        gp_weight = config.pop("gp_weight")
        
        # create a new instance of the model with the critic and generator models (This calls the __init__ method)
        model = cls(
            critic=critic, 
            generator=generator,
            num_classes=num_classes,
            latent_dim=latent_dim,
            learning_rate=learning_rate,
            learning_rate_warmup_epochs=learning_rate_warmup_epochs,
            learning_rate_decay=learning_rate_decay,
            critic_extra_steps=critic_extra_steps, 
                    gp_weight=gp_weight)
        # REVIEW: the line above model = cls() is where our code is erroring?
        # 'learning_rate', 'learning_rate_warmup_epochs', and 'learning_rate_decay' are not in the config and need to be
        return model # REVIEW: the line above model = cls() is where our code is erroring?
    
    def compile_from_config(self, config):
        """
        Custom loading step 2: compile_from_config()
        More info on custom loading and saving:
        https://keras.io/guides/customizing_saving_and_serialization/#getcompileconfig-and-compilefromconfig
        Runs as part of the loading process to deserialize the compiled parameters when calling keras.models.load_model()
        Runs after from_config() to deserialize the compiled parameters
        """
        # Deserialize the compiled parameters
        self.critic_optimizer = keras.optimizers.Adam.from_config(config["critic_optimizer"])
        self.critic_optimizer_state = config["critic_optimizer_state"]
        for var, val in zip(self.critic_optimizer.variables, self.critic_optimizer_state):
            var.assign(val)
        
        self.gen_optimizer = keras.optimizers.Adam.from_config(config["gen_optimizer"])
        self.gen_optimizer_state = config["gen_optimizer_state"]
        for var, val in zip(self.gen_optimizer.variables, self.gen_optimizer_state):
            var.assign(val)
        
        # call the compile method to set the optimizer and loss functions
        self.compile(
            critic_optimizer=self.critic_optimizer, 
            gen_optimizer=self.gen_optimizer,
        )
        return


class Training_Monitor(tf.keras.callbacks.Callback):
    """
    Training_Monitor is a custom Keras callback for monitoring and logging the training process of a GAN model. It performs various tasks at the end
    of each epoch, including logging loss metrics, saving model checkpoints, generating validation samples, and creating plots and GIFs of the 
    training progress.
    Attributes:
        model_training_output_dir (str): Directory to save outputs from the model training.
        model_checkpoints_dir (str): Directory to save model checkpoints and info at each epoch.
        num_classes (int): Number of classes in the dataset.
        num_img (int): Number of images to generate at the end of each epoch.
        latent_dim (int): Latent dimension of the generator.
        grid_size (tuple): Size of the grid for the generated images.
        samples_per_epoch (int): Number of samples in the training dataset.
        last_checkpoint_dir_path (str, optional): Path to the last checkpoint directory if the model is being reloaded.
        random_latent_vectors (tf.Tensor): Pre-generated random latent vectors for generating validation samples.
        loss_metrics_dataframe (pd.DataFrame): DataFrame to track loss metrics.
        model_recently_loaded (bool): Flag to indicate if the model was recently loaded from a checkpoint.
        gif_creation_frequency (int): Frequency (in epochs) to create a GIF of validation samples.
    Methods:
        on_epoch_end(epoch, logs=None):
            Called at the end of each epoch during training. Logs loss metrics, saves model checkpoints, generates validation samples, and creates 
            plots.
        on_train_end(logs=None):
            Called at the end of training. Generates a GIF of all saved validation samples.
        set_this_epoch_checkpoint_dir():
            Sets the directory for the current epoch's checkpoint and creates it if it doesn't exist.
        log_loss_to_dataframe(logs):
        create_loss_plots():
            Generates and saves individual and combined loss plots for the training process.
        generate_validation_samples():
            Generates a set of validation images using the generator model and saves them as a grid.
        generate_gif():
            Creates a GIF of the validation images generated at regular intervals.
    """
    def __init__(self, 
                model_training_output_dir, 
                model_checkpoints_dir, 
                num_classes,
                num_img, 
                latent_dim, 
                grid_size, 
                samples_per_epoch=0,
                last_checkpoint_dir_path=None
                ):
        """
        Parameters:
            model_training_output_dir: str, the main directory to save outputs from the model training
            model_checkpoints_dir: str, the directory to save the model checkpoints and info at each epoch
            num_img: int, the number of images to generate at the end of each epoch
            latent_dim: int, the latent dimension of the generator
            grid_size: tuple, the size of the grid for the generated images
            samples_per_epoch: int, the number of samples in the training dataset
        """
        self. model_training_output_dir = model_training_output_dir
        self.model_checkpoints_dir = model_checkpoints_dir
        self.this_epoch_checkpoint_dir = None  # the path to the current model checkpoint (will be set/updated in on_epoch_end)
        self.num_classes = num_classes
        
        # parameters for logging the duration of model training and metric handling
        self.epoch_start_time = None
        self.epoch_train_duration = None
        self.metric_calc_start_time = None
        # placeholder for the duration of logging the metrics because this is actually logging the duration of the last's epoch's metric logging
        # duration, but it's close enough for our purposes.
        self.metric_calc_duration = datetime.now() - datetime.now()
        
        # parameters for generating validation samples
        self.num_img = num_img
        self.latent_dim = latent_dim
        self.grid_size = grid_size  # Grid size as a tuple (rows, cols)
        
        # deterministically generate the random latent vectors for generating validation samples so that the same random latent vectors are used
        # every time the model is reloaded.
        generator = tf.random.Generator.from_seed(112)
        self.random_latent_vectors = generator.normal(shape=(self.num_img, self.latent_dim))
        self.random_labels = generator.uniform(shape=(self.num_img, 1), minval=0, maxval=self.num_classes, dtype=tf.int32)
        
        self.gif_creation_frequency = 5  # create a GIF every 5 epochs
        self.samples_per_epoch = samples_per_epoch
        
        # load the loss metrics csv to a dataframe if the last_checkpoint_dir_path is provided (we are reloading the model)
        if last_checkpoint_dir_path is not None:
            # get the path to the loss_metrics.csv file
            loss_metrics_path = last_checkpoint_dir_path.joinpath("loss_metrics.csv")
            # load the csv to a dataframe
            self.metrics_dataframe = pd.read_csv(loss_metrics_path)
            # remember that the model was loaded for logging loss metrics at the end of the next epoch
            self.model_recently_loaded = True
        else:  # create a new dataframe for tracking loss metrics for a fresh start
            self.metrics_dataframe = pd.DataFrame()  # empty dataframe placeholder that will be overwritten in log_loss_to_dataframe
            self.model_recently_loaded = False
        
        # if it does not exist in the model_training_output_dir, create a _NOTES.txt file for writing notes about the model training run
        notes_file_path = model_training_output_dir.joinpath("_NOTES.txt")
        if not os.path.exists(notes_file_path):
            with open(notes_file_path, "w") as file:
                file.write("A place to write notes about this model training run.\n")
        return
    
    def on_epoch_begin(self, epoch, logs=None):
        """
        Called at the start of each epoch during training.
        Parameters:
            epoch (int): The current epoch number.
            logs (dict, optional): Dictionary of logs containing loss metrics.
        Returns:
            None
        """
        # get the start time of the epoch
        self.epoch_start_time = datetime.now()
        return
    
    def on_epoch_end(self, epoch, logs=None):
        """
        Called at the end of each epoch during training.
        Parameters:
            epoch (int): The current epoch number.
            logs (dict, optional): Dictionary of logs containing loss metrics.
            This method performs the following tasks:
            1. Logs the loss metrics to a DataFrame.
            2. Plots individual loss metrics and saves the plots to the specified directory.
            3. Creates a combined plot for specific losses and saves it.
            4. Generates a set of validation images using the generator model.
            7. Every 5 epochs, generates a GIF of all saved images.
            8. Saves a copy of the model to the current epoch checkpoint directory.
        Returns:
            None
        """
        # get the duration of the time spent training this epoch
        self.epoch_train_duration = datetime.now() - self.epoch_start_time
        # start the timer for calculating the duration of logging metrics
        self.metric_calc_start_time = datetime.now()
        
        # get the current epoch number for logging metrics
        self.update_current_epoch()
        
        # Set the current epoch checkpoint directory
        self.set_this_epoch_checkpoint_dir()
        
        # Log the data about training to the DataFrame
        self.log_data_to_dataframe(logs)
        
        # Plot tracked metrics
        self.plot_training_metrics()
        
        # update the learning rates of the optimizers if past the warmup period
        self.learning_rate_scheduler()
        
        # Save a copy of the model to the current epoch checkpoint directory
        # NOTE: this could be changed to every X number of epochs, but then the model loading code would need to be updated as well.
        self.save_model_checkpoint()
        
        # Generate validation samples
        self.generate_validation_samples()
        
        # every X number of epochs, generate a GIF of all the saved images
        if self.current_epoch % self.gif_creation_frequency == 0 and epoch > 0:
            self.generate_gif()
        
        # Calculate the duration of logging the metrics (will be logged in the next epoch log_data_to_dataframe call)
        self.metric_calc_duration = datetime.now() - self.metric_calc_start_time
        
        # plot train time and metrics calculations and estimate the time to train for a number of epochs. Even though the metric calc time
        # doesn't include this time, it's tracked for the operations that take the most time.
        self.plot_epoch_duration_and_estimate_train_time()
        return
    
    def on_train_end(self, logs=None):
        epoch = len(self.metrics_dataframe)
        # skip generating a gif if training ends on a multiple of the frequency to create a gif because it was already generated in the 
        # generate_validation_samples method at the end of the last epoch.
        if epoch % self.gif_creation_frequency != 0:
            # Generate a GIF of all the saved images
            self.generate_gif()
        return
    
    def update_current_epoch(self):
        # get the current epoch number for logging metrics. +1 because we are saying: "These are metrics at the end of one epoch" for the first epoch
        # and we haven't yet logged metrics for the current epoch so the dataframe has one less row than the current epoch number at this point.
        self.current_epoch = len(self.metrics_dataframe) + 1
        return
    
    def set_this_epoch_checkpoint_dir(self):
        # create a new checkpoint directory for logging info at the end of this epoch
        self.this_epoch_checkpoint_dir = self.model_checkpoints_dir.joinpath(f"epoch_{self.current_epoch:04d}")
        # error out if this directory already exists
        if os.path.exists(self.this_epoch_checkpoint_dir):
            error_string = (
                f"Checkpoint directory already exists: {self.this_epoch_checkpoint_dir}. Please ensure that any checkpoint past the intended loading"
                " point is deleted before continuing."
            )
            raise ValueError(error_string)
        os.makedirs(self.this_epoch_checkpoint_dir, exist_ok=True)
        return
    
    def log_data_to_dataframe(self, logs):
        """
        Logs the loss metrics to a DataFrame and saves it to a CSV file.
        This method extracts the critic and generator loss values from the provided logs dictionary,
        prints them, and appends them to an internal DataFrame. The DataFrame is then saved to a CSV file.
        Parameters:
            logs (dict): A dictionary containing the loss values with the following keys:
                - "critic_loss": critic total loss
                - "critic_loss_real": critic loss on real samples
                - "critic_loss_fake": critic loss on fake samples
                - "gradient_penalty": critic gradient penalty loss
                - "gen_loss": Generator loss
        Returns:
            None
        """
        # extract info from logs dictionary
        critic_loss = float(logs["critic_loss"])
        critic_loss_real = float(logs["critic_loss_real"])
        critic_loss_fake = float(logs["critic_loss_fake"])
        gradient_penalty = float(logs["gradient_penalty"])
        gen_loss = float(logs["gen_loss"])
        
        # get the current memory usage in MB and GB
        mem_usage_mb, mem_usage_gb = get_memory_usage()
        # get the current GPU memory usage in MB and GB
        gpu_mem_usage_mb, gpu_mem_usage_gb = get_gpu_memory_usage()
        
        # get the critic learning rate
        critic_learning_rate = self.model.critic_optimizer.learning_rate.numpy()
        generator_learning_rate = self.model.gen_optimizer.learning_rate.numpy()
        
        # Get the current time in the US Central Time Zone
        formatted_timestamp = get_timestamp()
        
        # create a new DataFrame row with the current epoch data
        new_row = pd.DataFrame([{
            "epoch": self.current_epoch,
            "critic_loss": critic_loss,
            "critic_loss_real": critic_loss_real,
            "critic_loss_fake": critic_loss_fake,
            "gradient_penalty": gradient_penalty,
            "gen_loss": gen_loss,
            # calculate the number of samples trained on at the end of this epoch
            "samples_trained_on": self.current_epoch * self.samples_per_epoch,
            "model_loaded": self.model_recently_loaded,
            "memory_usage_gb": mem_usage_gb,
            "gpu_memory_usage_gb": gpu_mem_usage_gb,
            "critic_learning_rate": critic_learning_rate,
            "generator_learning_rate": generator_learning_rate,
            "timestamp_CST": formatted_timestamp,
            "epoch_train_time": self.epoch_train_duration.total_seconds(),
            "metric_calc_time": 0.0,  # placeholder value that will be replaced in plot_epoch_duration_and_estimate_train_time
            "total_iteration_time": 0.0,  # placeholder value that will be replaced in plot_epoch_duration_and_estimate_train_time
        }])
        
        # if the model was loaded, set the model_loaded column to False for the next epoch
        self.model_recently_loaded = False  # we only want to mark the first epoch after loading
        
        if self.current_epoch == 1:
            # if this is the first epoch, set the metrics_dataframe to the new row
            self.metrics_dataframe = new_row
        else:
            # concatenate the new row to the existing loss dataframe
            self.metrics_dataframe = pd.concat([self.metrics_dataframe, new_row], ignore_index=True)
        
        # dataframe is saved to a CSV file at in plot_epoch_duration_and_estimate_train_time
        return
    
    def plot_training_metrics(self):
        """
        Generates and saves individual and combined loss plots for the training process with a secondary x-axis showing
        the number of samples trained on.
        """
        # Define list of individual loss columns to plot
        plot_columns = ["critic_loss", "critic_loss_real", "critic_loss_fake", "gen_loss", "gradient_penalty"]
        plot_line_colors = ["#1F77B4", "#FF7F0E", "#2CA02C", "#D62728", "#9467BD"]
        
        # Identify locations where model was loaded
        model_load_indices = self.metrics_dataframe.loc[self.metrics_dataframe['model_loaded'], 'epoch']
        
        # get the number of samples trained and epochs for the x-axis
        samples_trained = self.metrics_dataframe['samples_trained_on']
        epochs = self.metrics_dataframe['epoch']
        
        ################################ Loop through each loss metric to create individual plots ###############################
        for color_index, col in enumerate(plot_columns):
            # Create the plot
            fig, ax1 = plt.subplots(figsize=(10, 6))
            
            # Plot the loss metric
            line_color = plot_line_colors[color_index] # get the color for the current loss metric
            ax1.plot(epochs, self.metrics_dataframe[col], label=col.replace("_", " ").title(), zorder=1, color=line_color)
            
            # Add vertical lines for model loading events with a single label for legend
            for i, model_load_epoch in enumerate(model_load_indices):
                if i == 0:
                    line_label = "ckpt loaded"
                else:
                    line_label = ""
                ax1.axvline(x=model_load_epoch, color='#5C5C5C', linestyle='--', alpha=0.7, linewidth=1.5, label=line_label, zorder=-1)
                # add text to show the number of samples at this checkpoint
                y_text_position = ((ax1.get_ylim()[1] - ax1.get_ylim()[0]) * 0.02) + ax1.get_ylim()[0]
                ax1.text(model_load_epoch, y_text_position, f"{model_load_epoch:,}", color='#5C5C5C', fontsize=9, rotation=90, va='bottom', 
                        ha='right', zorder=2)
            
            # add text to the plot to show the number of samples trained on at the end of the last epoch
            plt.text(0.86, 1.012, f"Samples Trained On: {samples_trained.iloc[-1]:,}", fontsize=9, ha='center', va='center', 
                    transform=plt.gca().transAxes)
            
            ax1.set_xlabel("Epoch")
            ax1.set_ylabel(col.replace("_", " ").title())
            ax1.legend()
            ax1.grid(True, alpha=0.4)
            plt.title(f"{col.replace('_', ' ').title()} vs. Epoch", fontsize=18, pad=10)
            plt.tight_layout()
            
            # move the word "loss" to the front of the filename for better sorting for the loss plots
            col_for_filename = col.replace("_loss", "")
            col_for_filename = f"loss_{col_for_filename}"
            
            # Save plot to the current epoch checkpoint directory
            plot_path = self.this_epoch_checkpoint_dir.joinpath(f"{col_for_filename}.png")
            plt.savefig(plot_path, dpi=200)
            shutil.copy(plot_path, self.model_training_output_dir.joinpath(f"{col_for_filename}.png"))
            plt.close()
            plt.clf()
        
        ################################ Combined plot for specific losses ###############################
        combined_loss_columns = ["critic_loss", "critic_loss_real", "critic_loss_fake", "gen_loss", "gradient_penalty"]
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        # Plot the loss metrics
        for color_index, col in enumerate(combined_loss_columns):
            line_color = plot_line_colors[color_index]  # get the color for the current loss metric
            ax1.plot(epochs, self.metrics_dataframe[col], label=col.replace("_", " ").title(), zorder=1, color=line_color)
        
        # Add vertical lines for model loading events with a single label for legend
        for i, model_load_epoch in enumerate(model_load_indices):
            if i == 0:
                line_label = "ckpt loaded"
            else:
                line_label = ""
            ax1.axvline(x=model_load_epoch, color='#5C5C5C', linestyle='--', alpha=0.7, linewidth=1.5, label=line_label, zorder=-1)
            # add text to show the number of samples at this checkpoint
            y_text_position = ((ax1.get_ylim()[1] - ax1.get_ylim()[0]) * 0.02) + ax1.get_ylim()[0]
            ax1.text(model_load_epoch, y_text_position, f"{model_load_epoch:,}", color='#5C5C5C', fontsize=9, rotation=90, va='bottom', ha='right',
                    zorder=2)
        
        # add text to the plot to show the number of samples trained on at the end of the last epoch
        plt.text(0.86, 1.012, f"Samples Trained On: {samples_trained.iloc[-1]:,}", fontsize=9, ha='center', va='center', 
                transform=plt.gca().transAxes)
        
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss Value")
        ax1.legend()
        ax1.grid(True, alpha=0.4)
        plt.title("All Losses vs. Epoch", fontsize=18, pad=10)
        plt.tight_layout()
        
        # Save the combined plot to the current epoch checkpoint directory
        combined_plot_path = self.this_epoch_checkpoint_dir.joinpath("losses_combined.png")
        plt.savefig(combined_plot_path, dpi=200)
        combined_plot_main_dir_path = self.model_training_output_dir.joinpath("losses_combined.png")
        shutil.copy(combined_plot_path, combined_plot_main_dir_path)
        plt.close()
        plt.clf()
        print(f"\nCombined loss plot saved to: {combined_plot_main_dir_path}")
        
        ################################ Stacked plots for critic and generator learning rates ###############################
        loss_columns = ["critic_loss", "gen_loss"]
        learning_rates_columns = ["critic_learning_rate", "generator_learning_rate"]
        plot_line_colors = ["#1F77B4", "#D62728"]
        
        for loss_col, lr_col, line_color in zip(loss_columns, learning_rates_columns, plot_line_colors):
            # Create a learning rate plot with a shared x-axis of the loss
            fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
            
            # Plot the loss metric
            axs[0].plot(epochs, self.metrics_dataframe[loss_col], label=loss_col.replace("_", " ").title(), zorder=1, color=line_color)
            axs[0].set_ylabel(loss_col.replace("_", " ").title())
            axs[0].legend()
            axs[0].grid(True, alpha=0.4)
            
            # Plot the learning rate
            axs[1].plot(epochs, self.metrics_dataframe[lr_col], label=lr_col.replace("_", " ").title(), zorder=1, color=line_color)
            axs[1].set_ylabel(lr_col.replace("_", " ").title())
            axs[1].legend()
            axs[1].grid(True, alpha=0.4)
            
            # Add vertical lines for model loading events with a single label for legend
            for i, model_load_epoch in enumerate(model_load_indices):
                if i == 0:
                    line_label = "ckpt loaded"
                else:
                    line_label = ""
                axs[0].axvline(x=model_load_epoch, color='#5C5C5C', linestyle='--', alpha=0.7, linewidth=1.5, label=line_label, zorder=-1)
                axs[1].axvline(x=model_load_epoch, color='#5C5C5C', linestyle='--', alpha=0.7, linewidth=1.5, label=line_label, zorder=-1)
                # add text to show the number of samples at this checkpoint
                y_text_position = ((axs[0].get_ylim()[1] - axs[0].get_ylim()[0]) * 0.02) + axs[0].get_ylim()[0]
                axs[0].text(model_load_epoch, y_text_position, f"{model_load_epoch:,}", color='#5C5C5C', fontsize=9, rotation=90, va='bottom', 
                        ha='right', zorder=2)
                axs[1].text(model_load_epoch, y_text_position, f"{model_load_epoch:,}", color='#5C5C5C', fontsize=9, rotation=90, va='bottom', 
                        ha='right', zorder=2)
            
            # add text to the plot to show the number of samples trained on at the end of the last epoch
            plt.text(0.86, 1.028, f"Samples Trained On: {samples_trained.iloc[-1]:,}", fontsize=9, ha='center', va='center',
                        transform=plt.gca().transAxes)
            
            plt.suptitle(f"{loss_col.replace('_', ' ').title()} and {lr_col.replace('_', ' ').title()} vs. Epoch", fontsize=18, y=0.96)
            axs[1].set_xlabel("Epoch")
            plt.tight_layout()
            
            # move "learning_rate" to the front of the filename for better sorting of the plots in the directory
            filename = lr_col.split("_")[0]  # get the name of the optimizer from the column name
            filename = f"learning_rate_{filename}"
            
            # Save the plot to the current epoch checkpoint directory
            plot_path = self.this_epoch_checkpoint_dir.joinpath(f"{filename}.png")
            plt.savefig(plot_path, dpi=200)
            shutil.copy(plot_path, self.model_training_output_dir.joinpath(f"{filename}.png"))
            plt.close()
            plt.clf()
        
        ############################### create a plot to show memory usage over time ##############################
        fig, ax1 = plt.subplots(figsize=(10, 6))
        # Plot the memory usage in GB
        ax1.plot(epochs, self.metrics_dataframe["memory_usage_gb"], label="Memory Usage (GB)", zorder=1)
        
        # Add vertical lines for model loading events with a single label for legend
        for i, model_load_epoch in enumerate(model_load_indices):
            if i == 0:
                line_label = "ckpt loaded"
            else:
                line_label = ""
            ax1.axvline(x=model_load_epoch, color='#5C5C5C', linestyle='--', alpha=0.7, linewidth=1.5, label=line_label, zorder=-1)
            # add text to show the number of samples at this checkpoint
            y_text_position = ((ax1.get_ylim()[1] - ax1.get_ylim()[0]) * 0.02) + ax1.get_ylim()[0]
            ax1.text(model_load_epoch, y_text_position, f"{model_load_epoch:,}", color='#5C5C5C', fontsize=9, rotation=90, va='bottom', ha='right',
                    zorder=2)
        
        # add text to the plot to show the number of samples trained on at the end of the last epoch
        plt.text(0.86, 1.012, f"Samples Trained On: {samples_trained.iloc[-1]:,}", fontsize=9, ha='center', va='center', 
                transform=plt.gca().transAxes)
        
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Memory Usage (GB)")
        ax1.legend()
        ax1.grid(True, alpha=0.4)
        plt.title("Memory Usage vs. Epoch")
        plt.tight_layout()
        
        # Save the memory usage plot to the model training output directory
        memory_plot_path = self.model_training_output_dir.joinpath("memory_usage.png")
        plt.savefig(memory_plot_path, dpi=200)
        plt.close()
        plt.clf()
        
        ############################## create a plot to show GPU memory usage over time ##############################
        fig, ax1 = plt.subplots(figsize=(10, 6))
        # Plot the GPU memory usage in GB
        ax1.plot(epochs, self.metrics_dataframe["gpu_memory_usage_gb"], label="GPU Memory Usage (GB)", zorder=1)
        
        for i, model_load_epoch in enumerate(model_load_indices):
            if i == 0:
                line_label = "ckpt loaded"
            else:
                line_label = ""
            ax1.axvline(x=model_load_epoch, color='#5C5C5C', linestyle='--', alpha=0.7, linewidth=1.5, label=line_label, zorder=-1)
            # add text to show the number of samples at this checkpoint
            y_text_position = ((ax1.get_ylim()[1] - ax1.get_ylim()[0]) * 0.02) + ax1.get_ylim()[0]
            ax1.text(model_load_epoch, y_text_position, f"{model_load_epoch:,}", color='#5C5C5C', fontsize=9, rotation=90, va='bottom', ha='right',
                    zorder=2)
        
        # add text to the plot to show the number of samples trained on at the end of the last epoch
        plt.text(0.86, 1.012, f"Samples Trained On: {samples_trained.iloc[-1]:,}", fontsize=9, ha='center', va='center', 
                transform=plt.gca().transAxes)
        
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("GPU Memory Usage (GB)")
        ax1.legend()
        ax1.grid(True, alpha=0.4)
        plt.title("GPU Memory Usage vs. Epoch")
        plt.tight_layout()
        
        # Save the GPU memory usage plot to the model training output directory
        plot_path = self.model_training_output_dir.joinpath("memory_usage_GPU.png")
        plt.savefig(plot_path, dpi=200)
        plt.close()
        plt.clf()
        
        return
    
    def learning_rate_scheduler(self):
        # if we have passed the warmup period, start to decay the learning rates
        if self.current_epoch > self.model.learning_rate_warmup_epochs:
            self.model.critic_optimizer.learning_rate = self.model.critic_optimizer.learning_rate * self.model.learning_rate_decay
            self.model.gen_optimizer.learning_rate = self.model.gen_optimizer.learning_rate * self.model.learning_rate_decay
        return
    
    def save_model_checkpoint(self):
        # save a copy of the model to the current epoch checkpoint directory
        model_save_path = self.this_epoch_checkpoint_dir.joinpath("model.keras")
        self.model.save(model_save_path)
        return
    
    def generate_validation_samples(self):
        """
        Generates and saves a grid of validation images produced by the generator model, 
        along with their critic scores. The images are saved to the current epoch 
        checkpoint directory and a copy is saved to the model training output directory. 
        Additionally, a GIF of all saved images is generated every specified number of epochs.
        
        Parameters:
            None
        
        Returns:
            None
        """
        # Generate a set of images (number is determined by num_img defined in the initializer)
        generated_images = self.model.generator([self.random_latent_vectors, self.random_labels], training=False)
        # Rescale the images from [-1, 1] to [0, 255]
        generated_images = (generated_images * 127.5) + 127.5
        
        # Get critic scores for generated images using CPU
        critic_scores = self.model.critic([generated_images, self.random_labels], training=False)
        
        # Creating a grid of images
        fig, axes = plt.subplots(self.grid_size[0], self.grid_size[1], figsize=(self.grid_size[1] * 2, self.grid_size[0] * 2))
        
        for i in range(self.num_img):
            row = i // self.grid_size[1]
            col = i % self.grid_size[1]
            img = generated_images[i].numpy()
            img = keras.utils.array_to_img(img)
            axes[row, col].imshow(img)
            axes[row, col].axis('off')
            # get the critic score for the generated image as a float value
            score = critic_scores[i].numpy().item()
            label = self.random_labels[i].numpy().item()
            axes[row, col].set_title(f'Label: {label} Score: {score:.2f}', fontsize=6)
        
        # Add a title to the plot with the epoch and number of samples trained
        samples_trained = self.metrics_dataframe['samples_trained_on'].iloc[-1]
        fig.suptitle(f"Validation Samples - Epoch {self.current_epoch}, Samples Trained: {samples_trained:,}", fontsize=12)
        
        plt.tight_layout()
        
        # Save the grid of images to the current epoch checkpoint directory
        fig_save_path = self.this_epoch_checkpoint_dir.joinpath(f"validation_samples.png")
        plt.savefig(fig_save_path, dpi=200)
        # copy the plot to the model_training_output_dir for easy access to the most recent plot
        fig_copy_path = self.model_training_output_dir.joinpath(f"validation_samples.png")
        shutil.copy(fig_save_path, fig_copy_path)
        
        # Clear the current figure and close the plot to avoid memory leaks
        plt.clf()
        plt.close()
        return
    
    def generate_gif(self):
        """
        Creates a GIF of the validation images generated at the end of each epoch
        without loading all images into memory at once.
        """
        print("Generating GIF of validation samples...")
        
        # Get a sorted list of all validation_samples.png paths
        validation_samples_paths = sorted(
            self.model_checkpoints_dir.glob("epoch_*/validation_samples.png"),
            key=lambda p: int(p.parent.name.split("_")[1])
        )
        
        if not validation_samples_paths:
            print("No validation samples found. Skipping GIF generation.")
            return
        
        # Use PIL to handle GIF creation incrementally
        gif_save_path = self.model_training_output_dir.joinpath("validation_samples.gif")
        
        with Image.open(validation_samples_paths[0]) as first_frame:
            # Convert the first frame to RGB (if not already)
            first_frame = first_frame.convert("RGB")
            # Save the GIF incrementally by appending each frame
            first_frame.save(
                gif_save_path,
                save_all=True,
                append_images=[
                    Image.open(file_path).convert("RGB") for file_path in validation_samples_paths[1:]
                ],
                duration=125,  # duration in milliseconds (adjust fps as needed)
                loop=0  # loop forever
            )
        
        # copy the generated gif to the current epoch checkpoint directory
        gif_copy_path = self.this_epoch_checkpoint_dir.joinpath("validation_samples.gif")
        shutil.copy(gif_save_path, gif_copy_path)
        
        print(f"GIF saved to {gif_save_path}")
        return
    
    def plot_epoch_duration_and_estimate_train_time(self):
        """
        method to estimate the time to train for a number of epochs based on the average time spent training
        """
        # calculate the total time spent on this epoch (training and metric logging)
        total_iteration_time = self.epoch_train_duration.total_seconds() + self.metric_calc_duration.total_seconds()
        
        # update the last row of the metrics dataframe with the correct values for the time spent training and logging metrics
        self.metrics_dataframe.iloc[-1, self.metrics_dataframe.columns.get_loc("metric_calc_time")] = self.metric_calc_duration.total_seconds()
        self.metrics_dataframe.iloc[-1, self.metrics_dataframe.columns.get_loc("total_iteration_time")] = total_iteration_time
        
        # Save the metrics dataframe to a CSV file now that we have the time spent logging metrics
        csv_save_path = self.this_epoch_checkpoint_dir.joinpath("metrics_dataframe.csv")
        self.metrics_dataframe.to_csv(csv_save_path, index=False)
        
        ############################## create plots for training and metric calculation durations ##############################
        # Identify locations where model was loaded
        model_load_indices = self.metrics_dataframe.loc[self.metrics_dataframe['model_loaded'], 'epoch']
        
        # get the number of samples trained and epochs for the x-axis
        samples_trained = self.metrics_dataframe['samples_trained_on']
        epochs = self.metrics_dataframe['epoch']
        
        fig, ax1 = plt.subplots(figsize=(10, 6))
        # Plot the time metric starting after the first epoch
        ax1.bar(epochs[:], self.metrics_dataframe["epoch_train_time"], label="epoch_train_time".replace("_", " ").title(), zorder=1, 
                color='#3B9DE3', edgecolor='black')
        ax1.bar(epochs[:], self.metrics_dataframe["metric_calc_time"], label="metric_calc_time".replace("_", " ").title(), zorder=1,
                color='#AE7E6F', edgecolor='black', bottom=self.metrics_dataframe["epoch_train_time"])
        
        for i, model_load_epoch in enumerate(model_load_indices):
            if i == 0:
                line_label = "ckpt loaded"
            else:
                line_label = ""
            ax1.axvline(x=model_load_epoch, color='#5C5C5C', linestyle='--', alpha=0.7, linewidth=1.5, label=line_label, zorder=2)
            # add text to show the number of samples at this checkpoint
            y_text_position = ((ax1.get_ylim()[1] - ax1.get_ylim()[0]) * 0.02) + ax1.get_ylim()[0]
            ax1.text(model_load_epoch, y_text_position, f"{model_load_epoch:,}", color='#5C5C5C', fontsize=9, rotation=90, va='bottom', ha='right',
                    zorder=2)
        
        # add text to the plot to show the number of samples trained on at the end of the last epoch
        plt.text(0.86, 1.012, f"Samples Trained On: {samples_trained.iloc[-1]:,}", fontsize=9, ha='center', va='center', 
                transform=plt.gca().transAxes)
        
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Duration (s)")
        ax1.legend()
        ax1.grid(True, alpha=0.4)
        plt.title(f"Time Breakdown Across Epochs")
        plt.tight_layout()
        
        filename = f"time_breakdown_across_epochs.png"
        plot_path = self.model_training_output_dir.joinpath(filename)
        print(f"Saving plot to: {plot_path}")
        plt.savefig(plot_path, dpi=200)
        plt.close()
        plt.clf()
        
        # if this is the first epoch, skip estimating the time to train to specific epochs because we don't have enough data.
        if self.current_epoch == 1:
            return
        
        # get the time spent on every epoch and replace the first value with the second value
        iteration_times = self.metrics_dataframe["total_iteration_time"].copy()  # (time spent training + time spent logging metrics)
        iteration_times.iloc[0] = iteration_times.iloc[1]
        
        # get the average time spent on an epoch
        avg_iteration_time = iteration_times.mean()
        stdev_iteration_time = iteration_times.std()
        # write to a file the time it took to train to a milestone or the estimated time to train to a milestone
        with open(self.model_training_output_dir.joinpath("_milestones_and_ETAs.txt"), "w") as file:
            file.write("Epoch Milestones and Estimated Training Times\n")
            file.write(f"{'='*72}\n")
            # convert to a readable format (HHH:MM:SS.ss)
            avg_time = get_readable_time_string(avg_iteration_time)
            stdev_time = get_readable_time_string(stdev_iteration_time)
            file.write(f"Average Epoch Duration (HHH:MM:SS.ss): {avg_time}\n")
            file.write(f"Standard Deviation (HHH:MM:SS.ss): {stdev_time}\n")
            file.write(f"NOTE: this time includes the time spent training and logging metrics.\n")
            """
            NOTE: the average and standard deviations are calculated with the second 
                epoch's time replacing the first epoch's time to not overstate the 
                time spent on the first epoch where a long time is spent to setup 
                the computation graph of the model.
            """
            file.write(f"NOTE: the average and standard deviations are calculated with the second\n" \
                    f"\tepoch's time replacing the first epoch's time to not overstate the\n" \
                    f"\ttime spent on the first epoch where a long time is spent to setup\n" \
                    f"\tthe computation graph of the model.\n")
            file.write(f"{'='*72}\n")
            epoch_milestones = [5, 10, 20, 40, 50, 60, 80, 100, 150, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 5000, 10000]
            for milestone in epoch_milestones:
                # if we have already passed the milestone, calculate the time it took
                if self.current_epoch > milestone:
                    # add up the iteration times up to the milestone
                    total_time_seconds = iteration_times[:milestone].sum()
                    # convert to a readable format (HHH:MM:SS.ss)
                    total_time = get_readable_time_string(total_time_seconds)
                    # write the time to the file
                    file.write(f"Epoch {milestone: <5} - Time to Train (HHH:MM:SS.ss): {total_time}\n")
                else:  # we haven't reached that milestone yet
                    total_time_seconds = avg_iteration_time * milestone
                    # convert to a readable format (HHH:MM:SS.ss)
                    total_time = get_readable_time_string(total_time_seconds)
                    # write the estimated time to the file
                    file.write(f"Epoch {milestone: <5} - Estimated Time to Train (HHH:MM:SS.ss): {total_time}\n")
            file.write(f"{'='*72}\n")
        return
