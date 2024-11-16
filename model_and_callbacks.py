"""
model definitions and callbacks
"""


import os
import time
import shutil
from pathlib import Path
import json
import numpy as np
import pandas as pd
import keras
import tensorflow as tf
from keras import layers
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")  # set to use the "Agg" to avoid tkinter error
import imageio
import glob


# Clear all previously registered custom objects
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
    x = layers.Conv2D(
        filters, kernel_size, strides=strides, padding=padding, use_bias=use_bias
    )(x)
    if use_bn:
        x = layers.BatchNormalization()(x)
    x = activation(x)
    if use_dropout:
        x = layers.Dropout(drop_value)(x)
    return x


@keras.saving.register_keras_serializable(package="generator_model", name="generator_model")
def get_discriminator_model(img_shape):
    img_input = layers.Input(shape=img_shape)
    # Zero pad the input to make the input images size to (32, 32, 1).
    x = layers.ZeroPadding2D((2, 2))(img_input)
    x = conv_block(
        x,
        64,
        kernel_size=(5, 5),
        strides=(2, 2),
        use_bn=False,
        use_bias=True,
        activation=layers.LeakyReLU(0.2),
        use_dropout=False,
        drop_value=0.3,
    )
    x = conv_block(
        x,
        128,
        kernel_size=(5, 5),
        strides=(2, 2),
        use_bn=False,
        activation=layers.LeakyReLU(0.2),
        use_bias=True,
        use_dropout=True,
        drop_value=0.3,
    )
    x = conv_block(
        x,
        256,
        kernel_size=(5, 5),
        strides=(2, 2),
        use_bn=False,
        activation=layers.LeakyReLU(0.2),
        use_bias=True,
        use_dropout=True,
        drop_value=0.3,
    )
    x = conv_block(
        x,
        512,
        kernel_size=(5, 5),
        strides=(2, 2),
        use_bn=False,
        activation=layers.LeakyReLU(0.2),
        use_bias=True,
        use_dropout=False,
        drop_value=0.3,
    )
    
    x = layers.Flatten()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(1)(x)
    
    disc_model = keras.models.Model(img_input, x, name="discriminator")
    return disc_model


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
def get_generator_model(noise_dim):
    noise = layers.Input(shape=(noise_dim,))
    x = layers.Dense(4 * 4 * 256, use_bias=False)(noise)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    
    x = layers.Reshape((4, 4, 256))(x)
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
        64,
        layers.LeakyReLU(0.2),
        strides=(1, 1),
        use_bias=False,
        use_bn=True,
        padding="same",
        use_dropout=False,
    )
    x = upsample_block(
        x, 1, layers.Activation("tanh"), strides=(1, 1), use_bias=False, use_bn=True
    )
    # At this point, we have an output which has the same shape as the input, (32, 32, 1).
    # We will use a Cropping2D layer to make it (28, 28, 1).
    x = layers.Cropping2D((2, 2))(x)
    
    gen_model = keras.models.Model(noise, x, name="generator")
    return gen_model


@keras.saving.register_keras_serializable(package="discriminator_loss", name="discriminator_loss")
def discriminator_loss(real_img, fake_img):
    # Define the loss functions for the discriminator,
    # which should be (fake_loss - real_loss).
    # We will add the gradient penalty later to this loss function.
    real_loss = tf.reduce_mean(real_img)
    fake_loss = tf.reduce_mean(fake_img)
    return fake_loss - real_loss


@keras.saving.register_keras_serializable(package="generator_loss", name="generator_loss")
def generator_loss(fake_img):
    # Define the loss functions for the generator.
    return -tf.reduce_mean(fake_img)


@keras.saving.register_keras_serializable(package="discriminator_optimizer", name="discriminator_optimizer")
def get_discriminator_optimizer():
    discriminator_optimizer = keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9)
    return discriminator_optimizer

@keras.saving.register_keras_serializable(package="generator_optimizer", name="generator_optimizer")
def get_generator_optimizer():
    generator_optimizer = keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9)
    return generator_optimizer

class WGAN_GP(keras.Model):
    """
    WGAN-GP model
    """
    def __init__(
        self,
        discriminator,
        generator,
        latent_dim,
        disc_optimizer,
        gen_optimizer,
        disc_loss_fn,
        gen_loss_fn,
        discriminator_extra_steps=3,
        gp_weight=10.0,
        # TODO: adding some code to allow reloading to pass relevant parameters to the model
        **kwargs
    ):
        super().__init__(**kwargs)
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        # in a WGAN-GP, the discriminator trains for a number of steps and then generator trains for one step
        self.num_disc_steps = discriminator_extra_steps
        self.gp_weight = gp_weight
        
        # REVIEW: moving these here so I can see if this is the right place to put them
        self.disc_optimizer = disc_optimizer
        self.gen_optimizer = gen_optimizer
        self.disc_loss_fn = disc_loss_fn
        self.gen_loss_fn = gen_loss_fn
        
        # set dummy value for self.optimizer avoid errors at the end of .fit() call
        self.optimizer = None
        
        return
    
    # def compile(self, disc_optimizer, gen_optimizer, disc_loss_fn, gen_loss_fn):
    def compile(self):
        super().compile()
        # REVIEW: should I do something with these for saving and loading?
        # REVIEW: I could move the saving of these attributes to the __init__, but I don't know how that would effect the .compile()??
        # REVIEW: ask chatGPT: do the setting of these attributes have to be done after the super().compile()?
        # self.disc_optimizer = disc_optimizer
        # self.gen_optimizer = gen_optimizer
        # self.disc_loss_fn = disc_loss_fn
        # self.gen_loss_fn = gen_loss_fn
        return
    
    def gradient_penalty(self, batch_size, real_images, fake_images):
        """Calculates the gradient penalty.
        
        This loss is calculated on an interpolated image
        and added to the discriminator loss.
        """
        # Get the interpolated image
        alpha = tf.random.uniform([batch_size, 1, 1, 1], 0.0, 1.0)
        diff = fake_images - real_images
        interpolated = real_images + alpha * diff
        
        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            # 1. Get the discriminator output for this interpolated image.
            pred = self.discriminator(interpolated, training=True)
            # 2. Calculate the gradients w.r.t to this interpolated image.
            grads = gp_tape.gradient(pred, [interpolated])[0]
            # 3. Calculate the norm of the gradients.
            norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
            gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp
    
    def train_step(self, real_images):
        if isinstance(real_images, tuple):
            real_images = real_images[0]
        # Get the batch size from the data as sometimes the last batch can be smaller
        batch_size = tf.shape(real_images)[0]
        
        # For each batch, we are going to perform the
        # following steps as laid out in the original paper:
        # 1. Train the generator and get the generator loss
        # 2. Train the discriminator and get the discriminator loss
        # 3. Calculate the gradient penalty
        # 4. Multiply this gradient penalty with a constant weight factor
        # 5. Add the gradient penalty to the discriminator loss
        # 6. Return the generator and discriminator losses as a loss dictionary
        
        # # Initialize lists to track discriminator scores on real and fake images
        discriminator_real_scores = []
        discriminator_fake_scores = []  # from the generated images
        gradient_penalties = []
        
        # Train the discriminator first. The original paper recommends training
        # the discriminator for `x` more steps (typically 5) as compared to
        # one step of the generator. Here we will train it for 3 extra steps
        # as compared to 5 to reduce the training time.
        for i in range(self.num_disc_steps):
            # Get a batch of random latent vectors
            random_latent_vectors = tf.random.normal(
                shape=(batch_size, self.latent_dim)
            )
            
            with tf.GradientTape() as tape:
                # Generate fake images from the latent vector
                fake_images = self.generator(random_latent_vectors, training=True)
                # Get the discriminator scores (logits) for the fake images
                fake_logits = self.discriminator(fake_images, training=True)
                # Get the discriminator scores (logits) for the real images
                real_logits = self.discriminator(real_images, training=True)
                
                # # Append the mean scores to track them
                discriminator_real_scores.append(tf.reduce_mean(real_logits))
                discriminator_fake_scores.append(tf.reduce_mean(fake_logits))
                
                # Calculate the discriminator loss using the fake and real image logits
                disc_cost = self.disc_loss_fn(real_img=real_logits, fake_img=fake_logits)
                # Calculate the gradient penalty
                gp = self.gradient_penalty(batch_size, real_images, fake_images)
                # Add the gradient penalty to the discriminator loss
                disc_loss = disc_cost + gp * self.gp_weight
                
                # Append the gradient penalty to track it
                gradient_penalties.append(gp)
            
            # Get the gradients for the discriminator loss
            disc_gradient = tape.gradient(disc_loss, self.discriminator.trainable_variables)
            # Update the discriminator's weights
            self.disc_optimizer.apply_gradients(zip(disc_gradient, self.discriminator.trainable_variables))

        
        # Train the generator
        # Generate a new batch of random latent vectors
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        
        with tf.GradientTape() as tape:
            # Generate fake images with the generator
            generated_images = self.generator(random_latent_vectors, training=True)
            # Get the discriminator scores (logits) for the generated images
            gen_img_logits = self.discriminator(generated_images, training=True)
            # Calculate the generator loss
            gen_loss = self.gen_loss_fn(gen_img_logits)
        
        # Get the gradients for the generator loss
        gen_gradient = tape.gradient(gen_loss, self.generator.trainable_variables)
        # Update the generator's weights
        self.gen_optimizer.apply_gradients(zip(gen_gradient, self.generator.trainable_variables))
        
        # # Calculate the average scores for real and fake images for this batch
        avg_real_score = tf.reduce_mean(discriminator_real_scores)
        avg_fake_score = tf.reduce_mean(discriminator_fake_scores)
        avg_gradient_penalty = tf.reduce_mean(gradient_penalties)
        
        #  "disc_loss_real", "disc_loss_fake",
        return {"disc_loss": disc_loss, "gen_loss": gen_loss, "disc_loss_real": avg_real_score, "disc_loss_fake": avg_fake_score, "disc_loss_gp": avg_gradient_penalty}
    
    def get_config(self):
        # get the default configuration for the model
        base_config = super().get_config()
        
        # get extra configurations for custom model attributes (not Python objects like ints, strings, etc.)
        custom_config = {
            "discriminator": keras.saving.serialize_keras_object(self.discriminator),
            "generator": keras.saving.serialize_keras_object(self.generator),
            "latent_dim": self.latent_dim,
            "discriminator_extra_steps": self.num_disc_steps,
            "gp_weight": self.gp_weight,
            "disc_optimizer": keras.saving.serialize_keras_object(self.disc_optimizer),
            "gen_optimizer": keras.saving.serialize_keras_object(self.gen_optimizer),
            "disc_loss_fn": keras.saving.serialize_keras_object(self.disc_loss_fn),
            "gen_loss_fn": keras.saving.serialize_keras_object(self.gen_loss_fn),
        }
        
        # combine the two configurations
        config = {**base_config, **custom_config}
        return config
    
    @classmethod
    def from_config(cls, config, custom_objects=None):
        # get the discriminator and generator models from the config
        discriminator = keras.saving.deserialize_keras_object(config.pop("discriminator"), custom_objects=custom_objects)
        generator = keras.saving.deserialize_keras_object(config.pop("generator"), custom_objects=custom_objects)
        latent_dim = config.pop("latent_dim")
        discriminator_extra_steps = config.pop("discriminator_extra_steps")
        gp_weight = config.pop("gp_weight")
        # REVIEW: if this is the line that is messing up, I can put the generation of the optimizer in it's own custom function and try again
        disc_optimizer = keras.saving.deserialize_keras_object(config.pop("disc_optimizer"), custom_objects=custom_objects)
        gen_optimizer = keras.saving.deserialize_keras_object(config.pop("gen_optimizer"), custom_objects=custom_objects)
        disc_loss_fn = keras.saving.deserialize_keras_object(config.pop("disc_loss_fn"), custom_objects=custom_objects)
        gen_loss_fn = keras.saving.deserialize_keras_object(config.pop("gen_loss_fn"), custom_objects=custom_objects)
        
        # create a new instance of the model with the discriminator and generator models (This calls the __init__ method)
        model = cls(discriminator=discriminator, generator=generator, latent_dim=latent_dim, discriminator_extra_steps=discriminator_extra_steps, 
                    gp_weight=gp_weight, disc_optimizer=disc_optimizer, gen_optimizer=gen_optimizer, disc_loss_fn=disc_loss_fn, gen_loss_fn=gen_loss_fn)
        return model


class Training_Monitor(tf.keras.callbacks.Callback):
    """
    # TODO: write a better docstring
    Custom callback to log metrics and save loss plots at the end of every epoch
    """
    def __init__(self, 
                model_training_output_dir, 
                model_checkpoints_dir, 
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
        
        self.num_img = num_img
        self.latent_dim = latent_dim
        self.grid_size = grid_size  # Grid size as a tuple (rows, cols)
        self.random_latent_vectors = tf.random.normal(shape=(self.num_img, self.latent_dim))
        
        self.samples_per_epoch = samples_per_epoch
        self.gif_creation_frequency = 5  # create a GIF every 5 epochs
        
        # load the loss metrics csv to a dataframe if the last_checkpoint_dir_path is provided (we are reloading the model)
        if last_checkpoint_dir_path is not None:
            # get the path to the loss_metrics.csv file
            loss_metrics_path = last_checkpoint_dir_path.joinpath("loss_metrics.csv")
            # load the csv to a dataframe
            self.loss_metrics_dataframe = pd.read_csv(loss_metrics_path)
            # remember that the model was loaded for logging loss metrics at the end of the next epoch
            self.model_recently_loaded = True
        else:  # create a new dataframe for tracking loss metrics for a fresh start
            self.loss_metrics_dataframe = pd.DataFrame(columns=[
                "epoch", "disc_loss", "disc_loss_real", "disc_loss_fake", "disc_loss_gp", "gen_loss", "samples_trained_on", "model_loaded"
            ])
            self.model_recently_loaded = False
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
        # Set the current epoch checkpoint directory
        self.set_this_epoch_checkpoint_dir()
        
        # Log the loss metrics to the DataFrame
        self.log_loss_to_dataframe(logs)
        
        # Plot individual loss metrics
        self.create_loss_plots()
        
        # Generate validation samples
        self.generate_validation_samples()
        
        # save a copy of the model to the current epoch checkpoint directory
        model_save_path = self.this_epoch_checkpoint_dir.joinpath("model.keras")
        self.model.save(model_save_path)
        return
    
    def on_train_end(self, logs=None):
        # Generate a GIF of all the saved images
        self.generate_gif()
        return
    
    def set_this_epoch_checkpoint_dir(self):
        # create a new checkpoint directory for logging info at the end of this epoch
        epoch = len(self.loss_metrics_dataframe) + 1
        self.this_epoch_checkpoint_dir = self.model_checkpoints_dir.joinpath(f"epoch_{epoch:04d}")
        # error out if this directory already exists
        if os.path.exists(self.this_epoch_checkpoint_dir):
            error_string = (
                f"Checkpoint directory already exists: {self.this_epoch_checkpoint_dir}. Please ensure that any checkpoint past the intended loading"
                " point is deleted before continuing."
            )
            raise ValueError(error_string)
        os.makedirs(self.this_epoch_checkpoint_dir, exist_ok=True)
        return
    
    def log_loss_to_dataframe(self, logs):
        """
        Logs the loss metrics to a DataFrame and saves it to a CSV file.
        This method extracts the discriminator and generator loss values from the provided logs dictionary,
        prints them, and appends them to an internal DataFrame. The DataFrame is then saved to a CSV file.
        Parameters:
            logs (dict): A dictionary containing the loss values with the following keys:
                - "disc_loss": Discriminator total loss
                - "disc_loss_real": Discriminator loss on real samples
                - "disc_loss_fake": Discriminator loss on fake samples
                - "disc_loss_gp": Discriminator gradient penalty loss
                - "gen_loss": Generator loss
        Returns:
            None
        """
        # extract info from logs dictionary
        disc_loss = float(logs["disc_loss"])
        disc_loss_real = float(logs["disc_loss_real"])
        disc_loss_fake = float(logs["disc_loss_fake"])
        disc_loss_gp = float(logs["disc_loss_gp"])
        gen_loss = float(logs["gen_loss"])
        # start epoch count from 1 as we are saying: "These are metrics at the end of one epoch" for the first epoch
        epoch = len(self.loss_metrics_dataframe) + 1
        
        # create a new DataFrame row with the current epoch data
        new_row = pd.DataFrame([{
            "epoch": epoch,
            "disc_loss": disc_loss,
            "disc_loss_real": disc_loss_real,
            "disc_loss_fake": disc_loss_fake,
            "disc_loss_gp": disc_loss_gp,
            "gen_loss": gen_loss,
            # calculate the number of samples trained on at the end of this epoch
            "samples_trained_on": epoch * self.samples_per_epoch,
            "model_loaded": self.model_recently_loaded
        }])
        
        # if the model was loaded, set the model_loaded column to False for the next epoch
        self.model_recently_loaded = False  # we only want to mark the first epoch after loading
        
        # concatenate the new row to the existing loss dataframe
        self.loss_metrics_dataframe = pd.concat([self.loss_metrics_dataframe, new_row], ignore_index=True)
        
        # Save the losses dataframe to a CSV file
        csv_save_path = self.this_epoch_checkpoint_dir.joinpath("loss_metrics.csv")
        self.loss_metrics_dataframe.to_csv(csv_save_path, index=False)
        return
    
    def create_loss_plots(self):
        """
        Generates and saves individual and combined loss plots for the training process. This method creates plots for various loss metrics across
        epochs. It also marks the points where the model was loaded from a checkpoint. The plots are saved to the specified directory.
        
        Loss metrics plotted:
        - disc_loss
        - disc_loss_real
        - disc_loss_fake
        - disc_loss_gp
        - gen_loss
        
        The method also creates a combined plot for the following loss metrics:
        - disc_loss
        - disc_loss_real
        - disc_loss_fake
        - gen_loss
        
        Parameters:
            None
        Returns:
            None
        
        # TODO: Somehow add to the plot the number of samples trained on at to the plot. Can I put this on the upper x-axis?
        """
        # Define list of individual loss columns to plot
        loss_columns = ["disc_loss", "disc_loss_real", "disc_loss_fake", "disc_loss_gp", "gen_loss"]
        
        # Identify locations where model was loaded
        model_load_indices = self.loss_metrics_dataframe.loc[self.loss_metrics_dataframe['model_loaded'], 'epoch']
        
        # Loop through each loss metric to create individual plots
        for col in loss_columns:
            # Create the plot
            plt.figure(figsize=(10, 6))
            plt.plot(self.loss_metrics_dataframe['epoch'], self.loss_metrics_dataframe[col], label=col.replace("_", " ").title(), zorder=1)
            
            # Add vertical lines for model loading events with a single label for legend
            for i, model_load_epoch in enumerate(model_load_indices):
                # add a label for only the first line to avoid multiple copies in the legend
                if i == 0:
                    line_label = "First checkpoint after loading"
                else:
                    line_label = ""
                plt.axvline(x=model_load_epoch, color='#5C5C5C', linestyle='--', alpha=0.7, linewidth=1.5, label=line_label, zorder=-1)
                # add text to show the number of samples at this checkpoint
                y_text_position = plt.ylim()[1] * 0.9
                plt.text(model_load_epoch, y_text_position, f"{model_load_epoch:,}", color='#5C5C5C', fontsize=8, rotation=90, va='bottom', ha='right', zorder=2)
            
            plt.xlabel("Epoch")
            plt.ylabel(col.replace("_", " ").title())
            plt.title(f"{col.replace('_', ' ').title()} vs. Epoch")
            plt.legend()
            plt.grid(True, alpha=0.4)
            plt.tight_layout()
            
            # reorder the wording of the column name for the filename so that loss comes first. This groups the saved plots togethers
            col_for_filename = col.replace("_loss", "")
            col_for_filename = f"loss_{col_for_filename}"
            
            # Save plot to the current epoch checkpoint directory
            plot_path = self.this_epoch_checkpoint_dir.joinpath(f"{col_for_filename}.png")
            plt.savefig(plot_path, dpi=200)
            # copy the plot to the model_training_output_dir for easy access to the most recent plot
            plot_copy_path = self.model_training_output_dir.joinpath(f"{col_for_filename}.png")
            shutil.copy(plot_path, plot_copy_path)
            
            # Clear the plot to free memory and avoid display
            plt.clf()
            plt.close()
        
        # Combined plot for specific losses
        combined_loss_columns = ["disc_loss", "disc_loss_real", "disc_loss_fake", "gen_loss"]
        
        plt.figure(figsize=(10, 6))
        for col in combined_loss_columns:
            plt.plot(self.loss_metrics_dataframe['epoch'], self.loss_metrics_dataframe[col], label=col.replace("_", " ").title(), zorder=1)
        
        # Add vertical lines for model loading events with a single label for legend
        for i, model_load_epoch in enumerate(model_load_indices):
            # add a label for only the first line to avoid multiple copies in the legend
            if i == 0:
                line_label = "First checkpoint after loading"
            else:
                line_label = ""
            plt.axvline(x=model_load_epoch, color='#5C5C5C', linestyle='--', alpha=0.7, linewidth=1.5, label=line_label, zorder=-1)
            # add text to show the number of samples at this checkpoint
            y_text_position = plt.ylim()[1] * 0.9
            plt.text(model_load_epoch, y_text_position, f"{model_load_epoch:,}", color='#5C5C5C', fontsize=8, rotation=90, va='bottom', ha='right', zorder=2)
        
        plt.xlabel("Epoch")
        plt.ylabel("Loss Value")
        plt.title("All Losses vs. Epoch")
        plt.legend()
        plt.grid(True, alpha=0.4)
        plt.tight_layout()
        
        # Save the combined plot to the current epoch checkpoint directory
        combined_plot_path = self.this_epoch_checkpoint_dir.joinpath("losses_combined.png")
        plt.savefig(combined_plot_path, dpi=200)
        print(f"\nCombined loss plot saved to: {combined_plot_path}")
        # copy the plot to the model_training_output_dir for easy access to the most recent plot
        combined_plot_copy_path = self.model_training_output_dir.joinpath("losses_combined.png")
        shutil.copy(combined_plot_path, combined_plot_copy_path)
        
        # Clear the plot to free memory
        plt.clf()
        plt.close()
        return
    
    def generate_validation_samples(self):
        """
        TODO: code that came with the tutorial that I plan to overwrite 
            - Add a title to the generated plot with epoch and number of samples trained
        """
        # Generate a set of images (number is determined by num_img defined in the initializer)
        generated_images = self.model.generator(self.random_latent_vectors, training=False)
        # Rescale the images from [-1, 1] to [0, 255]
        generated_images = (generated_images * 127.5) + 127.5
        
        # Get discriminator scores for generated images using CPU
        discriminator_scores = self.model.discriminator(generated_images, training=False)
        
        # Creating a grid of images
        fig, axes = plt.subplots(self.grid_size[0], self.grid_size[1], figsize=(self.grid_size[1] * 2, self.grid_size[0] * 2))
        
        for i in range(self.num_img):
            row = i // self.grid_size[1]
            col = i % self.grid_size[1]
            img = generated_images[i].numpy()
            img = keras.utils.array_to_img(img)
            axes[row, col].imshow(img)
            axes[row, col].axis('off')
            # get the discriminator score for the generated image as a float value
            score = discriminator_scores[i].numpy().item()
            axes[row, col].set_title(f'Score: {score:.2f}', fontsize=8)
        
        plt.tight_layout()
        # get epoch from the number of rows in the loss dataframe
        epoch = len(self.loss_metrics_dataframe)
        
        # Save the grid of images to the current epoch checkpoint directory
        fig_save_path = self.this_epoch_checkpoint_dir.joinpath(f"validation_samples.png")
        plt.savefig(fig_save_path, dpi=200)
        # copy the plot to the model_training_output_dir for easy access to the most recent plot
        fig_copy_path = self.model_training_output_dir.joinpath(f"validation_samples.png")
        shutil.copy(fig_save_path, fig_copy_path)
        
        # Clear the current figure and close the plot to avoid memory leaks
        plt.clf()
        plt.close()
        
        # every X number of epochs, generate a GIF of all the saved images
        if epoch % self.gif_creation_frequency == 0 and epoch > 0:
            self.generate_gif()
        return
    
    def generate_gif(self):
        """
        creates a gif of the validation images generated at the end of each epoch
        """
        # Get a sorted list of all validation_samples.png paths
        validation_samples_paths = sorted(
            self.model_checkpoints_dir.glob("epoch_*/validation_samples.png"),
            key=lambda p: int(p.parent.name.split("_")[1])
        )
        
        # create the list of images to be used in the gif
        images = []
        for file_path in validation_samples_paths:
            images.append(imageio.imread(file_path))
        
        # save the images as a gif to the model_training_output_dir
        gif_save_path = self.model_training_output_dir.joinpath("validation_samples.gif")
        imageio.mimsave(gif_save_path, images, fps=8)
        return
