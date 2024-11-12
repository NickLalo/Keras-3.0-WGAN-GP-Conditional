"""
model definitions and callbacks
"""


import os
import time
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
    
    d_model = keras.models.Model(img_input, x, name="discriminator")
    return d_model


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
    
    g_model = keras.models.Model(noise, x, name="generator")
    return g_model


class WGAN_GP(keras.Model):
    """
    WGAN-GP model
    """
    def __init__(
        self,
        discriminator,
        generator,
        latent_dim,
        discriminator_extra_steps=3,
        gp_weight=10.0,
    ):
        super().__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        # in a WGAN-GP, the discriminator trains for a number of steps and then generator trains for one step
        self.num_disc_steps = discriminator_extra_steps
        self.gp_weight = gp_weight
        return
    
    def compile(self, disc_optimizer, gen_optimizer, disc_loss_fn, gen_loss_fn):
        super().compile()
        self.disc_optimizer = disc_optimizer
        self.gen_optimizer = gen_optimizer
        self.disc_loss_fn = disc_loss_fn
        self.gen_loss_fn = gen_loss_fn
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
            self.disc_optimizer.apply_gradients(
                zip(disc_gradient, self.discriminator.trainable_variables)
            )
        
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
        self.gen_optimizer.apply_gradients(
            zip(gen_gradient, self.generator.trainable_variables)
        )
        
        # # Calculate the average scores for real and fake images for this batch
        avg_real_score = tf.reduce_mean(discriminator_real_scores)
        avg_fake_score = tf.reduce_mean(discriminator_fake_scores)
        avg_gradient_penalty = tf.reduce_mean(gradient_penalties)
        
        #  "disc_loss_real", "disc_loss_fake",
        return {"disc_loss": disc_loss, "gen_loss": gen_loss, "disc_loss_real": avg_real_score, "disc_loss_fake": avg_fake_score, "disc_loss_gp": avg_gradient_penalty}


class GANMonitor(keras.callbacks.Callback):
    def __init__(self, num_img, latent_dim, grid_size):
        self.num_img = num_img
        self.latent_dim = latent_dim
        self.grid_size = grid_size  # Grid size as a tuple (rows, cols)
        self.random_latent_vectors = tf.random.normal(shape=(self.num_img, self.latent_dim))
        return
    
    def on_epoch_end(self, epoch, logs=None):
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
        plt.savefig(f"data/generated_grid_epoch_{epoch:04d}.png", dpi=200)
        # Clear the current figure and close the plot to avoid memory leaks
        plt.clf()
        plt.close()
        
        # every 5 epochs, generate a GIF of all the saved images
        if epoch % 5 == 0:
            self.generate_gif()
        return
    
    def on_train_end(self, logs=None):
        # Generate a GIF of all the saved images
        self.generate_gif()
        return
    
    def generate_gif(self):
        print(f"Generating GIF from images saved in data/generated_grid_epoch_*.png")
        start_time = time.time()
        # Create a GIF from saved images
        images = []
        for file_name in sorted(glob.glob("data/generated_grid_epoch_*.png")):
            images.append(imageio.imread(file_name))
        imageio.mimsave("data/generated_images.gif", images, fps=8)  # fps was set to 2 when we only evaluated 20 epochs
        
        end_time = time.time()
        time_hours = int((end_time - start_time) // 3600)
        time_minutes = int(((end_time - start_time) % 3600) // 60)
        time_seconds = int(((end_time - start_time) % 3600) % 60)
        print(f"Total time to generate GIF: {time_hours:02d}:{time_minutes:02d}:{time_seconds:02d}")
        print(f"Generated GIF from images saved in data/generated_images.gif")
        return


class LossLogger(tf.keras.callbacks.Callback):
    """
    Custom callback to log metrics and save loss plots at the end of every epoch
    """
    def __init__(self, samples_per_epoch=0):
        """
        samples_per_epoch: int, the number of samples in the training dataset
        """
        self.samples_per_epoch = samples_per_epoch
        # Initialize the loss dataframe to track metrics
        self.loss_df = pd.DataFrame(columns=[
            "epoch", "disc_loss", "disc_loss_real", "disc_loss_fake", "disc_loss_gp", "gen_loss", "samples_trained_on", "model_loaded"
        ])
        return
    
    def on_epoch_end(self, epoch, logs=None):
        # extract info from logs dictionary
        disc_loss = logs["disc_loss"]
        disc_loss_real = logs["disc_loss_real"]
        disc_loss_fake = logs["disc_loss_fake"]
        disc_loss_gp = logs["disc_loss_gp"]
        gen_loss = logs["gen_loss"]
        
        # print the metrics
        print(f"/nEpoch: {epoch}")
        print(f"Discriminator Loss: {disc_loss:.4f}")
        print(f"Discriminator Loss (Real): {disc_loss_real:.4f}")
        print(f"Discriminator Loss (Fake): {disc_loss_fake:.4f}")
        print(f"Discriminator Loss (GP): {disc_loss_gp:.4f}")
        print(f"Generator Loss: {gen_loss:.4f}")
        
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
            "model_loaded": False
        }])
        
        # concatenate the new row to the existing loss dataframe
        self.loss_df = pd.concat([self.loss_df, new_row], ignore_index=True)
        
        # Save the losses dataframe to a CSV file
        self.loss_df.to_csv("loss_plots/loss.csv", index=False)
        
        ######################### PLOT LOSSES #########################
        # Ensure save directory exists
        loss_plot_save_dir = "loss_plots"
        os.makedirs(loss_plot_save_dir, exist_ok=True)
        
        #################################################################################################
        
        # Define list of individual loss columns to plot
        loss_columns = ["disc_loss", "disc_loss_real", "disc_loss_fake", "disc_loss_gp", "gen_loss"]
        
        # Identify locations where model was loaded
        model_load_indices = self.loss_df.loc[self.loss_df['model_loaded'], 'samples_trained_on']
        
        # Loop through each loss metric to create individual plots
        for col in loss_columns:
            # Create the plot
            plt.figure(figsize=(10, 6))
            plt.plot(self.loss_df['samples_trained_on'], self.loss_df[col], label=col.replace("_", " ").title(), zorder=1)
            
            # Add vertical lines for model loading events with a single label for legend
            for i, idx in enumerate(model_load_indices):
                # add a label for only the first line to avoid multiple copies in the legend
                if i == 0:
                    line_label = "First checkpoint after loading"
                else:
                    line_label = ""
                plt.axvline(x=idx, color='#5C5C5C', linestyle='--', alpha=0.7, linewidth=1.5, label=line_label, zorder=-1)
                # add text to show the number of samples at this checkpoint
                y_text_position = self.loss_df[col].max() * 0.85
                plt.text(idx, y_text_position, f"{idx:,}", color='#5C5C5C', fontsize=6, rotation=90, va='bottom', ha='right', zorder=2)
            
            plt.xlabel("Samples Trained On")
            plt.ylabel(col.replace("_", " ").title())
            plt.title(f"{col.replace('_', ' ').title()} vs. Samples Trained On")
            plt.legend()
            plt.grid(True, alpha=0.4)
            
            plt.tight_layout()
            
            # reorder the wording of the column name for the filename so that loss comes first.  This makes all the files appear together in the directory
            col_for_filename = col.replace("_loss", "")
            col_for_filename = f"loss_{col_for_filename}"
            
            # Define file path for each plot
            plot_path = os.path.join(loss_plot_save_dir, f"{col_for_filename}_vs_samples.png")
            
            # Save plot to the specified directory with specified dpi
            plt.savefig(plot_path, dpi=200)
            print(f"Plot saved to: {plot_path}")
            
            # Clear the plot to free memory and avoid display
            plt.clf()
            plt.close()
        
        # Combined plot for specific losses
        combined_loss_columns = ["disc_loss", "disc_loss_real", "disc_loss_fake", "gen_loss"]
        
        plt.figure(figsize=(10, 6))
        for col in combined_loss_columns:
            plt.plot(self.loss_df['samples_trained_on'], self.loss_df[col], label=col.replace("_", " ").title(), zorder=1)
        
        # Add vertical lines for model loading events with a single label for legend
        for i, idx in enumerate(model_load_indices):
            # add a label for only the first line to avoid multiple copies in the legend
            if i == 0:
                line_label = "First checkpoint after loading"
            else:
                line_label = ""
            plt.axvline(x=idx, color='#5C5C5C', linestyle='--', alpha=0.7, linewidth=1.5, label=line_label, zorder=-1)
            # add text to show the number of samples at this checkpoint
            y_text_position = self.loss_df[col].max() * 0.85
            plt.text(idx, y_text_position, f"{idx:,}", color='#5C5C5C', fontsize=6, rotation=90, va='bottom', ha='right', zorder=2)
        
        plt.xlabel("Samples Trained On")
        plt.ylabel("Loss Value")
        plt.title("All Losses vs. Samples Trained On")
        plt.legend()
        plt.grid(True, alpha=0.4)
        
        plt.tight_layout()
        
        # Save the combined plot
        combined_plot_path = os.path.join(loss_plot_save_dir, "losses_combined_vs_samples.png")
        plt.savefig(combined_plot_path, dpi=200)
        print(f"Combined loss plot saved to: {combined_plot_path}")
        
        # Clear the plot to free memory
        plt.clf()
        plt.close()
        return
