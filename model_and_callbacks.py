"""
model definitions and callbacks

Implementation modified from following this tutorial:
https://keras.io/examples/generative/wgan_gp/
"""


import os
import shutil
import pandas as pd
import keras
import tensorflow as tf
from keras import layers
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")  # set to use the "Agg" to avoid tkinter error
from PIL import Image
import warnings

from utils import get_memory_usage, get_gpu_memory_usage


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


@keras.saving.register_keras_serializable(package="custom_wgan_gp", name="custom_wgan_gp")
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
        discriminator_input_shape=None,
        gp_weight=10.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        # in a WGAN-GP, the discriminator trains for a number of steps and then generator trains for one step
        self.num_disc_steps = discriminator_extra_steps
        self.disc_input_shape = discriminator_input_shape
        self.gp_weight = gp_weight
        
        # set dummy value for self.optimizer avoid errors at the end of .fit() call
        self.optimizer = None
        
        # build the model
        self.build()
        return
    
    # def compile(self, disc_optimizer, gen_optimizer, disc_loss_fn, gen_loss_fn):
    def compile(self, disc_optimizer=None, gen_optimizer=None, disc_loss_fn=None, gen_loss_fn=None):
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
    
    def train_step(self, batch_data):
        # unpack the batch data into images and labels
        real_images, real_labels = batch_data
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
        
        return_dict = {
            "disc_loss": disc_loss,
            "gen_loss": gen_loss,
            "disc_loss_real": avg_real_score,
            "disc_loss_fake": avg_fake_score,
            "disc_loss_gp": avg_gradient_penalty
            }
        
        return return_dict
    
    def build(self, input_shape=None):
        # Explicitly build the generator and discriminator models
        self.generator.build(input_shape=(None, self.latent_dim))
        self.discriminator.build(input_shape=self.disc_input_shape)
        super().build(input_shape)
        return
    
    def get_config(self):
        """
        More info on custom loading and saving:
        https://keras.io/guides/serialization_and_saving/
        Runs as part of the saving process to serialize the model when calling model.save()
        Runs first to get the default configuration for the model
        """
        # get the default configuration for the model
        base_config = super().get_config()
        
        # get extra configurations for custom model attributes (not Python objects like ints, strings, etc.)
        custom_config = {
            "discriminator": keras.saving.serialize_keras_object(self.discriminator),
            "generator": keras.saving.serialize_keras_object(self.generator),
            "latent_dim": self.latent_dim,
            "discriminator_extra_steps": self.num_disc_steps,
            "gp_weight": self.gp_weight,
        }
        
        # combine the two configurations
        config = {**base_config, **custom_config}
        return config
    
    def get_compile_config(self):
        """
        More info on custom loading and saving:
        https://keras.io/guides/customizing_saving_and_serialization/#getcompileconfig-and-compilefromconfig
        Runs as part of the saving process to serialize the compiled parameters when calling model.save()
        Runs after get_config() to serialize the compiled parameters
        """
        # These parameters will be serialized at saving time
        config = {
            "disc_optimizer": self.disc_optimizer.get_config(),
            "disc_optimizer_state": [v.numpy() for v in self.disc_optimizer.variables],
            "gen_optimizer": self.gen_optimizer.get_config(),
            "gen_optimizer_state": [v.numpy() for v in self.gen_optimizer.variables],
            "disc_loss_fn": keras.saving.serialize_keras_object(self.disc_loss_fn),
            "gen_loss_fn": keras.saving.serialize_keras_object(self.gen_loss_fn),
        }
        return config
    
    @classmethod
    def from_config(cls, config, custom_objects=None):
        """
        More info on custom loading and saving:
        https://keras.io/guides/serialization_and_saving/
        Runs as part of the loading process to deserialize the model when calling keras.models.load_model()
        Runs first to deserialize the model configuration
        """
        # get the discriminator and generator models from the config
        discriminator = keras.saving.deserialize_keras_object(config.pop("discriminator"), custom_objects=custom_objects)
        generator = keras.saving.deserialize_keras_object(config.pop("generator"), custom_objects=custom_objects)
        latent_dim = config.pop("latent_dim")
        discriminator_extra_steps = config.pop("discriminator_extra_steps")
        gp_weight = config.pop("gp_weight")
        
        # create a new instance of the model with the discriminator and generator models (This calls the __init__ method)
        model = cls(discriminator=discriminator, generator=generator, latent_dim=latent_dim, discriminator_extra_steps=discriminator_extra_steps, 
                    gp_weight=gp_weight)
        return model
    
    def compile_from_config(self, config):
        """
        More info on custom loading and saving:
        https://keras.io/guides/customizing_saving_and_serialization/#getcompileconfig-and-compilefromconfig
        Runs as part of the loading process to deserialize the compiled parameters when calling keras.models.load_model()
        Runs after from_config() to deserialize the compiled parameters
        """
        # Deserialize the compiled parameters
        self.disc_optimizer = keras.optimizers.Adam.from_config(config["disc_optimizer"])
        self.disc_optimizer_state = config["disc_optimizer_state"]
        for var, val in zip(self.disc_optimizer.variables, self.disc_optimizer_state):
            var.assign(val)
        
        self.gen_optimizer = keras.optimizers.Adam.from_config(config["gen_optimizer"])
        self.gen_optimizer_state = config["gen_optimizer_state"]
        for var, val in zip(self.gen_optimizer.variables, self.gen_optimizer_state):
            var.assign(val)
            
        self.disc_loss_fn = keras.saving.deserialize_keras_object(config["disc_loss_fn"])
        self.gen_loss_fn = keras.saving.deserialize_keras_object(config["gen_loss_fn"])
        
        # call the compile method to set the optimizer and loss functions
        self.compile(
            disc_optimizer=self.disc_optimizer, 
            gen_optimizer=self.gen_optimizer, 
            disc_loss_fn=self.disc_loss_fn, 
            gen_loss_fn=self.gen_loss_fn
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
        
        # deterministically generate the random latent vectors for generating validation samples so that the same random latent vectors are used
        # every time the model is reloaded.
        generator = tf.random.Generator.from_seed(112)
        self.random_latent_vectors = generator.normal(shape=(self.num_img, self.latent_dim))
        
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
            self.loss_metrics_dataframe = pd.DataFrame()  # empty dataframe placeholder that will be overwritten in log_loss_to_dataframe
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
        
        # save a copy of the model to the current epoch checkpoint directory
        model_save_path = self.this_epoch_checkpoint_dir.joinpath("model.keras")
        self.model.save(model_save_path)
        
        # Plot individual loss metrics
        self.create_loss_plots()
        
        # Generate validation samples
        self.generate_validation_samples()
        return
    
    def on_train_end(self, logs=None):
        epoch = len(self.loss_metrics_dataframe)
        # skip generating a gif if training ends on a multiple of the frequency to create a gif because it was already generated in the 
        # generate_validation_samples method at the end of the last epoch.
        if epoch % self.gif_creation_frequency != 0:
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
        
        # get the current memory usage in MB and GB
        mem_usage_mb, mem_usage_gb = get_memory_usage()
        # get the current GPU memory usage in MB and GB
        gpu_mem_usage_mb, gpu_mem_usage_gb = get_gpu_memory_usage()
        
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
            "model_loaded": self.model_recently_loaded,
            "memory_usage_gb": mem_usage_gb,
            "gpu_memory_usage_gb": gpu_mem_usage_gb
        }])
        
        # if the model was loaded, set the model_loaded column to False for the next epoch
        self.model_recently_loaded = False  # we only want to mark the first epoch after loading
        
        if epoch == 1:
            # if this is the first epoch, set the loss_metrics_dataframe to the new row
            self.loss_metrics_dataframe = new_row
        else:
            # concatenate the new row to the existing loss dataframe
            self.loss_metrics_dataframe = pd.concat([self.loss_metrics_dataframe, new_row], ignore_index=True)
        
        # Save the losses dataframe to a CSV file
        csv_save_path = self.this_epoch_checkpoint_dir.joinpath("loss_metrics.csv")
        self.loss_metrics_dataframe.to_csv(csv_save_path, index=False)
        return
    
    def create_loss_plots(self):
        """
        Generates and saves individual and combined loss plots for the training process with a secondary x-axis showing
        the number of samples trained on.
        """
        # Define list of individual loss columns to plot
        loss_columns = ["disc_loss", "disc_loss_real", "disc_loss_fake", "disc_loss_gp", "gen_loss"]
        
        # Identify locations where model was loaded
        model_load_indices = self.loss_metrics_dataframe.loc[self.loss_metrics_dataframe['model_loaded'], 'epoch']
        
        # get the number of samples trained and epochs for the x-axis
        samples_trained = self.loss_metrics_dataframe['samples_trained_on']
        epochs = self.loss_metrics_dataframe['epoch']
        
        ################################ Loop through each loss metric to create individual plots ###############################
        for col in loss_columns:
            # Create the plot
            fig, ax1 = plt.subplots(figsize=(10, 6))
            
            # Plot the loss metric
            ax1.plot(epochs, self.loss_metrics_dataframe[col], label=col.replace("_", " ").title(), zorder=1)
            
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
            
            # move the word "loss" to the front of the filename for better sorting
            col_for_filename = col.replace("_loss", "")
            col_for_filename = f"loss_{col_for_filename}"
            
            # Save plot to the current epoch checkpoint directory
            plot_path = self.this_epoch_checkpoint_dir.joinpath(f"{col_for_filename}.png")
            plt.savefig(plot_path, dpi=200)
            shutil.copy(plot_path, self.model_training_output_dir.joinpath(f"{col_for_filename}.png"))
            plt.close()
            plt.clf()
        
        ################################ Combined plot for specific losses ###############################
        combined_loss_columns = ["disc_loss", "disc_loss_real", "disc_loss_fake", "gen_loss"]
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        # Plot the loss metrics
        for col in combined_loss_columns:
            ax1.plot(epochs, self.loss_metrics_dataframe[col], label=col.replace("_", " ").title(), zorder=1)
        
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
        shutil.copy(combined_plot_path, self.model_training_output_dir.joinpath("losses_combined.png"))
        plt.close()
        plt.clf()
        print(f"\nCombined loss plot saved to: {combined_plot_path}")
        
        ############################### create a plot to show memory usage over time ##############################
        fig, ax1 = plt.subplots(figsize=(10, 6))
        # Plot the memory usage in GB
        ax1.plot(epochs, self.loss_metrics_dataframe["memory_usage_gb"], label="Memory Usage (GB)", zorder=1)
        
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
        
        # Save the memory usage plot to the current epoch checkpoint directory
        memory_plot_path = self.this_epoch_checkpoint_dir.joinpath("memory_usage.png")
        plt.savefig(memory_plot_path, dpi=200)
        shutil.copy(memory_plot_path, self.model_training_output_dir.joinpath("memory_usage.png"))
        plt.close()
        plt.clf()
        
        ############################## create a plot to show GPU memory usage over time ##############################
        fig, ax1 = plt.subplots(figsize=(10, 6))
        # Plot the GPU memory usage in GB
        ax1.plot(epochs, self.loss_metrics_dataframe["gpu_memory_usage_gb"], label="GPU Memory Usage (GB)", zorder=1)
        
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
        
        # Save the GPU memory usage plot to the current epoch checkpoint directory
        gpu_memory_plot_path = self.this_epoch_checkpoint_dir.joinpath("memory_usage_GPU.png")
        plt.savefig(gpu_memory_plot_path, dpi=200)
        shutil.copy(gpu_memory_plot_path, self.model_training_output_dir.joinpath("memory_usage_GPU.png"))
        plt.close()
        plt.clf()
        
        return
    
    def generate_validation_samples(self):
        """
        Generates and saves a grid of validation images produced by the generator model, 
        along with their discriminator scores. The images are saved to the current epoch 
        checkpoint directory and a copy is saved to the model training output directory. 
        Additionally, a GIF of all saved images is generated every specified number of epochs.
        
        Parameters:
            None
        
        Returns:
            None
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
        
        # Add a title to the plot with the epoch and number of samples trained
        epoch = len(self.loss_metrics_dataframe)
        samples_trained = self.loss_metrics_dataframe['samples_trained_on'].iloc[-1]
        fig.suptitle(f"Validation Samples - Epoch {epoch}, Samples Trained: {samples_trained:,}", fontsize=12)
        
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
