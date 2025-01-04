"""
wgan-gp model definition

wgan-gp combines the critic and generator into one model for training.
"""


import time
import keras
import tensorflow as tf
import warnings


# choose to ignore the specific warning for loading optimizers as our WGAN_GP class handles this in the compile_from_config method. This method 
# may not perfectly recreate the optimizers, but it doesn't seem to cause any issues in testing so far.
warnings.filterwarnings(
    "ignore", 
    message=r"Skipping variable loading for optimizer.*",
    category=UserWarning
)


class WGAN_GP(keras.Model):
    """
    WGAN-GP model wrapper for training a Wasserstein GAN with gradient penalty.
    
    Parameters to __init__:
        critic (keras.Model): The critic model.
        generator (keras.Model): The generator model.
        num_classes (int): The number of classes in the dataset.
        latent_dim (int): The dimensionality of the latent space.
        critic_learning_rate (float): The learning rate for the critic.
        generator_learning_rate (float): The learning rate for the generator.
        learning_rate_warmup_epochs (int): The number of epochs before learning rate decay begins.
        learning_rate_decay (float): The rate at which the learning rate decays each epoch after the warmup period.
        min_critic_learning_rate (float): The minimum learning rate for the critic.
        min_generator_learning_rate (float): The minimum learning rate for the generator.
        critic_extra_steps (int): The number of extra critic training steps per generator training step.
        gp_weight (float): The weight of the gradient penalty term.
    
    Parameters to compile:
        critic_optimizer (keras.optimizers.Optimizer): The optimizer for the critic.
        gen_optimizer (keras.optimizers.Optimizer): The optimizer for the generator.
    
    Methods:
        __init__: Initializes the WGAN-GP model.
        compile: Compiles the model and adds the critic and generator optimizers.
        gradient_penalty: Computes the gradient penalty for the critic.
        train_step: The custom training step for the WGAN-GP
        build: Builds the model. Required to not throw errors when saving and loading the model.
        get_config: Custom saving step 1
        get_compile_config: Custom saving step 2
        from_config: Custom loading step 1
        compile_from_config: Custom loading step 2
    """
    def __init__(
        self,
        critic,
        generator,
        latent_dim,
        critic_learning_rate,
        generator_learning_rate,
        learning_rate_warmup_epochs,
        learning_rate_decay,
        min_critic_learning_rate,
        min_generator_learning_rate,
        critic_extra_steps=5,
        gp_weight=10.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        # the critic and generator models the WGAN-GP model will train
        self.critic = critic
        self.generator = generator
        
        # used in the training step to generate random latent vectors for the generator
        self.latent_dim = latent_dim
        
        # learning rate parameters used by the learning_rate_scheduler method in the Training_Monitor callback
        self.critic_learning_rate = critic_learning_rate
        self.generator_learning_rate = generator_learning_rate
        # learning rate warmup epochs value is the number of epochs the model trains for before the learning rate decay begins
        self.learning_rate_warmup_epochs = learning_rate_warmup_epochs
        self.learning_rate_decay = learning_rate_decay
        # minimum learning rates for the critic and generator where the learning rate decay ends
        self.min_critic_learning_rate = min_critic_learning_rate
        self.min_generator_learning_rate = min_generator_learning_rate
        
        # in a WGAN-GP, the critic trains for a number of steps and then generator trains for one step
        self.num_critic_steps = critic_extra_steps
        # the weight of the gradient penalty term in the critic loss
        self.gp_weight = gp_weight
        
        # set dummy value for self.optimizer avoid errors at the end of .fit() call because we are using a custom training loop
        self.optimizer = None
        
        # build the model to avoid errors when reloading the model after saving
        self.build()
        return
    
    def compile(self, critic_optimizer=None, gen_optimizer=None):
        super(WGAN_GP, self).compile()
        self.critic_optimizer = critic_optimizer
        self.gen_optimizer = gen_optimizer
        return
    
    def gradient_penalty(self, batch_size, real_images, fake_images, image_one_hot_labels):
        """
        Computes the gradient penalty for WGAN-GP.
        
        The gradient penalty is a term added to the model's loss to ensure that the critic (or discriminator) behaves in a stable and smooth way. 
        It encourages the critic to have gradients with a norm (magnitude) close to 1, which is a key requirement for Wasserstein GANs to work well. 
        This makes training more stable and helps avoid common problems like mode collapse or gradient explosion.
        
        **What does it mean practically?**
        - The gradient penalty acts as a guide for the critic, ensuring it doesn't make overly sharp distinctions between real and fake images.
        - It does this by looking at points in-between real and fake images (called "interpolated samples") and checking how the critic's predictions
            change there.
        - If the critic's predictions change too drastically (indicated by gradients that are too large or too small), the penalty corrects this 
            during training.
        
        This process ensures the critic learns to separate real and fake images in a way that is both robust and mathematically sound.
        
        Args:
            batch_size (int): The size of the batch.
            real_images (tf.Tensor): A batch of real images.
            fake_images (tf.Tensor): A batch of generated (fake) images.
            image_one_hot_labels (tf.Tensor): One-hot encoded labels corresponding to the images.
        
        Returns:
            tuple: A tuple containing:
                - gp (tf.Tensor): The gradient penalty, which is added to the critic's loss.
        """
        # Generate random values between 0 and 1 for interpolation weights.
        alpha = tf.random.uniform([batch_size, 1, 1, 1], minval=0.0, maxval=1.0, dtype="float16")
        
        # Ensure the real images are in the correct data type for mixed precision training.
        real_images = tf.cast(real_images, dtype="float16")
        
        # Create interpolated samples by blending real and fake images using the alpha weights.
        interpolated = real_images + alpha * (fake_images - real_images)
        
        # Use a gradient tape to calculate the gradients of the critic's predictions with respect to the interpolated images.
        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            pred = self.critic([interpolated, image_one_hot_labels], training=True)
        
        # Compute the gradients of the critic's predictions with respect to the interpolated images.
        grads = gp_tape.gradient(pred, [interpolated])[0]
        
        # Calculate the norm (magnitude) of the gradients for each interpolated sample.
        gradient_norms = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        
        # Compute the gradient penalty as the mean squared difference between the gradient norms and the target value of 1.
        gp = tf.reduce_mean((gradient_norms - 1.0) ** 2)
        return gp
    
    def train_step(self, batch_data):
        # unpack the batch data into images and labels
        real_images, real_labels = batch_data
        # Get the batch size from the data as sometimes the last batch can be smaller
        batch_size = tf.shape(real_images)[0]
        
        # # Initialize lists to track training metrics
        critic_real_scores = []
        critic_fake_scores = []
        gradient_penalties = []
        critic_losses = []
        
        ################################################################ Train Critic ################################################################
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
                
                # Calculate the critic loss using the Wasserstein loss
                critic_wasserstein_loss = tf.reduce_mean(fake_logits) - tf.reduce_mean(real_logits)
                
                # Calculate the gradient penalty
                gp = self.gradient_penalty(batch_size, real_images, fake_images, real_labels)
                
                # Add the gradient penalty to the critic loss
                critic_loss = critic_wasserstein_loss + gp * self.gp_weight
                
                # track critic loss and gradient penalty for this step
                gradient_penalties.append(gp)
                critic_losses.append(critic_loss)
            
            # Get the gradients for the critic loss
            critic_gradient = tape.gradient(critic_loss, self.critic.trainable_variables)
            # Update the critic's weights
            self.critic_optimizer.apply_gradients(zip(critic_gradient, self.critic.trainable_variables))
            
            # consider resampling the dataset to get a new batch of real images and labels here to avoid training on the same batch
            #         multiple times. This could be achieved by making a copy of the dataset a class attribute.
            # Example:
            # real_images, real_labels = next(iter(self.train_dataset))
        
        ############################################################## Train Generator ###############################################################
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
        
        # Calculate the average scores for real and fake images for this batch
        avg_real_score = tf.reduce_mean(critic_real_scores)
        avg_fake_score = tf.reduce_mean(critic_fake_scores)
        avg_gradient_penalty = tf.reduce_mean(gradient_penalties)
        avg_critic_loss = tf.reduce_mean(critic_losses)
        
        return_dict = {
            "critic_loss": avg_critic_loss,
            "gen_loss": gen_loss,
            "critic_loss_real": avg_real_score,
            "critic_loss_fake": avg_fake_score,
            "gradient_penalty": avg_gradient_penalty,
            }
        
        return return_dict
    
    def build(self, input_shape=None):
        # required to avoid errors when saving and loading the model
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
        
        # get the custom configuration for the model
        custom_config = {
            "critic": keras.saving.serialize_keras_object(self.critic),
            "generator": keras.saving.serialize_keras_object(self.generator),
            "latent_dim": self.latent_dim,
            "critic_extra_steps": self.num_critic_steps,
            "gp_weight": self.gp_weight,
            "critic_learning_rate": self.critic_learning_rate,
            "generator_learning_rate": self.generator_learning_rate,
            "learning_rate_warmup_epochs": self.learning_rate_warmup_epochs,
            "learning_rate_decay": self.learning_rate_decay,
            "min_critic_learning_rate": self.min_critic_learning_rate,
            "min_generator_learning_rate": self.min_generator_learning_rate,
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
        # get the critic and generator models from the config
        critic = keras.saving.deserialize_keras_object(config.pop("critic"), custom_objects=custom_objects)
        generator = keras.saving.deserialize_keras_object(config.pop("generator"), custom_objects=custom_objects)
        
        # get the extra configurations for custom model attributes
        latent_dim = config.pop("latent_dim")
        critic_learning_rate = config.pop("critic_learning_rate")
        generator_learning_rate = config.pop("generator_learning_rate")
        learning_rate_warmup_epochs = config.pop("learning_rate_warmup_epochs")
        learning_rate_decay = config.pop("learning_rate_decay")
        min_critic_learning_rate = config.pop("min_critic_learning_rate")
        min_generator_learning_rate = config.pop("min_generator_learning_rate")
        critic_extra_steps = config.pop("critic_extra_steps")
        gp_weight = config.pop("gp_weight")
        
        # create a new instance of the model with the critic and generator models (This calls the __init__ method)
        model = cls(
            critic=critic, 
            generator=generator,
            latent_dim=latent_dim,
            critic_learning_rate=critic_learning_rate,
            generator_learning_rate=generator_learning_rate,
            learning_rate_warmup_epochs=learning_rate_warmup_epochs,
            learning_rate_decay=learning_rate_decay,
            min_critic_learning_rate=min_critic_learning_rate,
            min_generator_learning_rate=min_generator_learning_rate,
            critic_extra_steps=critic_extra_steps, 
                    gp_weight=gp_weight)
        return model
    
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


if __name__ == "__main__":
    for num in range(10):
        print("This script contains the WGAN_GP class definition and is not meant to be run directly.")
        time.sleep(0.1)
