"""
training monitor callback for logging training metrics, saving model checkpoints, generating validation samples, and creating plots and 
visualizations of the wgan-gp training progress.
"""


import os
import time
from datetime import datetime
import shutil
import numpy as np
import pandas as pd
import cv2
import keras
import tensorflow as tf
import scipy.linalg
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")  # set to use the "Agg" to avoid tkinter error

from utils import get_memory_usage, get_gpu_memory_usage, get_timestamp, get_readable_time_string


class Training_Monitor(tf.keras.callbacks.Callback):
    """
    Training_Monitor is a custom Keras callback for monitoring and logging the training process of a GAN model. It performs various tasks at the end
    of each epoch, including logging loss metrics, saving model checkpoints, generating validation samples, and creating plots and videos of the 
    training progress.
    
    Parameters to __init__:
        model_training_output_dir: str, the main directory to save outputs from the model training
        model_checkpoints_dir: str, the directory to save the model checkpoints and info at each epoch
        noise_dim: int, the noise dimension of the generator
        model_save_frequency: int, the frequency (in epochs) to save the model
        video_of_validation_frequency: int, the frequency (in epochs) to generate a video of the validation samples
        FID_score_frequency: int, the frequency (in epochs) to calculate the FID score between the real and generated images
        train_dataset: tf.data.Dataset, the training dataset used to calculate the FID_score_frequency
        samples_per_epoch: int, the number of samples in the training dataset
        last_checkpoint_dir_path: str, the path to the last checkpoint directory if the model is being reloaded
    
    Methods:
        __init__: Initializes the Training_Monitor object
        on_train_begin: Called at the start of model training to print a message to the console
        on_epoch_begin: Called at the start of each epoch during training to get the start time of the epoch
        on_epoch_end: Called at the end of each epoch during training to perform various tasks (Main method of the class)
        on_train_end: Called at the end of model training to print a message to the console and save the final model checkpoint
        update_current_epoch: Updates the current epoch number for logging metrics
        set_this_epoch_checkpoint_dir: Creates a new checkpoint directory for logging info at the end of this epoch
        log_data_to_dataframe: Logs the loss metrics to a DataFrame and saves it to a CSV file
        plot_training_metrics: Generates and saves individual and combined loss plots for the training process
        learning_rate_scheduler: Decays the learning rates of the critic and generator models after the warmup period
        save_model_checkpoint: Saves a copy of the model to the current epoch checkpoint directory
        generate_validation_samples: Generates and saves a grid of validation images produced by the generator model
        generate_video_of_validation_samples: Generates a video of all saved validation images
        plot_epoch_duration_and_estimate_train_time: Plots the duration of the epoch and metric calculations and estimates the time to train for a
            number of epochs
    """
    def __init__(self,
                model_training_output_dir,
                model_checkpoints_dir,
                noise_dim,
                model_save_frequency=5,
                video_of_validation_frequency=5,
                FID_score_frequency=0,
                train_dataset=None,
                samples_per_epoch=None,
                last_checkpoint_dir_path=None
                ):
        self.model_training_output_dir = model_training_output_dir
        self.model_checkpoints_dir = model_checkpoints_dir
        self.this_epoch_checkpoint_dir = None  # the path to the current model checkpoint (will be set/updated in on_epoch_end)
        
        # parameters for logging the duration of model training and metric handling
        self.epoch_start_time = None
        self.epoch_train_duration = None
        self.metric_calc_start_time = None
        # placeholder for the duration of logging the metrics because this is actually logging the duration of the last's epoch's metric logging
        # duration, but it's close enough for our purposes.
        self.metric_calc_duration = datetime.now() - datetime.now()
        
        # generate static inputs for generating validation samples to see how the model is improving over time
        self.noise_dim = noise_dim # noise dimension from the generator model
        self.validation_sample_grid = (10, 15)  # rows, columns
        self.num_validation_samples = self.validation_sample_grid[0] * self.validation_sample_grid[1]
        # deterministically generate the random noise vectors for generating validation samples so that the same random noise vectors are used
        # every time the model is reloaded and across different runs.
        generator = tf.random.Generator.from_seed(112)
        self.random_noise_vectors = generator.normal(shape=(self.validation_sample_grid[0] * self.validation_sample_grid[1], self.noise_dim))
        # generate sample labels for MNIST (0-9)
        self.validation_sample_labels = []
        for MNIST_label in range(self.validation_sample_grid[0]):  # loop through the number of labels
            for _ in range(self.validation_sample_grid[1]):  # add the label for each image in the row
                self.validation_sample_labels.append(MNIST_label)
        # convert the list of labels to a tensor
        self.validation_sample_labels = tf.convert_to_tensor(self.validation_sample_labels, dtype=tf.int32)
        
        # frequency (in epochs) to save the model, create a video of the validation samples, and calculate the FID score
        self.model_save_frequency = model_save_frequency
        self.video_of_validation_frequency = video_of_validation_frequency
        self.FID_score_frequency = FID_score_frequency
        # if we are calculating the FID_score_frequency, initialize the variables needed for the FID score calculation
        if self.FID_score_frequency > 0:
            # By default, include_top=False + pooling='avg' gives a (N, 2048) feature vector
            self.inception_model = keras.applications.inception_v3.InceptionV3(weights='imagenet', include_top=False, pooling='avg')
        else:
            self.inception_model = None
        # placeholders for the real and generated images to calculate the FID score
        self.mu_real = None
        self.sigma_real = None
        
        # a reference to the training dataset to calculate the FID score between the real and generated images
        self.train_dataset = train_dataset
        
        # counter to track the number of samples the model has been trained on so far (not including the extra steps for the critic)
        self.samples_per_epoch = samples_per_epoch
        
        # load the training metrics csv to a dataframe if the last_checkpoint_dir_path is provided (we are reloading the model)
        if last_checkpoint_dir_path is not None:
            # get the path to the training_metrics.csv file
            training_metrics_csv_path = last_checkpoint_dir_path.joinpath("training_metrics.csv")
            # load the csv to a dataframe
            self.metrics_dataframe = pd.read_csv(training_metrics_csv_path)
            # remember that the model was loaded for logging loss metrics at the end of the next epoch
            self.model_recently_loaded = True
        else:  # create a new dataframe for tracking loss metrics for a fresh start
            self.metrics_dataframe = pd.DataFrame()  # empty dataframe placeholder that will be overwritten in log_loss_to_dataframe
            self.model_recently_loaded = False
        
        # if it does not exist in the model_training_output_dir, create a _NOTES.txt file for writing notes about the model training run
        notes_file_path = model_training_output_dir.joinpath("_NOTES.txt")
        if not os.path.exists(notes_file_path):
            with open(notes_file_path, "w") as file:
                file.write(f"{'='*140}\n")
                file.write(f"{' '*43}A place to write notes about this model training run.\n")
                file.write(f"{'='*140}\n\n")
        return
    
    def on_train_begin(self, logs=None):
        """
        Called at the start of model training.
        Parameters:
            logs (dict, optional): Dictionary of logs containing loss metrics.
        Returns:
            None
        """
        # print out a message to the console to indicate that the training has started
        print(f"\n{'#'*63} MODEL TRAINING STARTED {'#'*63}")
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
        
        This method performs the following tasks:
            1. Get the duration of the time spent training this epoch.
            2. Start the timer for calculating the duration of logging metrics.
            3. Get the current epoch number for logging metrics.
            4. Set the current epoch checkpoint directory.
            5. Log the data about training to the DataFrame.
            6. Plot tracked metrics.
            7. Update the learning rates of the optimizers if past the warmup period.
            8. Generate validation samples.
            9. Save the model at regular intervals.
            10. Visualize the training progress with a video of the validation samples at regular intervals.
            11. Calculate the duration of logging the metrics.
            12. Plot train time and metrics calculations and estimate the time to train for a number of epochs.
        
        Parameters:
            epoch (int): The current epoch number.
            logs (dict, optional): Dictionary of logs containing loss metrics.
        
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
        
        # Generate validation samples
        self.generate_validation_samples()
        
        # save the model at regular intervals (if the model_save_frequency is greater than 0)
        if self.model_save_frequency > 0 and self.current_epoch % self.model_save_frequency == 0:
            # Save a copy of the model to the current epoch checkpoint directory
            self.save_model_checkpoint()
        
        # visualize the training progress with a video of the validation samples at regular intervals (if the video_of_validation_frequency is
        # greater than 0)
        if self.video_of_validation_frequency > 0 and self.current_epoch > 1 and self.current_epoch % self.video_of_validation_frequency == 0:
            # Generate a video of all the saved images
            self.generate_video_of_validation_samples()
        
        # if calculate the FID score between the real and generated images at regular intervals (if the FID_score_frequency is greater than 0)
        if self.FID_score_frequency > 0 and self.current_epoch % self.FID_score_frequency == 0:
            # calculate the FID score between the real and generated images
            self.calculate_FID_score()
        
        # Calculate the duration of logging the metrics (will be logged in the next epoch log_data_to_dataframe call)
        self.metric_calc_duration = datetime.now() - self.metric_calc_start_time
        
        # plot train time and metrics calculations and estimate the time to train for a number of epochs. Even though the metric calc time
        # doesn't include this time, it's tracked for the operations that take the most time.
        self.plot_epoch_duration_and_estimate_train_time()
        return
    
    def on_train_end(self, logs=None):
        # if we haven't saved a model checkpoint at the end of the last epoch, save one now.  model_save_frequency of zero means no model has been
        # saved up to this point so we need to save one now.
        if self.model_save_frequency == 0 or self.current_epoch % self.model_save_frequency != 0:
            # Save a copy of the model to the current epoch checkpoint directory
            self.save_model_checkpoint()
        
        # if we haven't generated a video of the validation samples at the end of the last epoch, generate one now. video_of_validation_frequency
        # of zero means no video has been generated up to this point so we need to generate one now.
        if self.video_of_validation_frequency == 0 or (self.current_epoch > 1 and self.current_epoch % self.video_of_validation_frequency != 0):
            # Generate a video of all the saved images
            self.generate_video_of_validation_samples()
        
        # if we haven't calculated the FID score between the real and generated images at the end of the last epoch and the FID_score_frequency is
        # greater than 0, calculate it now.
        if self.FID_score_frequency > 0 and self.current_epoch % self.FID_score_frequency != 0:
            # calculate the FID score between the real and generated images
            self.calculate_FID_score()
        
        # print out a message to the console to indicate that the training has ended
        print(f"{'#'*64} MODEL TRAINING ENDED {'#'*64}\n")
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
        
        # get the critic and generator learning rates. Updated in the learning_rate_scheduler method. In the current implementation, the values
        # are the same so we will get the learning rate from the model object to avoid floating point rounding errors that slightly change the
        # values when extracting from self.model.critic_optimizer.learning_rate.numpy() and self.model.gen_optimizer.learning_rate.numpy()
        critic_learning_rate = self.model.critic_learning_rate
        generator_learning_rate = self.model.generator_learning_rate
        
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
            "FID_score": 0.0  # placeholder value that will be replaced if the FID score is calculated
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
        
        ########################################## Loop through each loss metric to create individual plots ##########################################
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
                # add text to show the epoch after the model was loaded
                y_text_position = ((ax1.get_ylim()[1] - ax1.get_ylim()[0]) * 0.02) + ax1.get_ylim()[0]
                ax1.text(model_load_epoch, y_text_position, f"{model_load_epoch:,}", color='#5C5C5C', fontsize=9, rotation=90, va='bottom', 
                        ha='right', zorder=2)
            
            # add text to the plot to show the number of samples trained on at the end of the last epoch
            plt.text(0.86, 1.02, f"Samples Trained On: {samples_trained.iloc[-1]:,}", fontsize=9, ha='center', va='center', 
                    transform=plt.gca().transAxes)
            
            ax1.set_xlabel("Epoch")
            ax1.set_ylabel(col.replace("_", " ").title())
            ax1.legend()
            ax1.grid(True, alpha=0.4)
            plt.title(f"{col.replace('_', ' ').title()} vs. Epoch", fontsize=18)
            plt.tight_layout()
            
            # move the word "loss" to the front of the filename for better sorting for the loss plots
            col_for_filename = col.replace("_loss", "")
            col_for_filename = f"loss_{col_for_filename}"
            
            # Save plot to the current epoch checkpoint directory
            plot_path = self.this_epoch_checkpoint_dir.joinpath(f"{col_for_filename}.png")
            plt.savefig(plot_path, dpi=100)
            shutil.copy(plot_path, self.model_training_output_dir.joinpath(f"{col_for_filename}.png"))
            plt.close()
            plt.clf()
        
        ##################################################### Combined plot for specific losses ######################################################
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
            # add text to show the epoch after the model was loaded
            y_text_position = ((ax1.get_ylim()[1] - ax1.get_ylim()[0]) * 0.02) + ax1.get_ylim()[0]
            ax1.text(model_load_epoch, y_text_position, f"{model_load_epoch:,}", color='#5C5C5C', fontsize=9, rotation=90, va='bottom', ha='right',
                    zorder=2)
        
        # add text to the plot to show the number of samples trained on at the end of the last epoch
        plt.text(0.86, 1.02, f"Samples Trained On: {samples_trained.iloc[-1]:,}", fontsize=9, ha='center', va='center', 
                transform=plt.gca().transAxes)
        
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss Value")
        ax1.legend()
        ax1.grid(True, alpha=0.4)
        plt.title("All Losses vs. Epoch", fontsize=18)
        plt.tight_layout()
        
        # Save the combined plot to the current epoch checkpoint directory
        combined_plot_path = self.this_epoch_checkpoint_dir.joinpath("losses_combined.png")
        plt.savefig(combined_plot_path, dpi=100)
        combined_plot_main_dir_path = self.model_training_output_dir.joinpath("losses_combined.png")
        shutil.copy(combined_plot_path, combined_plot_main_dir_path)
        plt.close()
        plt.clf()
        print(f"\nCombined loss plot saved to: {combined_plot_main_dir_path}")
        
        #################################################### critic loss real vs fake difference #####################################################
        # Create a plot for the difference of the critic loss on real and fake samples
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        # Calculate the difference between critic_loss_real and critic_loss_fake
        diff_data = self.metrics_dataframe["critic_loss_real"] - self.metrics_dataframe["critic_loss_fake"]
        
        # Plot the diff data
        ax1.plot(epochs, diff_data, label="Critic Loss Real - Fake", zorder=1, color=plot_line_colors[0])
        
        # Add vertical lines for model loading events with a single label for legend
        for i, model_load_epoch in enumerate(model_load_indices):
            if i == 0:
                line_label = "ckpt loaded"
            else:
                line_label = ""
            ax1.axvline(x=model_load_epoch, color='#5C5C5C', linestyle='--', alpha=0.7, linewidth=1.5, label=line_label, zorder=-1)
            # add text to show the epoch after the model was loaded
            y_text_position = ((ax1.get_ylim()[1] - ax1.get_ylim()[0]) * 0.02) + ax1.get_ylim()[0]
            ax1.text(model_load_epoch, y_text_position, f"{model_load_epoch:,}", color='#5C5C5C', fontsize=9, rotation=90, va='bottom', ha='right',
                    zorder=2)
        
        # find the epoch where the minimum value occurred
        min_diff_epoch = diff_data.idxmin() + 1  # +1 because the index starts from 0 and epoch starts from 1
        # if the minimum difference is in the first 10 epochs, find the minimum difference after the first 10 epochs
        if min_diff_epoch < 10 and len(diff_data) > 10:
            diff_data_after_5 = diff_data[10:]
            min_diff_epoch = diff_data_after_5.idxmin() + 1
        
        # Plot a vertical line at the epoch with the minimum critic loss difference
        ax1.axvline(x=min_diff_epoch, color='black', linestyle='-', linewidth=1.5, label='Minimum Difference', zorder=-1)
        
        # Add text annotation for the minimum line
        y_text_position = ((ax1.get_ylim()[1] - ax1.get_ylim()[0]) * 0.25) + ax1.get_ylim()[0]
        x_text_position = min_diff_epoch + (ax1.get_xlim()[1] - ax1.get_xlim()[0]) * 0.007
        ax1.text(x_text_position, y_text_position, f"  Min diff at Epoch {min_diff_epoch:,}", color='black', fontsize=9, rotation=90, 
                va='bottom', ha='left',zorder=2)
        
        # add text to the plot to show the number of samples trained on at the end of the last epoch
        plt.text(0.86, 1.02, f"Samples Trained On: {samples_trained.iloc[-1]:,}", fontsize=9, ha='center', va='center',
                transform=plt.gca().transAxes)
        
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Critic Loss Real - Fake")
        ax1.legend()
        ax1.grid(True, alpha=0.4)
        plt.title("Critic Loss Real - Fake vs. Epoch", fontsize=18, loc='left')
        
        plot_path = self.this_epoch_checkpoint_dir.joinpath("loss_critic_real_fake_diff.png")
        plt.savefig(plot_path, dpi=100)
        shutil.copy(plot_path, self.model_training_output_dir.joinpath("loss_critic_real_fake_diff.png"))
        plt.close()
        plt.clf()
        
        ########################################### Stacked plots for critic and generator learning rates ############################################
        loss_columns = ["critic_loss", "gen_loss"]
        learning_rates_columns = ["critic_learning_rate", "generator_learning_rate"]
        plot_line_colors = ["#1F77B4", "#D62728"]
        
        for loss_col, lr_col, line_color in zip(loss_columns, learning_rates_columns, plot_line_colors):
            # Create a learning rate plot with a shared x-axis of the loss
            fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
            
            # Plot the loss metric
            axs[0].plot(epochs, self.metrics_dataframe[loss_col], label=loss_col.replace("_", " ").title(), zorder=1, color=line_color)
            axs[0].set_ylabel(loss_col.replace("_", " ").title())
            
            # Plot the learning rate
            axs[1].plot(epochs, self.metrics_dataframe[lr_col], label=lr_col.replace("_", " ").title(), zorder=1, color=line_color)
            axs[1].set_ylabel(lr_col.replace("_", " ").title())
            
            # Add vertical lines for model loading events with a single label for legend
            for i, model_load_epoch in enumerate(model_load_indices):
                if i == 0:
                    line_label = "ckpt loaded"
                else:
                    line_label = ""
                axs[0].axvline(x=model_load_epoch, color='#5C5C5C', linestyle='--', alpha=0.7, linewidth=1.5, label=line_label, zorder=-1)
                axs[1].axvline(x=model_load_epoch, color='#5C5C5C', linestyle='--', alpha=0.7, linewidth=1.5, label=line_label, zorder=-1)
                # add text to show the epoch after the model was loaded
                y_text_position = ((axs[0].get_ylim()[1] - axs[0].get_ylim()[0]) * 0.02) + axs[0].get_ylim()[0]
                axs[0].text(model_load_epoch, y_text_position, f"{model_load_epoch:,}", color='#5C5C5C', fontsize=9, rotation=90, va='bottom', 
                            ha='right', zorder=2)
                axs[1].text(model_load_epoch, y_text_position, f"{model_load_epoch:,}", color='#5C5C5C', fontsize=9, rotation=90, va='bottom', 
                            ha='right', zorder=2)
            
            # add text to the plot to show the number of samples trained on at the end of the last epoch
            plt.text(0.86, 1.02, f"Samples Trained On: {samples_trained.iloc[-1]:,}", fontsize=9, ha='center', va='center',
                    transform=plt.gca().transAxes)
            
            axs[0].legend()
            axs[0].grid(True, alpha=0.4)
            axs[1].legend()
            axs[1].grid(True, alpha=0.4)
            
            plt.suptitle(f"{loss_col.replace('_', ' ').title()} and {lr_col.replace('_', ' ').title()} vs. Epoch", fontsize=18, y=0.96)
            axs[1].set_xlabel("Epoch")
            
            # move "learning_rate" to the front of the filename for better sorting of the plots in the directory
            filename = lr_col.split("_")[0]  # get the name of the optimizer from the column name
            filename = f"learning_rate_{filename}"
            
            # Save the plot to the current epoch checkpoint directory
            plot_path = self.this_epoch_checkpoint_dir.joinpath(f"{filename}.png")
            plt.savefig(plot_path, dpi=100)
            shutil.copy(plot_path, self.model_training_output_dir.joinpath(f"{filename}.png"))
            plt.close()
            plt.clf()
        
        ################################################ create a plot to show memory usage over time ################################################
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
            # add text to show the epoch after the model was loaded
            y_text_position = ((ax1.get_ylim()[1] - ax1.get_ylim()[0]) * 0.02) + ax1.get_ylim()[0]
            ax1.text(model_load_epoch, y_text_position, f"{model_load_epoch:,}", color='#5C5C5C', fontsize=9, rotation=90, va='bottom', ha='right',
                    zorder=2)
        
        # add text to the plot to show the number of samples trained on at the end of the last epoch
        plt.text(0.86, 1.02, f"Samples Trained On: {samples_trained.iloc[-1]:,}", fontsize=9, ha='center', va='center', 
                transform=plt.gca().transAxes)
        
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Memory Usage (GB)")
        ax1.legend()
        ax1.grid(True, alpha=0.4)
        plt.title("Memory Usage vs. Epoch")
        plt.tight_layout()
        
        # Save the memory usage plot to the model training output directory
        memory_plot_path = self.model_training_output_dir.joinpath("memory_usage.png")
        plt.savefig(memory_plot_path, dpi=100)
        plt.close()
        plt.clf()
        
        ############################################## create a plot to show GPU memory usage over time ##############################################
        fig, ax1 = plt.subplots(figsize=(10, 6))
        # Plot the GPU memory usage in GB
        ax1.plot(epochs, self.metrics_dataframe["gpu_memory_usage_gb"], label="GPU Memory Usage (GB)", zorder=1)
        
        for i, model_load_epoch in enumerate(model_load_indices):
            if i == 0:
                line_label = "ckpt loaded"
            else:
                line_label = ""
            ax1.axvline(x=model_load_epoch, color='#5C5C5C', linestyle='--', alpha=0.7, linewidth=1.5, label=line_label, zorder=-1)
            # add text to show the epoch after the model was loaded
            y_text_position = ((ax1.get_ylim()[1] - ax1.get_ylim()[0]) * 0.02) + ax1.get_ylim()[0]
            ax1.text(model_load_epoch, y_text_position, f"{model_load_epoch:,}", color='#5C5C5C', fontsize=9, rotation=90, va='bottom', ha='right',
                    zorder=2)
        
        # add text to the plot to show the number of samples trained on at the end of the last epoch
        plt.text(0.86, 1.02, f"Samples Trained On: {samples_trained.iloc[-1]:,}", fontsize=9, ha='center', va='center', 
                transform=plt.gca().transAxes)
        
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("GPU Memory Usage (GB)")
        ax1.legend()
        ax1.grid(True, alpha=0.4)
        plt.title("GPU Memory Usage vs. Epoch")
        plt.tight_layout()
        
        # Save the GPU memory usage plot to the model training output directory
        plot_path = self.model_training_output_dir.joinpath("memory_usage_GPU.png")
        plt.savefig(plot_path, dpi=100)
        plt.close()
        plt.clf()
        return
    
    def learning_rate_scheduler(self):
        # if we have passed the warmup period, start to decay the learning rates
        if self.current_epoch > self.model.learning_rate_warmup_epochs:
            # decay the learning rates by the learning_rate_decay factor using wgan-gp attributes so that we don't run into errors with floating
            # point rounding errors that slightly change the values when extracting from self.model.critic_optimizer.learning_rate.numpy() and
            # self.model.gen_optimizer.learning_rate.numpy().
            self.model.critic_learning_rate = self.model.critic_learning_rate * self.model.learning_rate_decay
            self.model.generator_learning_rate = self.model.generator_learning_rate * self.model.learning_rate_decay
            
            # if the learning rate is less than the minimum learning rate, set it to the minimum learning rate
            if self.model.critic_learning_rate < self.model.min_critic_learning_rate:
                self.model.critic_learning_rate = self.model.min_critic_learning_rate
            if self.model.generator_learning_rate < self.model.min_generator_learning_rate:
                self.model.generator_learning_rate = self.model.min_generator_learning_rate
            
            # update the learning rates of the optimizers
            self.model.critic_optimizer.learning_rate = self.model.critic_learning_rate
            self.model.gen_optimizer.learning_rate = self.model.generator_learning_rate
        return
    
    def save_model_checkpoint(self):
        # pass a copy of the checkpoint directory to the model so that it can save the critic and generator models to the correct directory
        self.model.this_epoch_checkpoint_dir = self.this_epoch_checkpoint_dir
        # save a copy of the model to the current epoch checkpoint directory
        model_save_path = self.this_epoch_checkpoint_dir.joinpath("model_save")
        self.model.save(model_save_path, zipped=False)
        return
    
    def generate_validation_samples(self):
        """
        Generates and saves a grid of validation images produced by the generator model, along with their critic scores. The images are saved to the
        current epoch checkpoint directory and a copy is saved to the model training output directory. Using these generated images a video of all 
        saved images is generated every specified number of epochs.
        
        Parameters:
            None
        
        Returns:
            None
        """
        # Generate a set of images (number is determined by num_img defined in the initializer)
        generated_images = self.model.generator([self.random_noise_vectors, self.validation_sample_labels], training=False)
        # Rescale the images from [-1, 1] to [0, 255]
        generated_images = (generated_images * 127.5) + 127.5
        # transformed generated images from the current shape [100, 28, 28, 1], to [10, 10, 28, 28, 1] for easier plotting (hardcoded for MNIST)
        generated_images = tf.reshape(generated_images, (self.validation_sample_grid[0], self.validation_sample_grid[1], 28, 28, 1))
        
        # Plot the generated MNIST images
        fig, ax = plt.subplots(
            self.validation_sample_grid[0], self.validation_sample_grid[1],
            figsize=(self.validation_sample_grid[1], self.validation_sample_grid[0]),
            gridspec_kw={
                'wspace': -0.795,  # left and right
                'hspace': 0.05  # up and down
                }
        )
        
        # Add labels for each row and column
        for row in range(self.validation_sample_grid[0]):
            for col in range(self.validation_sample_grid[1]):
                # Plot the image
                ax[row, col].imshow(generated_images[row, col, :, :, 0])
                ax[row, col].axis('off')
        
            # Add class label on the left
            ax[row, 0].annotate(f'{row}',
                                xy=(-0.42, 0.5), xycoords='axes fraction',
                                va='center', ha='center', fontsize=25, rotation=0)
        
        # Add column labels on the top
        for col in range(self.validation_sample_grid[1]):
            ax[0, col].annotate(f'Generated\nSample {col+1:>2}',
                                xy=(0.5, 1.05), xycoords='axes fraction',
                                va='bottom', ha='center', fontsize=9.5, rotation=0)
        
        # Add a title to the plot with info about the current epoch and number of samples trained on
        fig.text(x=0.0625, y=0.9625, s=f"Validation Samples", ha='left', fontsize=24)
        fig.text(x=0.409, y=0.9625, s=f"Epoch: {self.current_epoch:}", ha='left', fontsize=24)
        fig.text(x=0.581, y=0.9625, s=f"Samples Trained On: {self.metrics_dataframe['samples_trained_on'].iloc[-1]:,}", ha='left', fontsize=24)
        
        # remove all axes from the plot
        plt.axis('off')
        # remove all box lines from the plot
        plt.box(False)
        
        # Adjust figure to remove extra padding and ensure spacing consistency
        plt.subplots_adjust(left=-0.015, right=1.113, top=0.90, bottom=0.01)
        
        # Save the grid of images to the current epoch checkpoint directory
        fig_save_path = self.this_epoch_checkpoint_dir.joinpath(f"validation_samples.png")
        plt.savefig(fig_save_path, dpi=150, bbox_inches='tight')
        # copy the plot to the model_training_output_dir for easy access to the most recent plot
        fig_copy_path = self.model_training_output_dir.joinpath(f"validation_samples.png")
        shutil.copy(fig_save_path, fig_copy_path)
        
        # Clear the current figure and close the plot to avoid memory leaks
        plt.clf()
        plt.close()
        
        print(f"Validation samples saved to: {fig_copy_path}")
        return
    
    def generate_video_of_validation_samples(self):
        """
        Creates a video of the validation images generated at the end of each epoch
        """
        # Get a sorted list of all validation_samples.png paths
        validation_samples_paths = sorted(
            self.model_checkpoints_dir.glob("epoch_*/validation_samples.png"),
            key=lambda p: int(p.parent.name.split("_")[1])
        )
        
        if not validation_samples_paths:
            print("No validation samples found. Skipping video generation.")
            return
        
        # Read the first image to get its dimensions
        first_image = cv2.imread(validation_samples_paths[0])
        original_height, original_width, _ = first_image.shape
        
        # Set the resizing dimensions (e.g., scale down to 75% of original size)
        resize_factor = 0.50
        resized_width = int(original_width * resize_factor)
        resized_height = int(original_height * resize_factor)
        
        # Define the codec and create a VideoWriter object
        video_save_path = self.this_epoch_checkpoint_dir.joinpath("validation_samples.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for mp4
        video = cv2.VideoWriter(video_save_path, fourcc, 10, (resized_width, resized_height))  # 10 FPS
        
        # Loop through all validation sample images and add them to the video
        for image_path in validation_samples_paths:
            frame = cv2.imread(image_path)
            if frame is not None:
                # Resize the frame
                resized_frame = cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
                video.write(resized_frame)
                video.write(resized_frame)  # Duplicate frame to increase duration
            else:
                print(f"Warning: Unable to read image {image_path}")
        
        # Release the video writer
        video.release()
        
        # copy the generated video to the current epoch checkpoint directory
        video_copy_path = self.model_training_output_dir.joinpath("validation_samples.mp4")
        shutil.copy(video_save_path, video_copy_path)
        
        print(f"Video of validation samples across epochs saved to: {video_copy_path}")
        return
    
    def calculate_FID_score(self):
        """
        Uses the activations of an intermediate layer of the InceptionV3 model to calculate the Frechet Inception Distance (FID) between the real
        images and the generated images. The FID score is a measure of the similarity between two sets of images which can give a good estimate of
        the quality of the generated images compared to the real images. The lower the FID score, the better the generated images are. This can be a
        time-consuming process. 10,000 is the recommended number of samples to use for this calculation, but a smaller number can be used for faster
        computation.
        
        for more info on this metric, 
        see this W&B article: https://wandb.ai/authors/frechet-inception-distance/reports/How-to-Evaluate-GANs--VmlldzoxMjYwMjI 
        see the original paper: https://arxiv.org/abs/1706.08500
        
        Also consider computing precision, recall, F1, ROC, AUC, IS, and Minimum Mean Distance (MMD) scores for a more comprehensive evaluation of the
        generated images.
        """
        ############################################################ calculate FID score #############################################################
        # hardcoded batch size for the FID score calculation
        batch_size = 128
        # hardcoded number of classes for the MNIST dataset
        num_classes = 10
        # number of samples to generate for the FID score calculation
        sample_count = 10_000
        
        # get the mu and sigma values for real images if they haven't been calculated yet
        if self.mu_real is None or self.sigma_real is None:
            print(f"\nCalculating mu and sigma values for real images...\nThese values will be saved to speed up future FID calculations.")
            # get the real images from the MNIST dataset
            samples_real = []
            # get samples from the MNIST dataset to calculate the FID score
            while len(samples_real) < sample_count:
                for real_images, real_labels in self.train_dataset.take(1):
                    for image, label in zip(real_images, real_labels):
                        samples_real.append(image)
            # cap the number of samples to 10,000 and convert the list to a numpy array
            samples_real = np.array(samples_real[:sample_count])
            # preprocess the real samples for the InceptionV3 model
            samples_real = self._preprocess_MNIST_for_FID(samples_real)
            # get the activations for the real samples
            activations_real = self._get_inception_activations_for_FID(samples_real, batch_size=batch_size)
            # calculate the mu and sigma values for the real samples
            self.mu_real = np.mean(activations_real, axis=0)
            self.sigma_real = np.cov(activations_real, rowvar=False)
            # clean up the real samples
            del samples_real, activations_real
        
        # get the mu and sigma values for generated images
        samples_generated = np.array([])
        # generate samples to calculate the FID score
        while samples_generated.shape[0] < sample_count:
            noise = tf.random.normal([batch_size, self.model.latent_dim])
            labels = tf.random.uniform([batch_size], minval=0, maxval=num_classes, dtype=tf.int32)
            batch_generated = self.model.generator([noise, labels], training=False)
            if samples_generated.shape[0] == 0:
                samples_generated = batch_generated
            else:
                samples_generated = np.concatenate((samples_generated, batch_generated), axis=0)
        # cap the number of samples to and convert the list to a numpy array
        samples_generated = np.array(samples_generated[:sample_count])
        # preprocess the generated samples for the InceptionV3 model
        samples_generated = self._preprocess_MNIST_for_FID(samples_generated)
        # get the activations for the generated samples
        activations_generated = self._get_inception_activations_for_FID(samples_generated, batch_size=batch_size)
        # calculate the mu and sigma values for the generated samples
        mu_generated = np.mean(activations_generated, axis=0)
        sigma_generated = np.cov(activations_generated, rowvar=False)
        # clean up the generated samples
        del samples_generated, activations_generated
        
        # calculate the FID score
        fid = self._calculate_fid(self.mu_real, self.sigma_real, mu_generated, sigma_generated)
        
        # update the last row of the metrics dataframe with the correct value for the FID score
        self.metrics_dataframe.iloc[-1, self.metrics_dataframe.columns.get_loc("FID_score")] = fid
        
        ######################################################### create plot for FID score ##########################################################
        fig, ax1 = plt.subplots(figsize=(10, 6))
        epochs = self.metrics_dataframe['epoch'].copy()
        FID_scores = self.metrics_dataframe['FID_score'].copy()
        # drop rows with zero values for plotting
        epochs = epochs[FID_scores != 0]
        FID_scores = FID_scores[FID_scores != 0]
        ax1.plot(epochs, FID_scores, label="FID Score", zorder=1, color='#1F77B4')
        
        # add vertical lines for model loading events with a single label for legend
        model_load_indices = self.metrics_dataframe.loc[self.metrics_dataframe['model_loaded'], 'epoch']
        for i, model_load_epoch in enumerate(model_load_indices):
            if i == 0:
                line_label = "ckpt loaded"
            else:
                line_label = ""
            ax1.axvline(x=model_load_epoch, color='#5C5C5C', linestyle='--', alpha=0.7, linewidth=1.5, label=line_label, zorder=-1)
            # add text to show the epoch after the model was loaded
            y_text_position = ((ax1.get_ylim()[1] - ax1.get_ylim()[0]) * 0.02) + ax1.get_ylim()[0]
            ax1.text(model_load_epoch, y_text_position, f"{model_load_epoch:,}", color='#5C5C5C', fontsize=9, rotation=90, va='bottom', ha='right',
                    zorder=2)
        
        # find the epoch where the minimum FID score occurred
        min_fid_epoch = FID_scores.idxmin() + 1
        min_fid_score = FID_scores.min()
        
        # Plot a vertical line at the epoch with the minimum FID score
        ax1.axvline(x=min_fid_epoch, color='black', linestyle='-', linewidth=1.5, label='Minimum FID', zorder=-1)
        
        # Add text annotation for the minimum line
        y_text_position = ((ax1.get_ylim()[1] - ax1.get_ylim()[0]) * 0.25) + ax1.get_ylim()[0]
        x_text_position = min_fid_epoch + (ax1.get_xlim()[1] - ax1.get_xlim()[0]) * 0.007
        ax1.text(x_text_position, y_text_position, f"  Min FID at Epoch {min_fid_epoch:,}", color='black', fontsize=9, rotation=90,
                va='bottom', ha='left', zorder=2)
        
        # add text to the plot to show the minimum FID score
        plt.text(0.84, 1.02, f"Minimum FID Score: {min_fid_score:.3f} at Epoch {min_fid_epoch}", fontsize=9, ha='center', va='center', transform=plt.gca().transAxes)
        
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("FID Score")
        ax1.legend()
        ax1.grid(True, alpha=0.4)
        plt.title("FID Score vs. Epoch", fontsize=18, loc='left')
        plt.tight_layout()
        
        # Save the FID score plot to the current epoch checkpoint directory
        plot_path = self.this_epoch_checkpoint_dir.joinpath("FID_score.png")
        plt.savefig(plot_path, dpi=100)
        shutil.copy(plot_path, self.model_training_output_dir.joinpath("FID_score.png"))
        plt.close()
        plt.clf()
        return
    
    def _preprocess_MNIST_for_FID(self, images):
        """
        Convert grayscale images to RGB and resize them to 299x299 for the InceptionV3 model so we can calculate the FID score.
        
        args:
            images: a numpy array of MNIST images with shape (num_images, height, width, channels) with 1 channel (grayscale)
        
        returns:
            preprocessed_images: a numpy array of RGB images with shape (num_images, 299, 299, 3)
        """
        # Convert grayscale images to RGB by repeating the single channel 3 times
        preprocessed_images = np.repeat(images, 3, axis=-1)
        # Resize images to 299x299
        preprocessed_images = np.array([keras.preprocessing.image.smart_resize(image, (299, 299)) for image in preprocessed_images])
        # use built-in preprocessing for InceptionV3 model
        preprocessed_images = keras.applications.inception_v3.preprocess_input(preprocessed_images)
        return preprocessed_images
    
    def _get_inception_activations_for_FID(self, images, batch_size=256):
        """
        Get the activations of the InceptionV3 model for a set of images.
        
        args:
            images: a numpy array of RGB images with shape (num_images, 299, 299, 3)
            batch_size: the batch size to use for calculating the activations
        
        returns:
            activations: a numpy array of the activations for the images with shape (num_images, 2048)
        """
        activations = []
        n_images = images.shape[0]
        for start in range(0, n_images, batch_size):
            end = min(start + batch_size, n_images)
            batch = images[start:end]
            activations.append(self.inception_model.predict(batch, verbose=0))
        activations = np.concatenate(activations, axis=0)
        return activations
    
    def _calculate_fid(self, mu1, sigma1, mu2, sigma2):
        """
        Compute the Fréchet Inception Distance (FID) between two multivariate Gaussians 
        with means mu1, mu2 and covariances sigma1, sigma2:
        
            FID = ||mu1 - mu2||^2 + Tr(sigma1 + sigma2 - 2*sqrt(sigma1*sigma2))
        """
        # Difference of means
        diff = mu1 - mu2
        diff_squared = diff.dot(diff)
        
        # Product of covariances (sqrtm)
        covmean, _ = scipy.linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        
        # Numerical stability: if imaginary components remain, take real part
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        
        fid = diff_squared + np.trace(sigma1 + sigma2 - 2.0 * covmean)
        return fid
    
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
        csv_save_path = self.this_epoch_checkpoint_dir.joinpath("training_metrics.csv")
        self.metrics_dataframe.to_csv(csv_save_path, index=False)
        
        ######################################### create plots for training and metric calculation durations #########################################
        # Plot the time metric starting after the first epoch
        fig, ax1 = plt.subplots(figsize=(10, 6))
        epochs = self.metrics_dataframe['epoch']
        ax1.bar(epochs[:], self.metrics_dataframe["epoch_train_time"], label="epoch_train_time".replace("_", " ").title(), zorder=1, 
                color='#3B9DE3', edgecolor='black')
        ax1.bar(epochs[:], self.metrics_dataframe["metric_calc_time"], label="metric_calc_time".replace("_", " ").title(), zorder=1,
                color='#AE7E6F', edgecolor='black', bottom=self.metrics_dataframe["epoch_train_time"])
        
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Duration (s)")
        ax1.legend()
        ax1.grid(True, alpha=0.4)
        plt.title(f"Time Breakdown Across Epochs")
        plt.tight_layout()
        
        filename = f"time_breakdown_across_epochs.png"
        plot_path = self.model_training_output_dir.joinpath(filename)
        plt.savefig(plot_path, dpi=100)
        plt.close()
        plt.clf()
        
        # if this is the first epoch, skip estimating the time to train to specific epochs because we don't have enough data and the first epoch
        # often takes longer due to some first epoch setup time.
        if self.current_epoch == 1:
            return
        
        # get the time spent on every epoch and replace the first value with the second value
        iteration_times = self.metrics_dataframe["total_iteration_time"].copy()  # (time spent training + time spent logging metrics)
        iteration_times.iloc[0] = iteration_times.iloc[1]
        
        # get the total, average, and standard deviation of the time spent on each epoch
        total_iteration_time = iteration_times.sum()
        avg_iteration_time = iteration_times.mean()
        stdev_iteration_time = iteration_times.std()
        # write to a file the time it took to train to a milestone or the estimated time to train to a milestone
        with open(self.model_training_output_dir.joinpath("_milestones_and_ETAs.txt"), "w") as file:
            file.write("Epoch Milestones and Estimated Training Times\n")
            file.write(f"{'='*72}\n")
            # convert to a readable format (HHH:MM:SS.ss)
            total_time = get_readable_time_string(total_iteration_time)
            avg_time = get_readable_time_string(avg_iteration_time)
            stdev_time = get_readable_time_string(stdev_iteration_time)
            file.write(f"Current Total Train Time (HHH:MM:SS.ss): {total_time}\n")
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
                if self.current_epoch >= milestone:
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


if __name__ == "__main__":
    for num in range(10):
        print("This script contains the ModelAndCallbacks class for training a GAN model and is not intended to be run directly.")
        time.sleep(0.1)
