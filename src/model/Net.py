# Copyright (c) 2022 Alix Routhier-Lalonde. Licence included in root of package.

"""
Contains the Net class, a class for training a model.
"""

import time
import os
from torch.optim import optimizer as optim
from sklearn.metrics import mean_squared_error
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from src.data import AggregateDataset
from src.utils import get_base_path, progress_bar


class Net:
    """
    The Net class will build the model and train it.
    """

    def __init__(
        self,
        device,
        optimizer: optim,
        criterion,
        model,
        dataset: AggregateDataset,
    ):
        """
        The __init__ function will set all training parameters and generate the model
        :param device: the device on which the model and data will be on (cpu vs cuda)
        :param optimizer_g1: the training optimizer for the generator
        :param optimizer_g2: the training optimizer for the generator for semi-supervised learning
        :param optimizer_d: the training optimizer for the discriminator
        :param criterion_g: the training loss function for the generator
        :param criterion_d: the training loss function for the discriminator
        :param generator: the pytorch generator model
        :param discriminator: the pytorch discriminator model
        :param dataset: the dataset to train on
        """
        self.device = device
        self.optimizer = optimizer
        self.criterion = criterion
        self.model = model.to(device=self.device)
        self.dataset = dataset

        # getting the right file name
        destination_folder = os.path.abspath(get_base_path() + "src/model/models")

        self.model_filepath = os.path.join(destination_folder, "model.hdf5")

    def train(self, epochs: int, verbosity_interval: int = 1):
        """
        The training loop for the net
        :param epochs: the number of epochs the training loop will run
        :param semi_supervised_interval: the interval at which there will be
                semi-supervised learning (in batches)
        :param verbosity_interval: at which epoch interval there will be logging
        """
        writer = SummaryWriter()

        start_time = time.time()

        batch_time_avg = np.array([])

        lowest_validation_error = 10000000000

        for epoch in range(1, epochs + 1):

            log_error_train = 0
            log_error_validation = 0

            ##################
            # model training #
            ##################

            hidden = self.model.init_hidden(self.dataset.x_train.shape[1])

            self.model.train()
            with torch.set_grad_enabled(True):
                for batch in range(self.dataset.num_train_batches):

                    batch_start_time = time.time()

                    # getting the batch data from the dataset
                    x_train, y_train = self.dataset.get_train(batch)

                    output, hidden = self.model(x_train, hidden.detach())
                    error = self.criterion(output, y_train.unsqueeze(1))
                    log_error_train += error.mean().item()

                    self.model.zero_grad()
                    error.backward()
                    self.optimizer.step()

                    # updating progress
                    batch_training_time = time.time() - batch_start_time
                    batch_time_avg = np.append(batch_time_avg, batch_training_time)
                    total = batch_time_avg.mean() * (
                        (epochs - epoch)
                        * (self.dataset.num_train_batches + self.dataset.num_validation_batches)
                        + (self.dataset.num_train_batches - batch)
                        + self.dataset.num_validation_batches
                    )
                    hours = int(total / 3600)
                    minutes = int((total % 3600) / 60)
                    seconds = int(total % 60)
                    progress_bar(
                        batch,
                        self.dataset.num_train_batches + self.dataset.num_validation_batches,
                        f"epoch {str(epoch)}/{str(epochs)}, remaining time: "
                        + f"{str(hours)}h{str(minutes)}m{str(seconds)}s",
                    )

            ####################
            # model validation #
            ####################

            hidden = self.model.init_hidden(self.dataset.x_validation.shape[1])

            self.model.eval()
            with torch.set_grad_enabled(False):
                for batch in range(self.dataset.num_validation_batches):

                    batch_start_time = time.time()

                    # getting the batch data from the dataset
                    x_validation, y_validation = self.dataset.get_validation(batch)

                    output, hidden = self.model(x_validation, hidden.detach())

                    error = self.criterion(output, y_validation.unsqueeze(1))
                    log_error_validation += error.mean().item()

                    self.model.zero_grad()

                    # updating progress
                    batch_training_time = time.time() - batch_start_time
                    batch_time_avg = np.append(batch_time_avg, batch_training_time)
                    total = batch_time_avg.mean() * (
                        (epochs - epoch)
                        * (self.dataset.num_train_batches + self.dataset.num_validation_batches)
                        + self.dataset.num_validation_batches
                    )
                    hours = int(total / 3600)
                    minutes = int((total % 3600) / 60)
                    seconds = int(total % 60)
                    progress_bar(
                        self.dataset.num_train_batches + batch,
                        self.dataset.num_train_batches + self.dataset.num_validation_batches,
                        f"epoch {str(epoch)}/{str(epochs)}, remaining time: "
                        + f"{str(hours)}h{str(minutes)}m{str(seconds)}s",
                    )

            log_error_train /= self.dataset.num_train_batches
            log_error_validation /= self.dataset.num_validation_batches

            if log_error_validation < lowest_validation_error:
                lowest_validation_error = log_error_validation
                self.save()

            # logging losses
            writer.add_scalar("Training Error", log_error_train, epoch)
            writer.add_scalar("Validation Error", log_error_validation, epoch)
            if epoch == 1 or epoch % verbosity_interval == 0:
                print(
                    f"Epoch {epoch}, Training Error: {log_error_train}, "
                    + f"Validation Error: {log_error_validation}"
                )
        training_time = time.time() - start_time
        print(f"Training time: {training_time}")
        self.load()
        torch.cuda.empty_cache()

    def evaluate(self):
        """
        This function will evaluate the model and print the MSE loss to console
        """

        self.model.eval()

        scaled_mse = 0
        with torch.set_grad_enabled(True):

            hidden_g = self.model.init_hidden(self.dataset.x_test.shape[1])

            for batch in range(self.dataset.num_test_batches):

                # getting the batch data from the dataset
                x_test, y_unscaled_test = self.dataset.get_test(batch)

                output, hidden_g = self.model(
                    x_test.reshape(x_test.shape[0], 1, 1), hidden_g.detach()
                )

                scaled_mse += mean_squared_error(
                    output.squeeze().detach().cpu().numpy(),
                    y_unscaled_test.squeeze().detach().cpu().numpy(),
                )

        print(f"scaled_mse_y: {scaled_mse/self.dataset.num_test_batches}")

    def save(self):
        """
        This method will save the trained model according to the dataset's code(s) and the
        current date
        """
        if os.path.exists(self.model_filepath):
            os.remove(self.model_filepath)
        torch.save(self.model.state_dict(), self.model_filepath)

    def load(self):
        """
        This method will load and return a model
        """
        self.model = self.model.to(self.device)
        self.model.load_state_dict(torch.load(self.model_filepath))
