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
        optimizer_g: optim,
        optimizer_d: optim,
        criterion_g,
        criterion_d,
        generator,
        discriminator,
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
        self.optimizer_g = optimizer_g
        self.optimizer_d = optimizer_d
        self.criterion_g = criterion_g
        self.criterion_d = criterion_d
        self.generator_copy = generator
        self.discriminator_copy = discriminator
        self.generator = generator.to(device=self.device)
        self.discriminator = discriminator.to(device=self.device)
        self.dataset = dataset
        self.hist = None

        # getting the right file name
        destination_folder = os.path.abspath(get_base_path() + "src/model/models")

        self.generator_filepath = os.path.join(destination_folder, "generator.hdf5")
        self.discriminator_filepath = os.path.join(destination_folder, "discriminator.hdf5")

    def train(self, epochs: int, verbosity_interval: int = 1):
        """
        The training loop for the net
        :param epochs: the number of epochs the training loop will run
        :param semi_supervised_interval: the interval at which there will be
                semi-supervised learning (in batches)
        :param verbosity_interval: at which epoch interval there will be logging
        """
        writer = SummaryWriter()

        self.hist = np.zeros((epochs, 2))
        start_time = time.time()

        batch_time_avg = np.array([])

        self.generator.train()
        self.discriminator.train()

        semi_supervised_interval = 100
        semi_supervised_times = int(self.dataset.num_train_batches / semi_supervised_interval)

        for epoch in range(1, epochs + 1):
            with torch.set_grad_enabled(True):

                log_error_g = 0
                log_error_d = 0
                log_mean_g = 0

                hidden_g = self.generator.init_hidden(self.dataset.x_train.shape[1])
                hidden_d = self.discriminator.init_hidden(self.dataset.x_train.shape[1])

                for batch in range(self.dataset.num_train_batches):

                    batch_start_time = time.time()

                    # getting the batch data from the dataset
                    x_train, y_train = self.dataset.get_train(batch)

                    ###########################################################################
                    # Update the discriminator network: maximize log(D(x)) + log(1 - D(G(z))) #
                    ###########################################################################

                    fake, hidden_g = self.generator(
                        x_train.reshape(x_train.shape[0], 1, 1), hidden_g.detach()
                    )

                    split = int(y_train.shape[0] * 0.5)

                    # Train
                    hidden_d_copy = torch.clone(hidden_d.detach())
                    input_temp = None
                    if batch % 2 == 0:
                        input_temp = torch.cat(
                            (
                                y_train.reshape(y_train.shape[0], 1, 1)[:split],
                                fake.detach().reshape(fake.shape[0], 1, 1)[split:],
                            )
                        )
                    else:
                        input_temp = torch.cat(
                            (
                                fake.detach().reshape(fake.shape[0], 1, 1)[:split],
                                y_train.reshape(y_train.shape[0], 1, 1)[split:],
                            )
                        )

                    disc, hidden_d = self.discriminator(
                        input_temp,
                        hidden_d.detach(),
                    )

                    if batch % 2 == 0:
                        input_temp = torch.cat(
                            (
                                torch.zeros_like(disc[:split]),
                                torch.ones_like(disc[split:]),
                            )
                        )
                    else:
                        input_temp = torch.cat(
                            (
                                torch.ones_like(disc[split:]),
                                torch.zeros_like(disc[:split]),
                            )
                        )

                    error_d = self.criterion_d(disc, input_temp)
                    log_error_d += error_d.mean().item()

                    self.discriminator.zero_grad()
                    error_d.backward()
                    self.optimizer_d.step()

                    #######################################################
                    # Update the generator network: maximize log(D(G(z))) #
                    #######################################################

                    gen, _ = self.discriminator(fake.reshape(fake.shape[0], 1, 1), hidden_d_copy)
                    error_g = self.criterion_d(gen, torch.zeros_like(gen))
                    log_mean_g += gen.mean().item()
                    log_error_g += error_g.mean().item()

                    self.generator.zero_grad()
                    error_g.backward()
                    self.optimizer_g.step()

                    # updating progress
                    batch_training_time = time.time() - batch_start_time
                    batch_time_avg = np.append(batch_time_avg, batch_training_time)
                    total = batch_time_avg.mean() * (
                        (epochs - epoch) * self.dataset.num_train_batches
                        + (self.dataset.num_train_batches - batch)
                    )
                    hours = int(total / 3600)
                    minutes = int((total % 3600) / 60)
                    seconds = int(total % 60)
                    progress_bar(
                        batch,
                        self.dataset.num_train_batches,
                        f"epoch {str(epoch)}/{str(epochs)}, remaining time: "
                        + f"{str(hours)}h{str(minutes)}m{str(seconds)}s",
                    )

            # getting the averages
            avg_divisor = self.dataset.num_train_batches - semi_supervised_times

            log_error_g /= avg_divisor
            log_error_d /= avg_divisor
            log_mean_g /= avg_divisor

            self.hist[epoch - 1] = np.array([log_error_g, log_error_d])

            # logging losses
            writer.add_scalar("Generator Error", log_error_g, epoch)
            writer.add_scalar("Discriminator Error", log_error_d, epoch)
            writer.add_scalar("Generator average", log_mean_g, epoch)
            if epoch == 1 or epoch % verbosity_interval == 0:
                print(
                    f"Epoch {epoch}, Generator Error: {log_error_g}, "
                    + f"Discriminator Error: {log_error_d}, G: {log_mean_g}"
                )
        training_time = time.time() - start_time
        print(f"Training time: {training_time}")
        torch.cuda.empty_cache()

    def evaluate(self):
        """
        This function will evaluate the model and print the MSE loss to console
        """

        self.generator.eval()
        self.discriminator.eval()

        scaled_mse = 0
        with torch.set_grad_enabled(True):

            hidden_g = self.generator.init_hidden(self.dataset.x_test.shape[1])

            for batch in range(self.dataset.num_test_batches):

                # getting the batch data from the dataset
                x_test, y_unscaled_test = self.dataset.get_test(batch)

                output, hidden_g = self.generator(
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
        if os.path.exists(self.generator_filepath):
            os.remove(self.generator_filepath)
        if os.path.exists(self.discriminator_filepath):
            os.remove(self.discriminator_filepath)
        torch.save(self.generator.state_dict(), self.generator_filepath)
        torch.save(self.discriminator.state_dict(), self.discriminator_filepath)

    def load(self):
        """
        This method will load and return a model
        """
        self.generator = self.generator_copy.to(self.device)
        self.generator.load_state_dict(torch.load(self.generator_filepath))
        self.discriminator = self.discriminator_copy.to(self.device)
        self.discriminator.load_state_dict(torch.load(self.discriminator_filepath))
