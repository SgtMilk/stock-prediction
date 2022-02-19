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
        :param optimizer_g: the training optimizer for the generator
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

        self.generator_filepath = os.path.join(
            destination_folder, f"generator-{dataset.interval}.hdf5"
        )
        self.discriminator_filepath = os.path.join(
            destination_folder, f"discriminator-{dataset.interval}.hdf5"
        )

    def train(self, epochs: int, verbosity_interval: int = 1):
        """
        The training loop for the net
        :param epochs: the number of epochs the training loop will run
        :param verbosity_interval: at which epoch interval there will be logging
        """
        writer = SummaryWriter()

        self.hist = np.zeros((epochs, 2))
        start_time = time.time()

        error_g = error_d = None

        self.generator.train()
        self.discriminator.train()

        for epoch in range(1, epochs + 1):
            with torch.set_grad_enabled(True):

                error_g = 0
                error_d = 0

                d_x = 0
                d_g1 = 0
                d_g2 = 0

                for batch in range(self.dataset.num_train_batches):
                    # getting the batch data from the dataset
                    x_train, y_train = self.dataset.get_train(batch)

                    labels_size_d = y_train.size(0)
                    real_labels = torch.full(
                        (labels_size_d,), 0, dtype=torch.float, device=self.device
                    )
                    fake_labels = torch.full(
                        (labels_size_d,), 1, dtype=torch.float, device=self.device
                    )

                    ####################################
                    # Update the discriminator network #
                    ####################################
                    self.optimizer_g.zero_grad()
                    self.optimizer_d.zero_grad()

                    # Train with all-real batch
                    output_real_d = self.discriminator(torch.cat((x_train, y_train), 1)).view(-1)
                    error_real_d = self.criterion_d(output_real_d, real_labels)
                    error_real_d.backward()
                    d_x += output_real_d.mean().item()

                    self.optimizer_d.step()
                    self.optimizer_d.zero_grad()

                    # Train with all-fake (noise) batch
                    fake = self.generator(x_train)
                    output_g = fake
                    input_d = torch.cat((x_train, fake.detach()), 1)
                    output_fake_d = self.discriminator(input_d).view(-1)

                    error_fake_g = self.criterion_g(fake, y_train)
                    error_fake_g.backward()
                    error_fake_d = self.criterion_d(output_fake_d, fake_labels)
                    error_fake_d.backward()
                    d_g1 += output_fake_d.mean().item()
                    d_g2 += output_g.mean().item()

                    error_g += error_fake_g.item()
                    error_d += error_real_d.item() + error_fake_d.item()

                    self.optimizer_g.step()
                    self.optimizer_d.step()

                    progress_bar(batch, self.dataset.batch_div, f"epoch {str(epoch)}/{str(epochs)}")

            # getting the averages
            error_g /= self.dataset.batch_div
            error_d /= self.dataset.batch_div
            d_x /= self.dataset.batch_div
            d_g1 /= self.dataset.batch_div
            d_g2 /= self.dataset.batch_div

            self.hist[epoch - 1] = np.array([error_d, error_g])

            # logging losses
            writer.add_scalar("Error/generator", error_g, epoch)
            writer.add_scalar("Error/discriminator", error_d, epoch)
            if epoch == 1 or epoch % verbosity_interval == 0:
                print(
                    f"Epoch {epoch}, Generator Error: {error_g}, "
                    + f"Discriminator Error: {error_g}, Dx: {d_x}, Dg1: {d_g1}, Dg2: {d_g2}"
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

        predicted_y_test = y_test = y_unscaled_test = None
        with torch.set_grad_enabled(False):
            for batch in range(self.dataset.num_test_batches):
                if batch == 0:
                    x_test, y_test, y_unscaled_test = self.dataset.get_test(batch)
                    predicted_y_test = self.generator(x_test)
                    predicted_y_test = torch.reshape(
                        predicted_y_test, (predicted_y_test.shape[0], predicted_y_test.shape[1])
                    )

                    y_test = y_test.detach().cpu().numpy()
                    y_unscaled_test = y_unscaled_test.detach().cpu().numpy()
                    predicted_y_test = predicted_y_test.detach().cpu().numpy()
                else:
                    x_temp, y_temp, y_unscaled_temp = self.dataset.get_test(batch)
                    predicted_y_temp = self.generator(x_temp)
                    predicted_y_temp = torch.reshape(
                        predicted_y_temp, (predicted_y_temp.shape[0], predicted_y_temp.shape[1])
                    )
                    y_test = np.concatenate((y_test, y_temp.detach().cpu().numpy()))
                    y_unscaled_test = np.concatenate(
                        (y_unscaled_test, y_unscaled_temp.detach().cpu().numpy())
                    )
                    predicted_y_test = np.concatenate(
                        (predicted_y_test, predicted_y_temp.detach().cpu().numpy())
                    )

        unscaled_predicted = self.dataset.inverse_transform(predicted_y_test)

        assert predicted_y_test.shape == unscaled_predicted.shape
        assert predicted_y_test.shape == y_unscaled_test.shape

        scaled_mse = mean_squared_error(np.squeeze(y_test), predicted_y_test)
        print(f"scaled_mse_y: {scaled_mse}")

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
