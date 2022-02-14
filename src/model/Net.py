# Copyright (c) 2022 Alix Routhier-Lalonde. Licence included in root of package.

from typing import Union
import time
import os
from torch.optim import optimizer as optim
from sklearn.metrics import mean_squared_error
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
from src.data import Dataset, AggregateDataset
from src.utils import get_base_path



class Net:
    """
    The Net class will build the model and train it.
    """

    def __init__(self, device, optimizer_g: optim, optimizer_d: optim, criterion, generator, discriminator, dataset: Union[Dataset, AggregateDataset]):
        """
        The __init__ function will set all training parameters and generate the model
        :param optimizer_g: the training optimizer for the generator
        :param optimizer_d: the training optimizer for the discriminator
        :param criterion: the training loss function
        :param generator: the pytorch generator model
        :param discriminator: the pytorch discriminator model
        :param dataset: the dataset to train on
        """
        self.device = device
        self.optimizer_g = optimizer_g
        self.optimizer_d = optimizer_d
        self.criterion = criterion
        self.generator = generator.to(device=self.device)
        self.discriminator = discriminator.to(device=self.device)
        self.dataset = dataset

        self.weights_train = None
        self.loss_train = None
        self.loss_validation = None
        self.hist = None

        # getting the right file name
        destination_folder = os.path.abspath(
            get_base_path() + 'src/model/models')

        self.generator_filepath = os.path.join(destination_folder, f"generator-{dataset.interval}.hdf5")
        self.discriminator_filepath = os.path.join(destination_folder, f"discriminator-{dataset.interval}.hdf5")


    def train(self, epochs: int, patience: int, verbosity_interval: int = 1):
        """
        The training loop for the net
        :param patience: the number of epochs the validation loss doesn't improve before we stop training the model
        :param epochs: the number of epochs the training loop will run
        :param dataset: the dataset
        :param validation_split: the split between validation and training data
        :param verbosity_interval: at which epoch interval there will be logging
        """
        writer = SummaryWriter()

        self.hist = np.zeros((epochs, 2))
        start_time = time.time()

        # lowest_validation_loss = None
        error_g = error_d = None
        # patience_counter = 0


        self.generator.train()
        self.discriminator.train()

        for epoch in range(1, epochs + 1):
            # finishing the training if the patience doesn't improve
            # if patience_counter >= patience:
            #     break
            # patience_counter += 1
            with torch.set_grad_enabled(True):

                error_g = 0
                error_d = 0

                d_x = 0
                d_g1 = 0
                d_g2 = 0

                for batch in range(self.dataset.batch_div):     
                    # getting the batch data from the dataset
                    x_train, y_train = self.dataset.get_train(batch)

                    labels_size_d = y_train.size(0)
                    real_labels = torch.full((labels_size_d,), 1, dtype=torch.float, device=self.device)
                    fake_labels = torch.full((labels_size_d,), 0, dtype=torch.float, device=self.device)

                    ####################################
                    # Update the discriminator network #
                    ####################################

                    # Train with all-real batch
                    self.discriminator.zero_grad()
                    output_real_d = self.discriminator(y_train).view(-1)
                    error_real_d = self.criterion(output_real_d, real_labels)
                    error_real_d.backward()
                    d_x += output_real_d.mean().item()

                    # Train with all-fake (noise) batch
                    noise = torch.randn(labels_size_d, x_train.shape[1], 1, device=self.device)
                    fake = self.generator(noise)
                    output_fake_d = self.discriminator(fake.detach()).view(-1)
                    error_fake_d = self.criterion(output_fake_d, fake_labels)
                    error_fake_d.backward()
                    d_g1 += output_fake_d.mean().item()

                    error_d += error_real_d.item() + error_fake_d.item()
                    self.optimizer_d.step()

                    ################################
                    # Update the generator network #
                    ################################

                    self.generator.zero_grad()
                    output_g = self.discriminator(fake).view(-1)
                    error_temp_g = self.criterion(output_g, real_labels)
                    error_temp_g.backward()
                    error_g += error_temp_g.item()
                    d_g2 += output_g.mean().item()
                    self.optimizer_g.step()

            # getting the averages
            error_g /= self.dataset.batch_div
            error_d /= self.dataset.batch_div
            d_x /= self.dataset.batch_div
            d_g1 /= self.dataset.batch_div
            d_g2 /= self.dataset.batch_div

            self.hist[epoch - 1] = np.array([error_d, error_g])

            # if lowest_validation_loss is None or lowest_validation_loss > self.loss_validation:
            #     self.save()
            #     lowest_validation_loss = self.loss_validation

            # logging losses
            writer.add_scalar('Error/generator', error_d, epoch)
            writer.add_scalar('Error/discriminator', error_g, epoch)
            if epoch == 1 or epoch % verbosity_interval == 0:
                print(f"Epoch {epoch}, Generator Error: {error_g}, " +
                      f"Discriminator Error: {error_g}, Dx: {d_x}, Dg1: {d_g1}, Dg2: {d_g2}")
        # self.load()
        training_time = time.time() - start_time
        print(f"Training time: {training_time}")

    def evaluate_training(self):
        """
        This function will plot the training and validation losses
        """
        plt.gcf().set_size_inches(22, 15, forward=True)

        plt.plot([value[0] for value in self.hist], label='training loss')
        plt.plot([value[1] for value in self.hist], label='validation loss')

        plt.legend(['Training Loss', 'Validation Loss'])

        plt.show()

    def evaluate(self, dataset: Union[Dataset, AggregateDataset]):
        """
        This function will evaluate the model and plot the results
        :param dataset: the dataset to evaluate
        """
        x_test, y_test, y_unscaled_test = dataset.get_test()
        predicted_y_test = np.squeeze(self.generator(x_test.unsqueeze(-1)))

        # re-transforming to numpy
        predicted_y_test = predicted_y_test.detach().cpu().numpy()
        y_test = y_test.detach().cpu().numpy()
        y_unscaled = y_unscaled_test.detach().cpu().numpy()

        unscaled_predicted = dataset.inverse_transform(predicted_y_test)

        assert predicted_y_test.shape == unscaled_predicted.shape
        assert predicted_y_test.shape == y_unscaled.shape

        scaled_mse = mean_squared_error(np.squeeze(y_test), predicted_y_test)
        print(f"scaled_mse_y: {scaled_mse}")

        plt.gcf().set_size_inches(22, 15, forward=True)

        plt.plot(np.squeeze(y_test), label='real', marker='o')
        plt.plot(predicted_y_test, label='predicted', marker='o')

        plt.legend(['Real', 'Predicted'])

        plt.show()

    def save(self):
        """
        This method will save the trained model according to the dataset's code(s) and the current date
        """
        torch.save(self.generator.state_dict(), self.generator_filepath)
        torch.save(self.discriminator.state_dict(), self.discriminator_filepath)

    def load(self):
        """
        This method will load and return a model
        """
        self.generator = torch.load(self.generator_filepath)
        self.discriminator = torch.load(self.discriminator_filepath)
