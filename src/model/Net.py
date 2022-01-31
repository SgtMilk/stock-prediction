# Copyright (c) 2021 Alix Routhier-Lalonde. Licence included in root of package.

from torch.optim import optimizer as optim
from sklearn.metrics import mean_squared_error
import torch
from src.data import Dataset, AggregateDataset
from torch.utils.tensorboard import SummaryWriter
from src.utils import get_base_path
from typing import Union
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import datetime


class Net:
    """
    The Net class will build the model and train it.
    """

    def __init__(self, optimizer: optim, loss_func, model, dataset: Union[Dataset, AggregateDataset]):
        """
        The __init__ function will set all training parameters and generate the model
        :param optimizer: the training optimizer
        :param loss_func: the training loss function
        :param model: the pytorch model
        """
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.model = model
        gpu = torch.cuda.is_available()
        if gpu:
            self.model = self.model.to(device='cuda')

        self.weights_train = None
        self.loss_train = None
        self.loss_validation = None
        self.hist = None

        # getting the right file name
        destination_folder = os.path.abspath(
            get_base_path() + 'src/model/models')
        condition = False
        try:
            dataset.code
        except AttributeError:
            condition = True

        current_date = str(datetime.date.today())

        if condition and len(dataset.datasets) != 1:
            self.filepath = os.path.join(
                destination_folder, f"model-{dataset.interval}.hdf5")
        else:
            if condition:
                code_string = dataset.datasets[0].code
            else:
                code_string = dataset.code
            self.filepath = os.path.join(
                destination_folder, f"{code_string}-{dataset.interval}-{current_date}.hdf5")

    def train(self, epochs: int, dataset: Union[Dataset, AggregateDataset], validation_split: float, patience: int,
              verbosity_interval: int = 1):
        """
        The training loop for the net
        :param patience: the number of epochs the validation loss doesn't improve before we stop training the model
        :param epochs: the number of epochs the training loop will run
        :param dataset: the dataset
        :param validation_split: the split between validation and training data
        :param verbosity_interval: at which epoch interval there will be logging
        """
        writer = SummaryWriter()
        x, y = dataset.get_train()
        n = int(x.shape[0] * (1 - validation_split))
        x_train, y_train = x[:n], y[:n]
        x_validation, y_validation = x[n:], y[n:]

        self.hist = np.zeros((epochs, 2))
        start_time = time.time()

        lowest_validation_loss = None
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            # finishing the training if the patience doesn't improve
            if patience_counter >= patience:
                break
            patience += 1

            # training
            self.optimizer.zero_grad()
            self.model.train()
            with torch.set_grad_enabled(True):
                y_predicted_train = self.model(x_train)
                self.loss_train = self.loss_func(y_predicted_train, y_train)

                self.loss_train.backward()
                self.optimizer.step()

            # validation
            self.optimizer.zero_grad()
            self.model.eval()
            with torch.set_grad_enabled(False):
                y_predicted_validation = self.model(x_validation)
                self.loss_validation = self.loss_func(y_predicted_validation, y_validation)

            self.hist[epoch - 1] = np.array([self.loss_train.cpu().detach().numpy(), self.loss_validation.cpu().detach().numpy()])

            if lowest_validation_loss is None or lowest_validation_loss > self.loss_validation:
                self.save()
                lowest_validation_loss = self.loss_validation

            # logging losses
            writer.add_scalar('Loss/train', self.loss_train, epoch)
            writer.add_scalar('Loss/validation', self.loss_validation, epoch)
            if epoch == 1 or epoch % verbosity_interval == 0:
                print(f"Epoch {epoch}, Training Loss: {self.loss_train.item()}, " +
                      f"Validation Loss: {self.loss_validation}")
        # self.load()
        training_time = time.time() - start_time
        print("Training time: {}".format(training_time))

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
        x, y, y_unscaled = dataset.get_test()
        predicted_y_test = np.squeeze(self.model(x))

        # re-transforming to numpy
        predicted_y_test = predicted_y_test.detach().cpu().numpy()
        y = y.detach().cpu().numpy()
        y_unscaled = y_unscaled.detach().cpu().numpy()

        unscaled_predicted = dataset.inverse_transform(predicted_y_test)

        assert predicted_y_test.shape == unscaled_predicted.shape
        assert predicted_y_test.shape == y_unscaled.shape

        scaled_mse = mean_squared_error(y, predicted_y_test)
        print(f"scaled_mse_y: {scaled_mse}")

        plt.gcf().set_size_inches(22, 15, forward=True)

        plt.plot(y, label='real', marker='o')
        plt.plot(predicted_y_test, label='predicted', marker='o')

        plt.legend(['Real', 'Predicted'])

        plt.show()

    def save(self):
        """
        This method will save the trained model according to the dataset's code(s) and the current date
        """
        torch.save(self.model.state_dict(), self.filepath)

    def load(self):
        self.model = torch.load(self.filepath)
