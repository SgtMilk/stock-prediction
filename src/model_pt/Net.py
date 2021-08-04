from torch.optim import optimizer as optim
import numpy as np
import time


class Net:
    """
    The Net class will build the model and train it.
    """

    def __init__(self, optimizer: optim, loss_func, model):
        """
        The __init__ function will set all training parameters and generate the model
        :param optimizer: the training optimizer
        :param loss_func: the training loss function
        :param model: the pytorch model
        """
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.model = model

        self.weights_train = None
        self.loss_train = None
        self.loss_validation = None
        self.hist = None

    def train(self, epochs: int, data, validation_split: float, verbosity_interval: int = 1):
        """
        The training loop for the net
        :param epochs: the number of epochs the training loop will run
        :param data: the dataset's train data
        :param validation_split: the split between validation and training data
        :param verbosity_interval: at which epoch interval there will be logging
        """
        x, y = data
        n = int(x.shape[0] * (1 - validation_split))
        x_train, y_train = x[:n], y[:n]
        x_validation, y_validation = x[n:], x[n:]

        self.hist = np.zeros((epochs, 2))
        start_time = time.time()

        for epoch in range(1, epochs + 1):
            # training
            y_predicted_train = self.model(x_train)
            self.loss_train = self.loss_func(y_predicted_train, y_train)

            # validation
            y_predicted_validation = self.model(x_validation)
            self.loss_validation = self.loss_func(y_predicted_validation, y_validation)

            self.hist[epoch] = np.array([self.loss_train, self.loss_validation])

            # optimizer
            self.optimizer.zero_grad()
            self.loss_train.backward()
            self.optimizer.step()

            # logging losses
            if epoch == 1 or epoch % verbosity_interval == 0:
                print(f"Epoch {epoch}, Training Loss: {self.loss_train.item():.4f}, " +
                      f"Validation Loss: {self.loss_validation:.4f}")

        training_time = time.time() - start_time
        print("Training time: {}".format(training_time))

    def evaluate(self, test_data):
        # TODO: implement this
        """
        This function will evaluate the model
        :param test_data: the dataset
        """
        print(test_data)
        print(self.loss_train)
