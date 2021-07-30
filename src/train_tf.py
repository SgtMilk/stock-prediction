from model_tf import build_model, compile_model, train_model, evaluate_model
from src.utils.print_colors import colors
from data import Dataset, Mode
import datetime
import os
import tensorflow as tf


def train_stock(code: str, mode: int = Mode.daily):
    """
    does the whole tensorflow training process
    :param code: the stock's code
    :param mode: Mode.daily, Mode.weekly, Mode.monthly
    """

    # downloading and making the data pwetty 👉️👈️
    dataset = Dataset(code, y_flag=True)

    # getting the right data
    x_train, y_train = dataset.get_train(mode)
    x_test, y_unscaled_test = dataset.get_test(mode)
    normalizer = dataset.get_normalizer(mode)

    if not x_train or not x_test or not normalizer:
        raise NameError(colors.FAIL + "Could not fetch data" + colors.ENDC)

    # building the model
    model = build_model(y_train.shape)

    # compiling the model
    compile_model(model)

    # training the model
    train_model(model, x_train, y_train)

    # saving the model
    model_dir = os.path.abspath('./src/model/models')
    file_name = os.path.join(model_dir, str(datetime.date.today()) + '-' + code + ".hdf5")
    model.save(file_name)

    evaluate_model(model, x_test, y_unscaled_test, normalizer, mode)


if __name__ == '__main__':
    tf.random.set_seed(3)
    tf.autograph.set_verbosity(1)
    train_stock('ARL')
