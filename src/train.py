from model import build_model, compile_model, train_model
from data import build_dataset, download_data
import datetime
import os

model_dir = os.path.abspath('./src/model/models')


def train_stock(code: str):
    # making sure the data is downloaded
    download_data([code], allFlag=True)

    # making the data pwetty 👉️👈️
    x_train, y_train, y_unscaled_train, x_test, y_test, y_unscaled_test = build_dataset(
        code)
    print(x_train.shape)
    print(y_train.shape)

    # building the model
    model = build_model(x_train[0].shape)

    # compiling the model
    compile_model(model)

    # training the model
    train_model(model, x_train, y_train)

    file_name = os.path.join(model_dir, str(
        datetime.date.today()) + '-' + code + ".hdf5")

    model.save(file_name)


if __name__ == '__main__':
    train_stock('AAPL')
