from tensorflow.keras.models import Model


def train_model(model: Model, x_train, y_train):
    """
    Trains the tensorflow model
    :param model: the tensorflow model
    :param x_train
    :param y_train
    """
    model.fit(x_train, y_train, epochs=100, shuffle=True,
              validation_split=0.1)
