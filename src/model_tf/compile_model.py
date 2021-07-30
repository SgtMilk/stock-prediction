from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


def compile_model(model: Model):
    """
    Compiles the tensorflow model
    :param model: the model
    """
    adam = Adam(learning_rate=0.0005)
    model.compile(loss='mse', optimizer=adam)
