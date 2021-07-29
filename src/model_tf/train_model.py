from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping


def train_model(model: Model, x_train, y_train):
    callbacks = [EarlyStopping(monitor='acc', patience=10)]
    model.fit(x_train, y_train, epochs=100, shuffle=True,
              validation_split=0.1)
