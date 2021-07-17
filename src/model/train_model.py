from tensorflow.keras.models import Sequential
import json
import os

params_folder = os.path.abspath('./src/experiments/base_model/params.json')


def train_model(model: Sequential, x_train, y_train):
    # params = json.loads(params_folder)
    model.fit(x_train, y_train, epochs=1,
              batch_size=1, verbose=2)
