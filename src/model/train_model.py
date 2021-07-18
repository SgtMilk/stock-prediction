from tensorflow.keras.models import Model
import os

params_folder = os.path.abspath('./src/experiments/base_model/params.json')


def train_model(model: Model, x_train, y_train):
    # params = json.loads(params_folder)
    model.fit(x_train, y_train, epochs=50, shuffle=True, validation_split=0.1)
