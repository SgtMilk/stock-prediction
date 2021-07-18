from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import os

params_folder = os.path.abspath('./src/experiments/base_model/params.json')


def compile_model(model: Model):
    # params = json.loads(params_folder)
    adam = Adam(learning_rate=0.0005)
    model.compile(loss='mse', optimizer=adam)
