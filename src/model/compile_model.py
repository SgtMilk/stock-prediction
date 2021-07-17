from tensorflow.keras.models import Sequential
import json
import os

params_folder = os.path.abspath('./src/experiments/base_model/params.json')


def compile_model(model: Sequential):
    # params = json.loads(params_folder)
    model.compile(loss='mean_squared_error', optimizer='adam')
