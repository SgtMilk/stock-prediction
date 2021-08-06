from flask import Flask
from src.predict import predict as predict_data
import json


class Backend:
    def __init__(self):
        self.app = Flask(__name__)

        @self.app.route("/hello")
        def hello():
            return "Hello World!"

        @self.app.route("/predict/<code>/<int:num_days>", methods=['GET'])
        def predict(code, num_days):
            data = predict_data([code], num_days)
            return json.dumps(data.tolist())

        @self.app.route("/predict/<code>/<int:num_days>/overwrite", methods=['GET'])
        def predict_overwrite(code, num_days):
            data = predict_data([code], num_days, True)
            return json.dumps(data.tolist())

    def run(self):
        self.app.run(host='0.0.0.0', port=8000)


if __name__ == '__main__':
    app = Backend()
    app.run()
