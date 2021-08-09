from flask import Flask
from src.predict import predict as predict_data
from flask_cors import CORS
import json
import datetime


class Backend:
    def __init__(self):
        self.app = Flask(__name__)
        CORS(self.app)

        @self.app.route("/hello")
        def hello():
            return "Hello World!"

        @self.app.route("/predict/<code>/<int:num_days>", methods=['GET'])
        def predict(code, num_days):
            data = predict_data([code], num_days)
            data = data.squeeze().tolist()
            returned_data = []
            if not isinstance(data, list):
                data = [data]
            for index, d in enumerate(data):
                returned_data.append({
                    "date": str(datetime.date.today() + datetime.timedelta(days=index)),
                    "price": d
                })
            return json.dumps(returned_data)

        @self.app.route("/predict/<code>/<int:num_days>/overwrite", methods=['GET'])
        def predict_overwrite(code, num_days):
            data = predict_data([code], num_days, True)
            data = data.squeeze().tolist()
            returned_data = []
            if not isinstance(data, list):
                data = [data]
            for index, d in enumerate(data):
                returned_data.append({
                    "date": str(datetime.date.today() + datetime.timedelta(days=index)),
                    "price": d
                })
            return json.dumps(returned_data)

        @self.app.route("/portfolios", methods=['GET'])
        def portfolios():
            portfolio_list = [
                {
                    "name": "AAPL",
                    "mode": 22
                },
                {
                    "name": "AMZN",
                    "mode": 22
                },
                {
                    "name": "AAPL",
                    "mode": 5
                },
                {
                    "name": "AMZN",
                    "mode": 5
                },
                {
                    "name": "AAPL",
                    "mode": 1
                },
                {
                    "name": "AMZN",
                    "mode": 1
                },
                {
                    "name": "AMZN",
                    "mode": 1
                },
                {
                    "name": "AMZN",
                    "mode": 1
                },
            ]
            return json.dumps(portfolio_list)

    def run(self):
        self.app.run(host='0.0.0.0', port=8000)


if __name__ == '__main__':
    app = Backend()
    app.run()
