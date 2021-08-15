from flask import Flask
from src.predict import predict as predict_data
import pandas_market_calendars as mcal
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
            data = predict_data(code, num_days)
            data = data.squeeze().tolist()
            returned_data = []
            if not isinstance(data, list):
                data = [data]

            # getting dates
            nyse = mcal.get_calendar('NYSE')
            num_days = len(data) * 2
            if num_days < 6:
                num_days = 6
            current_date = datetime.date.today() + datetime.timedelta(days=1)
            max_date = current_date + datetime.timedelta(days=num_days)
            open_days = nyse.valid_days(start_date=str(current_date), end_date=str(max_date))
            for index, d in enumerate(data):
                returned_data.append({
                    "date": str(open_days[index]).split()[0],
                    "price": d
                })
            return json.dumps(returned_data)

        @self.app.route("/predict/<code>/<int:num_days>/overwrite", methods=['GET'])
        def predict_overwrite(code, num_days):
            data = predict_data(code, num_days, True)
            data = data.squeeze().tolist()
            returned_data = []
            if not isinstance(data, list):
                data = [data]

            # getting dates
            nyse = mcal.get_calendar('NYSE')
            num_days = len(data) * 2
            if num_days < 6:
                num_days = 6
            current_date = datetime.date.today() + datetime.timedelta(days=1)
            max_date = current_date + datetime.timedelta(days=num_days)
            open_days = nyse.valid_days(start_date=str(current_date), end_date=str(max_date))
            for index, d in enumerate(data):
                returned_data.append({
                    "date": str(open_days[index]).split()[0],
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
                    "name": "GOOG",
                    "mode": 22
                },
                {
                    "name": "TSLA",
                    "mode": 22
                },
                {
                    "name": "^GSPC",
                    "mode": 22
                },
                {
                    "name": "MSFT",
                    "mode": 22
                },
                {
                    "name": "FB",
                    "mode": 22
                },
            ]
            return json.dumps(portfolio_list)

    def run(self):
        self.app.run(host='0.0.0.0', port=8000, threaded=False)


if __name__ == '__main__':
    app = Backend()
    app.run()
