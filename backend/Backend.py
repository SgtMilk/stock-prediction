from flask import Flask, abort
from src.predict import predict as predict_data
from src.utils import get_base_path
import pandas_market_calendars as mcal
from flask_cors import CORS
from hashlib import sha256
import json
import datetime
import os
import sys


class Backend:
    def __init__(self):
        self.app = Flask(__name__)
        CORS(self.app)

        @self.app.route("/hello")
        def hello():
            """
            tester route
            :return: 'Hello World!'
            """
            return "Hello World!"

        @self.app.route("/predict/<code>/<int:num_days>", methods=['GET'])
        def predict(code, num_days):
            """
            Predicts stock prices for the next `num_days`
            :param code: the stock code
            :param num_days: the number of days to predict
            :return: the prediction data
            """
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

        @self.app.route("/portfolios", methods=['GET'])
        def get_portfolios():
            """
            Will return all portfolio data in the backend
            If no portfolios data exists, will return a 404 code
            :return: the portfolio data
            """
            file = os.path.join(get_base_path(), "backend/data/portfolios.json")
            if os.path.exists(file):
                with open(file) as json_file:
                    data = json.load(json_file)
                    return json.dumps(data)
            abort(404)

        @self.app.route("/portfolios/<portfolio_id>", methods=['GET'])
        def get_portfolio(portfolio_id):
            """
            Given a portfolio id, it will return the portfolio data associated with that id
            If that portfolio cannot be found, it will return a 404 code
            :param portfolio_id: portfolio's id
            :return: the portfolio data
            """
            file = os.path.join(get_base_path(), "backend/data/portfolios.json")
            if os.path.exists(file):
                with open(file) as json_file:
                    data = json.load(json_file)
                    portfolios = data['portfolios']
                    if portfolio_id in portfolios:
                        return json.dumps(portfolios[portfolio_id])
                    else:
                        abort(404)
            return abort(404)

        @self.app.route("/portfolios/<name>", methods=['POST'])
        def add_portfolio(name):
            """
            Will add a portfolio to all the backend's portfolios.
            If no portfolio data exists in the backend, it will initialize it and create a file for it
            If the portfolio already exists, it will return all the current portfolios with a 208 code.
            :param name: the portfolio's name
            :return: all backend portfolios
            """
            file = os.path.join(get_base_path(), "backend/data/portfolios.json")
            portfolio_id = str(sha256(name.encode('utf-8')).hexdigest())
            if os.path.exists(file):
                with open(file) as json_file:
                    data = json.load(json_file)
                    portfolios = data['portfolios']
                    if portfolio_id in portfolios:
                        return json.dumps(data), 208
                    else:
                        new_portfolio = {
                            "id": portfolio_id,
                            "name": name,
                            "stocks": []
                        }
                        portfolios[portfolio_id] = new_portfolio
            else:
                data = {
                    "portfolios": {
                        portfolio_id: {
                            "id": portfolio_id,
                            "name": name,
                            "stocks": []
                        }
                    },
                    "stocks": {}
                }

            with open(file, 'w') as outfile:
                json.dump(data, outfile)
            return json.dumps(data)

        @self.app.route("/portfolios/<portfolio_id>", methods=['DELETE'])
        def delete_portfolio(portfolio_id):
            """
            Given a portfolio id, it delete the portfolio attached to it
            If that portfolio cannot be found, it will return a 404 code
            :param portfolio_id: portfolio's id
            :return: the portfolio data
            """
            file = os.path.join(get_base_path(), "backend/data/portfolios.json")
            if os.path.exists(file):
                with open(file) as json_file:
                    data = json.load(json_file)
                    portfolios = data['portfolios']
                    stocks = data['stocks']
                    if portfolio_id in portfolios:
                        portfolio = portfolios[portfolio_id]
                        for stock_id in portfolio['stocks']:
                            condition = False
                            for portfolio in portfolios.values():
                                if stock_id in portfolio['stocks']:
                                    condition = True
                            if not condition and stock_id in stocks:
                                del stocks[stock_id]
                        del portfolios[portfolio_id]
                        with open(file, 'w') as outfile:
                            json.dump(data, outfile)
                        return json.dumps(data)
                    else:
                        abort(404)
            return abort(404)

        @self.app.route("/portfolios/<portfolio_id>/stocks/<name>/<int:mode>", methods=['POST'])
        def add_stock_to_portfolio(portfolio_id, name, mode):
            """
            Will add a stock to a portfolio.
            If the stock already exists in the portfolio, it will return all the current portfolios with a 208 code.
            If the portfolio cannot be found, a 404 code will be returned
            :param portfolio_id: the portfolio's id
            :param name: the stock's code
            :param mode: the number of predicted days wanted
            :return: the whole portfolios data
            """
            file = os.path.join(get_base_path(), "backend/data/portfolios.json")
            if os.path.exists(file):
                with open(file) as json_file:
                    data = json.load(json_file)
                    portfolios = data['portfolios']
                    stocks = data['stocks']
                    if portfolio_id in portfolios:
                        stock_id = str(sha256((name + str(mode)).encode('utf-8')).hexdigest())
                        if stock_id in portfolios[portfolio_id]["stocks"] and stock_id in stocks:
                            return json.dumps(data), 208
                        else:
                            if stock_id not in stocks:
                                stocks[stock_id] = {
                                    "id": stock_id,
                                    "name": name,
                                    "mode": mode
                                }
                            if stock_id not in portfolios[portfolio_id]["stocks"]:
                                portfolios[portfolio_id]["stocks"].append(stock_id)
                            with open(file, 'w') as outfile:
                                json.dump(data, outfile)
                            return json.dumps(data)
                    else:
                        abort(404)
            else:
                abort(404)

        @self.app.route("/portfolios/<portfolio_id>/stocks/<stock_id>", methods=['DELETE'])
        def delete_stock_from_portfolio(portfolio_id, stock_id):
            """
            Will delete a stock from a portfolio.
            If that stock doesn't exist in that portfolio, will return a 404 error
            :param portfolio_id: the portfolio's id
            :param stock_id: the stock's id
            :return: the whole portfolios data
            """
            file = os.path.join(get_base_path(), "backend/data/portfolios.json")
            if os.path.exists(file):
                with open(file) as json_file:
                    data = json.load(json_file)
                    portfolios = data['portfolios']
                    stocks = data['stocks']
                    if portfolio_id in portfolios:
                        if stock_id in portfolios[portfolio_id]["stocks"]:
                            portfolios[portfolio_id]["stocks"].remove(stock_id)

                            # purging in the stocks
                            condition = False
                            for portfolio in portfolios.values():
                                if stock_id in portfolio['stocks']:
                                    condition = True
                            if not condition and stock_id in stocks:
                                del stocks[stock_id]
                            with open(file, 'w') as outfile:
                                json.dump(data, outfile)
                            return json.dumps(data)
                        else:
                            abort(404)
                    else:
                        abort(404)
            else:
                abort(404)

    def run(self):
        self.app.run(host='0.0.0.0', port=8000, threaded=False)


if __name__ == '__main__':
    app = Backend()
    app.run()
