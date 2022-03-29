from yfinance import Ticker
from pandas import DataFrame
from src.data import get_stock_symbols

stock_symbols = get_stock_symbols()
new_list = []
for i in stock_symbols:
    stock = Ticker(i)
    if not stock.history().empty:
        new_list.append(i)

new_list = DataFrame(new_list, columns=["Name"]).to_csv("./src/data/stock_prices/symbols2.csv")
