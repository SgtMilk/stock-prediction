import os
from typing import List
from yahoofinancials import YahooFinancials
import datetime
import pandas as pd
import csv

destination_folder = os.path.abspath('./src/data/source')
csv_columns = ['date', 'high', 'low', 'open',
               'close', 'volume', 'adjclose', 'formatted_date']


def download(codes: List[str]):
    for code in codes:
        destination = os.path.join(destination_folder, code + '.csv')
        if os.path.exists(destination):
            response = input("would you like to overwrite that file? (y/n)")
            if response == 'y' and response == 'yes':
                os.remove(destination)
            else:
                continue

        currentStock = YahooFinancials(code)

        currentDate = datetime.date.today()
        initialDate = currentDate - datetime.timedelta(days=365.24*5)

        data = currentStock.get_historical_price_data(start_date=str(initialDate),
                                                      end_date=str(
                                                          currentDate),
                                                      time_interval='daily')
        prices = data[code]['prices']

        with open(destination, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            for price in prices:
                writer.writerow(price)


if __name__ == '__main__':
    download(['AAPL'])
