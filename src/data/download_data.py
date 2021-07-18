import os
from typing import List
from yahoofinancials import YahooFinancials
import datetime
import csv

destination_folder = os.path.abspath('./src/data/source')
csv_columns = ['date', 'high', 'low', 'open',
               'close', 'volume', 'adjclose', 'formatted_date']


def download_data(codes: List[str], allFlag: bool = False):
    print("If data takes more than 10 seconds to download, to a ctrl + c to go to next stock")
    for code in codes:
        destination = os.path.join(destination_folder, code + '.csv')
        if allFlag and os.path.exists(destination):
            os.remove(destination)

        if not allFlag and os.path.exists(destination):
            response = input(
                "would you like to overwrite that file(" + destination + ")? (y/n)")
            if response == 'y' or response == 'yes':
                os.remove(destination)
            else:
                continue

        currentStock = YahooFinancials(code)

        currentDate = datetime.date.today()
        initialDate = currentDate - datetime.timedelta(days=365.24*5)
        data = []
        try:
            data = currentStock.get_historical_price_data(start_date=str(initialDate),
                                                          end_date=str(
                currentDate),
                time_interval='daily')
        except:
            print("\nCould not download data for " +
                  code + " (it probably doesn't exist")
            continue

        prices = data[code]['prices']

        with open(destination, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            for price in prices:
                writer.writerow(price)


if __name__ == '__main__':
    array = ['^GSPTSE', '^N225', '^TNX', '^VIX', 'AAPL',
             'ARL', 'BTC-USD', 'CL=F', 'GC=F', 'YVR']
    download_data(array)
    print("All data downloaded!")
