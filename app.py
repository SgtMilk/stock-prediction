# Copyright (c) 2022 Alix Routhier-Lalonde. Licence included in root of package.

"""
==========
app module
==========

This module will run the backend services of stock-prediction.
It is to be used with the front-end in the https://github.com/SgtMilk/stock-prediction-frontend
github repo.
"""

from backend import Backend

if __name__ == "__main__":
    app = Backend()
    app.run()
