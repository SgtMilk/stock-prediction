# stock-prediction

This repository is trying to generate estimates of future stock value.
Yes, I know it is impossible to get accurate future values of the stock market because of it's volatility, but can you really blame me for trying? 😉️
Like I said above, please do not assume that you will get accurate stock values out of this project and get rich this way because this path will probably make you lose a lot of money.

## Required packages

All required packages are specified in `./requirements.txt`.
To install them, run `pip3 install -r requirements.txt`

## Running the project

To run the backend services, run `./app.py`

## Training the model

The model is currently a work-in-progress. To train it, run `./src/train.py`.
You can play with hyperparameters in `./src/hyperparameters/train.py`.

The model-training code is all in the `./src` folder.
If a helper function has to do with data or cleaning that data, it is situated in the `./src/data` folder.
If a helper function has anything to do with the pytorch model, it is situated in the `./src/model` folder.

## Note to the user

The backend services included in this repo are not scalable and were not designed to be so. I would strongly advise to write your own backend services if you want to use this in a large-scale system (the database is a json file 🙂️)

## Copyright

Copyright (c) 2021 Alix Routhier-Lalonde. Licence included in root of package.
