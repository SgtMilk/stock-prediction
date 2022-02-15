# stock-prediction

This repository is trying to generate estimates of future stock value.
Yes, I know it is impossible to get accurate future values of the stock market. If you use this, I am not liable for any losses of money. 
The prediction uses a PyTorch convolution/GRU GAN model.

# Setup

This project is set up to use VS Code devcontainers. If you do not wish to use them, the following section tells you how to to install required packages.

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

To use Tensorboard alonside your training, run `tensorboard --logdir=runs` to run a tensorboard server.

## Note to the user

The backend services included in this repo are not scalable and were not designed to be so. I would strongly advise to write your own backend services if you want to use this in a large-scale system (the database is a json file 🙂️)

## Copyright

Copyright (c) 2022 Alix Routhier-Lalonde. Licence included in root of package.
