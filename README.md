# stock-prediction

This repository is trying to generate estimates of future stock value.
Yes, I know it is impossible to get accurate future values of the stock market. If you use this, I am not liable for any losses of money. 
The prediction uses a PyTorch convolution/GRU GAN model.

## Front-end
You can find a front-end user interface at this github repo: https://github.com/SgtMilk/stock-prediction-frontend 

![UI Screenshot](./assets/ui_screenshot.png)

## Setup

This project is set up to use VS Code devcontainers. If you do not wish to use them, the following section tells you how to to install required packages.

## Required packages

All required packages are specified in `./requirements.txt`.
To install them, run `pip3 install -r requirements.txt`

## Running the project

To run the backend services, run `python3 app.py`

## Training the model

The model is currently a work-in-progress. To train it, run `python3 train.py`.
You can play with hyperparameters in `./src/hyperparameters/gan.py`.

NOTE: You will have to download stock data the first time. You can change the `no_download` paramenter in `./src/hyperparameters/gan.py` to be `False` for that first run.

The model-training code is all in the `./src` folder.
If a helper function has to do with data or cleaning that data, it is situated in the `./src/data` folder.
If a helper function has anything to do with the pytorch model, it is situated in the `./src/model` folder.

To use Tensorboard alonside your training, run `tensorboard --logdir=runs` to run a tensorboard server.

## Note to the user

The backend services included in this repo are not scalable and were not designed to be so. I would strongly advise to write your own backend services if you want to use this in a large-scale system (the database is a json file 🙂️)

## Training Journal
All following trainings were done over 150 epochs.

To change training mode for GeneratorV1, change the `pred_length` parameter in the `./src/hyperparameters/ gan.py` to 1 (or 30).
 
### Training (1) results (GeneratorV1 with `pred_length = 30`)
Training time: 3h25m23s  
MSE: 23702.3984375

This training method gave an output of 30 from the generator, but inside, it's calling itself 30 times and predicting from data it predicted previously and real data. This generator unfortunately suffered from exploding gradients, as we can see in the output example.

![Generator Error](./assets/training1/generator_error.png)
![Discriminator Error](./assets/training1/discriminator_error.png)
![Output example](./assets/training1/exploding_gradients_problem.png)

### Training (2) results (GeneratorV1 with `pred_length = 1`)
Training time: 43m28s  
MSE: 71499.296875

This training method is similar to the previous one, but it is only predicting one number in the generator while training, but calls itself in a loop when serving predictions in eval mode. Unfortunately, this training method also suffered from exploding gradients.

![Generator Error](./assets/training2/generator_error.png)
![Discriminator Error](./assets/training2/discriminator_error.png)
![Output example](./assets/training2/exploding_gradients_problem.png)

### Training (3) results (GeneratorV2)
Training time: TODO (forgot to write it down)
MSE: TODO (forgot to write it down)

I used another, more traditional convolutional and GRU GAN network for this, and it seemed to train well. It actually showed the underlying flaw of using GANs for stock price generation. All the outputs looked similar and did not seem to indicate where the stock was going and seemed almost unresponsive to the input. This is what I get I guess for trying to input actual data instead of noise in the GAN 🙂️.

![Generator Error](./assets/training3/generator_error.png)
![Discriminator Error](./assets/training3/discriminator_error.png)
![Output example](./assets/training3/gan_problem.png)

### Training (4) results (GeneratorV3)
Training time: 54m18s  
MSE: 0.5389038920402527

I updated the previous training (classic convolutional gan) to be a modified cgan. Instead of being divided in classes like a normal cgan, I just inputted the raw past stock prices. It seemed to do exactly what a cgan does: outputs seemed to be similar in some contexts. For example, two ETFs following the S&P 500 were extremely similar, just like the GAN above. Unfortunately, I don't think it would accurately predict real outputs because it grouped some stocks in the same category such as `CL=F` and `AAPL`. Maybe a good idea would be to quantize stock ouput labels so that they would fit better in set categories.

![Generator Error](./assets/training4/generator_error.png)
![Discriminator Error](./assets/training4/discriminator_error.png)
![Output example](./assets/training4/gan_problem.png)

### Analysis
I may be doing something wrong hahahaha

## Copyright

Copyright (c) 2022 Alix Routhier-Lalonde. Licence included in root of package.
