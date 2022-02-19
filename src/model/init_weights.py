# Copyright (c) 2022 Alix Routhier-Lalonde. Licence included in root of package.

"""
Contains the init_weights function, to initialize the weights in a model.
"""

from torch.nn.init import normal_, constant_


def init_weights(model):
    """
    Initiates the weights of convolutional and batchnorm layers of a model
    """
    classname = model.__class__.__name__
    if classname.find("Conv") != -1:
        normal_(model.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        normal_(model.weight.data, 0.0, 0.02)
        constant_(model.bias.data, 0)
