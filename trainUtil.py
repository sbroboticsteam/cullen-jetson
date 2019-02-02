import os

import torch
from PIL import Image
from skimage.transform import resize
from torch.utils.data import Dataset


def initWeightsNormal(module):
    """
    Sets the module's weights to a random number in a normal distribution

    :param module: a layer in the network
    """
    classname = module.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(module.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(module.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(module.bias.data, 0.0)
