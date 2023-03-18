import os
import time
import argparse
from PIL import Image
import PIL.ImageOps
import torch
from torchvision import transforms
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from torch.autograd import Variable
import numpy as np
import copy
from skimage.metrics import structural_similarity
import albumentations as album
from albumentations.pytorch import ToTensorV2
from collections import Counter
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import PIL
import pickle

def color_bit_squeezer(img, i):
    #img: tensor 3*400*400 [0, 1]
    #i: number of target color bit depth (1 leq i leq 7)
    #return: tensor 3*400*400 [0, 1]
    ret = torch.round(img * (2 ** i - 1))
    ret = img / (2 ** i - 1)
    return ret

def median_filter(img, size):
    #img: tensor 3*400*400 [0, 1]
    #size: filter size (odd)
    #return: tensor 3*400*400 [0, 1]
    image = img.numpy()
    ret = np.zeros((3,400,400))
    ret[0] = ndimage.median_filter(image[0], size=size)
    ret[1] = ndimage.median_filter(image[1], size=size)
    ret[2] = ndimage.median_filter(image[2], size=size)
    ret = torch.from_numpy(ret).float()
    return ret