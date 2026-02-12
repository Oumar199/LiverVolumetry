import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt

from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
from keras.layers import Dropout

from transformers import AutoProcessor, AutoModelForImageTextToText
import io
import base64
