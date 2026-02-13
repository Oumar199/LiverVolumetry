import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt

import tensorflow as tf
tf.config.set_visible_devices([], 'GPU') # disable GPU 
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
from keras.layers import Dropout

from transformers import AutoProcessor, AutoModelForImageTextToText
from huggingface_hub import snapshot_download
import io
import base64
