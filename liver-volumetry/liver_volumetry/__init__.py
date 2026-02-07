import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

from transformers import AutoProcessor, AutoModelForImageTextToText