import random

import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

def save_arr(image):
    image = [[[int((y * 255.0))] for y in x] for x in image]

    str_t = tf.image.encode_jpeg(image, format='grayscale')
    with tf.Session():
        str_b = str_t.eval()

    with open("test.jpeg", "wb") as f:
        f.write(str_b)

