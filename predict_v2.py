#!/usr/bin/env python
import sys
import tensorflow as tf
from tensorflow import keras
import matplotlib.image as mpimg
from matplotlib.pyplot import imshow
import model_v2

from pprint import pprint

checkpoint_path = "training_checkpoints/checkpoint.ckpt"

# Load model
model = model_v2.create_model()

# Load weights
model.load_weights(checkpoint_path)

#model.summary()

# Load image
def load_test_image():
    image_str = tf.read_file("./predict-data/seetha-white.jpeg")
    image = tf.image.decode_jpeg(image_str)

    image = tf.image.resize_images(image, [28, 28])
    image = tf.image.rgb_to_grayscale(image)

    image = 1 - image
    
    return image

image = load_test_image()

with tf.Session():
    image = image.eval()

image = [[(1 - y[0]) / 255.0 for y in x] for x in image]


# Predict
predictions = model.predict(image, steps=1)
print(predictions)
