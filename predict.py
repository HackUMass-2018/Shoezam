#!/usr/bin/env python
import sys
import tensorflow as tf
from tensorflow import keras
import matplotlib.image as mpimg

import libmodel
import libimage

# Configuration
IMAGE_PATH = "./predict-data/seetha-white.jpeg"

# Prepare image
image_str = tf.read_file(IMAGE_PATH)
image = tf.image.decode_jpeg(image_str)

image = tf.image.resize_images(image, [28, 28])
image = tf.image.rgb_to_grayscale(image)

image = tf.constant(1, shape=image.shape) - image

image = [[image]]

# Setup model
model = libmodel.create_model()

# Load trained weights
model.load_weights(libmodel.checkpoint_path)

model.summary()

# Predict
predictions = model.predict(image, steps=1)

for image_p in predictions:
    i = 0
    for w in image_p:
        print("{}: {}".format(libmodel.class_names[i], w))
        i += 1

