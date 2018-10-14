#!/usr/bin/env python
import sys
import tensorflow as tf
from tensorflow import keras
import matplotlib.image as mpimg
import pprint
import png

import libmodel
import libimage

# Configuration
IMAGE_PATH = "./predict-data/seetha.jpeg"

# Prepare image
#image = mpimg.imread(IMAGE_PATH)

image_str = tf.read_file(IMAGE_PATH)
image = tf.image.decode_jpeg(image_str)

#image = tf.image.decode_image(image_bytes)
image = tf.image.resize_images(image, [28, 28])
image = tf.image.rgb_to_grayscale(image)
image = tf.image.flip_left_right(image)
#image = tf.reshape(image, [28, 28])
#image = tf.constant([image], dtype=tf.uint8)

with tf.Session():
    image = image.eval()

#image = [[int(y[0]) for y in x] for x in image]
image = [[(1 - y[0] / 255.0) for y in x] for x in image]

libimage.save_arr(image)
print("a")

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

