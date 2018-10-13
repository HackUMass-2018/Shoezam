#!/usr/bin/env python
import tensorflow as tf
from tensorflow import keras
import matplotlib.image as mpimg

import libmodel

# Configuration
IMAGE_PATH = "./predict-data/predict.jpeg"

# Setup model
model = libmodel.create_model()

# Load trained weights
model.load_weights(libmodel.checkpoint_path)

# Prepare image
image = mpimg.imread(IMAGE_PATH)

#image = tf.image.decode_image(image_bytes)
#image = tf.image.resize_image_with_pad(image, 28, 28)

#image = tf.image.convert_image_dtype(image, tf.float32)

# Predict
prediction = model.predict(image, steps=1)

print(prediction)
