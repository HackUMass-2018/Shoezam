#!/usr/bin/env python
import sys
import tensorflow as tf
from tensorflow import keras
import matplotlib.image as mpimg
import pprint

import libmodel

import model_v2

checkpoint_path = "training_checkpoints/checkpoint.ckpt"

# Configuration
IMAGE_PATH = "./predict-data/seetha.jpeg"

# Prepare image
def is_shoe(image_path):
	image_str = tf.read_file(image_path)
	image = tf.image.decode_jpeg(image_str)

	image = tf.image.resize_images(image, [28, 28])
	image = tf.image.rgb_to_grayscale(image)
	image = 1 - image
	image = tf.expand_dims(image, 0)

	with tf.Session():
	    image = image.eval()

	image = image / 255.0

	# Setup model
	model = model_v2.create_model()

	# Load trained weights
	model.load_weights(checkpoint_path)

	#model.summary()

	# Predict
	predictions = model.predict(image, steps=1)

	max_p = 0
	max_p_class = None

	for image_p in predictions:
		i = 0
		for w in image_p:
			if w > max_p:
				max_p = w
				max_p_class = libmodel.class_names[i]

			print("{}: {}".format(libmodel.class_names[i], w))
			i += 1
	return max_p_class in ['Sandal', 'Sneaker', 'Ankle boot']

print(is_shoe(IMAGE_PATH))
