#!/usr/bin/env python
import sys
import tensorflow as tf
from tensorflow import keras
import matplotlib.image as mpimg

import model

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
	model = model.create_model()

	# Load trained weights
	model.load_weights(model.checkpoint_path)

	#model.summary()

	# Predict
	predictions = model.predict(image, steps=1)

	shoe_p = 0
	shoe_classes = ['Sandal', 'Sneaker', 'Ankle boot']

	for image_p in predictions:
		i = 0
		for w in image_p:
			if model.class_names[i] in shoe_classes:
				shoe_p += w

			print("{}: {}".format(model.class_names[i], w))
			i += 1
	print(shoe_p)
	return (shoe_p > .6) 
