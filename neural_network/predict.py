#!/usr/bin/env python
import sys
import tensorflow as tf
from tensorflow import keras
import matplotlib.image as mpimg

from . import libmodel

def is_shoe(model, image_path):
	# Prepare image
	# ... Read file
	image_str = tf.read_file(image_path)
	image = tf.image.decode_jpeg(image_str)

	# ... Resize 28x28
	image = tf.image.resize_images(image, [28, 28])

	# ... Grayscale
	image = tf.image.rgb_to_grayscale(image)

	# ... Invert image
	image = 1 - image

	# ... Add batch dimension
	image = tf.expand_dims(image, 0)

	# ... Execute tensors above to get actual values
	with tf.Session():
	    image = image.eval()

	# ... Normalize image values to be in [0, 1]
	image = image / 255.0

	# Predict
	# ... Run prediction
	predictions = model.predict(image, steps=1)

	# ... Print prediction results
	shoe_p = 0
	total_p = 0

	# ... For each image in prediction results
	for image_p in predictions:
		# For each class in image prediction results
		i = 0
		for w in image_p:
			total_p += w

			if libmodel.class_names[i] in libmodel.shoe_classes:
				shoe_p += w

			print("{}: {}".format(libmodel.class_names[i], w))
			i += 1

	# ... Compute shoe cutoff based on total probability
	shoe_cutoff = total_p * 0.6

	print("Total shoe: {}".format(shoe_p))
	print("Shoe cutoff: {}".format(shoe_cutoff))

	return shoe_p > shoe_cutoff 

if __name__ == '__main__':
	model = libmodel.load_trained_model()

	test_images = ['noah.jpeg', 'seetha-cropped.jpeg', 'seetha.jpeg']

	for img_name in test_images:
		file_path = "./predict-data/{}".format(img_name)

		padding_len = (len(img_name) + 2)

		print("=" * padding_len)
		print("# {}".format(img_name))
		print("=" * padding_len)

		predict_result = is_shoe(model, file_path)

		print(predict_result)
