#!/usr/bin/env python
import sys
import tensorflow as tf
from tensorflow import keras
import matplotlib.image as mpimg

import libmodel
import libimage

def is_shoe(model, image_path):
	image = libimage.load_image(image_path)

	# Predict
	# ... Run prediction
	predictions = model.predict(image, steps=1)

	# ... Print prediction results
	max_shoe_p = 0
	total_p = 0

	# ... For each image in prediction results
	for image_p in predictions:
		# For each class in image prediction results
		i = 0
		for w in image_p:
			total_p += w

			if libmodel.class_names[i] in libmodel.shoe_classes and w > max_shoe_p:
				max_shoe_p = w

			print("{}: {}".format(libmodel.class_names[i], w))
			i += 1

	# ... Compute shoe cutoff based on total probability
	shoe_cutoff = total_p * 0.6

	print("Max shoe: {}".format(max_shoe_p))
	print("Shoe cutoff: {}".format(shoe_cutoff))

	return max_shoe_p > shoe_cutoff 

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
