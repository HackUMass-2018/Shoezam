#!/usr/bin/env python
import sys
import os
import random

import libmodel
import libimage

from pprint import pprint

import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

with tf.device("/device:GPU:0"):
	# Import fashion training set
	fashion_mnist = keras.datasets.fashion_mnist

	(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

	"""
	# Add noise to images
	noise = tf.random_normal(shape=tf.shape(train_images),
				 mean=0.0,
				 stddev=1)
	train_images += noise
	"""

	# Setup model
	model = libmodel.create_model()

	# Setup training checkpoints
	checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(libmodel.checkpoint_path,
								 save_weights_only=True,
								 verbose=1,
								 period=1)

	# Compile model
	model.compile(optimizer=tf.train.AdamOptimizer(),
		      loss='sparse_categorical_crossentropy',
		      metrics=['accuracy'])

	# Train model
	model.fit(train_images, train_labels, epochs=10, steps_per_epoch=100,
		  callbacks=[checkpoint_callback])

	# Test model
	test_loss, test_acc = model.evaluate(test_images, test_labels)

	print("loss: {}, acc: {}".format(test_loss, test_acc))
