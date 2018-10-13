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

# Import fashion training set
fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Add noise to images
noise = tf.random_normal(shape=tf.shape(train_images),
                         mean=0.0,
                         stddev=4)
train_images += noise

print("generated noise", train_images.shape)

# Normalize images
#train_images = train_images / 255.0
test_images = test_images / 255.0

# Setup model
model = libmodel.create_model()

print("created model")

# Setup training checkpoints
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(libmodel.checkpoint_path,
                                                         save_weights_only=True,
                                                         verbose=1,
                                                         period=1)

# Compile model
model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print("compiled model")

# Train model
model.fit(train_images, train_labels, epochs=5, steps_per_epoch=100,
          callbacks=[checkpoint_callback])

print("fit model")

# Test model
test_loss, test_acc = model.evaluate(test_images, test_labels)

print("loss: {}, acc: {}".format(test_loss, test_acc))
