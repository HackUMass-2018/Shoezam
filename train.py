#!/usr/bin/env python
import sys
import os

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

# Normalize images
train_images = train_images / 255.0
test_images = test_images / 255.0

libimage.save_arr(test_images[0])

sys.exit(1)

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
model.fit(train_images, train_labels, epochs=5, callbacks=[checkpoint_callback])

# Test model
test_loss, test_acc = model.evaluate(test_images, test_labels)

print("loss: {}, acc: {}".format(test_loss, test_acc))
