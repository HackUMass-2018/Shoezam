#!/usr/bin/env python
import sys
import tensorflow as tf
from tensorflow import keras
import matplotlib.image as mpimg
from matplotlib.pyplot import imshow
import model_v2

from pprint import pprint

checkpoint_path = "training_checkpoints/checkpoint.ckpt"

# Train
# ... Load training set
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

train_images = train_images / 255.0
test_images = test_images / 255.0

with tf.Session():
    train_images = tf.expand_dims(train_images, 3).eval()
    test_images = tf.expand_dims(test_images, 3).eval()

model = model_v2.create_model()
    
# ... Make save weights
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                             save_weights_only=True,
                             verbose=1,
                             period=1)

# ... Compile model
model.compile(optimizer=tf.train.AdamOptimizer(),
          loss='sparse_categorical_crossentropy',
          metrics=['accuracy'])

# ... Fit model
model.fit(train_images, train_labels, epochs=15, steps_per_epoch=10, callbacks=[checkpoint_callback])

# ... Test model
test_loss, test_acc = model.evaluate(test_images, test_labels)

print("loss: {}, acc: {}".format(test_loss, test_acc))


