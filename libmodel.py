import os

import tensorflow as tf
from tensorflow import keras

# Configuration
checkpoint_path = "training_checkpoints/checkpoint.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

def create_model():
    return keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(10, activation=tf.nn.softmax),
    ])
