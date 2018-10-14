import sys
import tensorflow as tf
from tensorflow import keras
import matplotlib.image as mpimg
from matplotlib.pyplot import imshow

from pprint import pprint

checkpoint_path = "training_checkpoints/checkpoint.ckpt"

def convolute(images, kernel):
    kernel = tf.constant(kernel, dtype=tf.float32)
    
    kernel = tf.expand_dims(kernel, 2)
    kernel = tf.expand_dims(kernel, 3)

    #images = tf.expand_dims(images, 0)
    
    #raise ValueError(images.shape, kernel.shape)
    processed = tf.nn.convolution(images, kernel, padding="VALID")
    processed = tf.clip_by_value(processed, 0, 255)
    
    return processed

IDENTITY_KERNEL = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
EDGE_KERNEL = [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]
BLUR_KERNEL = [[1, 2, 1], [2, 4, 2], [1, 2, 1]]
HV_EDGE_KERNEL = [[0, 1, 0], [1, -4, 1], [0, 1, 0]]
DIAGONAL_EDGE_KERNEL = [[1, 0, -1], [0, 0, 0], [-1, 0, 1]]

def apply_filters(images):
    edged = convolute(images, EDGE_KERNEL)
    blured = convolute(images, BLUR_KERNEL)
    #hv_edged = convolute(images, HV_EDGE_KERNEL)
    #diagonal_edged = convolute(images, DIAGONAL_EDGE_KERNEL)

    
    #cropped_images = convolute(images, IDENTITY_KERNEL)
    
    """
    return tf.concat([cropped_images, edged, blured, hv_edged, diagonal_edged],
                     3,
                     name='concat')
    """
    return tf.concat([edged, blured], 3)
    #return tf.concat([edged, blured, hv_edged, diagonal_edged, cropped_images], 3)

#image = load_test_image()

def create_model():
	return keras.Sequential([
	    keras.layers.Lambda(apply_filters),
	    keras.layers.Flatten(input_shape=(28, 28)),
	    keras.layers.Dense(128, activation=tf.nn.relu),
	    keras.layers.Dense(10, activation=tf.nn.softmax),
	])
