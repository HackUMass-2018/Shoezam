import sys
import tensorflow as tf
from tensorflow import keras
import matplotlib.image as mpimg
from matplotlib.pyplot import imshow

from pprint import pprint

checkpoint_path = "training_checkpoints/checkpoint.ckpt"

def load_test_image():
    image_str = tf.read_file("../predict-data/seetha-white.jpeg")
    image = tf.image.decode_jpeg(image_str)

    image = tf.image.resize_images(image, [28, 28])
    image = tf.image.rgb_to_grayscale(image)

    image = 1 - image
    
    return image

def convolute(images, kernel):
    kernel = tf.constant(kernel, dtype=tf.float32)
    
    kernel = tf.expand_dims(kernel, 2)
    kernel = tf.expand_dims(kernel, 3)

    #image = tf.expand_dims(image, 0)

    processed = tf.nn.convolution(images, kernel, padding="VALID")
    #processed = tf.squeeze(processed, 0)
    
    return processed

IDENTITY_KERNEL = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
EDGE_KERNEL = [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]
BLUR_KERNEL = [[1, 2, 1], [2, 4, 2], [1, 2, 1]]
HV_EDGE_KERNEL = [[0, 1, 0], [1, -4, 1], [0, 1, 0]]
DIAGONAL_EDGE_KERNEL = [[1, 0, -1], [0, 0, 0], [-1, 0, 1]]

def apply_filters(images):
    edged = convolute(images, EDGE_KERNEL)
    blured = convolute(images, BLUR_KERNEL)
    hv_edged = convolute(images, HV_EDGE_KERNEL)
    diagonal_edged = convolute(images, DIAGONAL_EDGE_KERNEL)

    cropped_images = convolute(images, IDENTITY_KERNEL)

    return tf.concat([cropped_images, edged, blured, hv_edged, diagonal_edged], 3, name='concat')

#image = load_test_image()
relu_layer = tf.layers.Dense(4, input_shape=(26, 26, 5),
                             activation=tf.nn.relu,
                                name='relu')
softmax_layer = keras.layers.Dense(2, activation=tf.nn.softmax, name='softmax')

model = keras.Sequential([
    relu_layer,
    softmax_layer
])

# Train
# ... Load training set
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

shaped_train_images = tf.cast(tf.expand_dims(train_images, 3), dtype=tf.float32)

print("filtering inputs")
with tf.Session():
    train_images = apply_filters(shaped_train_images).eval()

print("done")

# ... Make save weights"
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                             save_weights_only=True,
                             verbose=1,
                             period=1)

# ... Compile model
model.compile(optimizer=tf.train.AdamOptimizer(),
          loss='sparse_categorical_crossentropy',
          metrics=['accuracy'])

# ... Fit model
model.fit(train_images, train_labels, epochs=5, steps_per_epoch=5, callbacks=[checkpoint_callback])
print("fit")

# ... Test model
test_loss, test_acc = model.evaluate(test_images, test_labels)

print("loss: {}, acc: {}".format(test_loss, test_acc))
