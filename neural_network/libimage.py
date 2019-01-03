import tensorflow as tf

def load_image(image_path):
    # Read file
    image_str = tf.read_file(image_path)
    image = tf.image.decode_jpeg(image_str)

    # Resize 28x28
    image = tf.image.resize_images(image, [28, 28])

    # Grayscale
    image = tf.image.rgb_to_grayscale(image)

    # Invert image
    image = 1 - image

    # Add batch dimension
    image = tf.expand_dims(image, 0)

    # Execute tensors above to get actual values
    with tf.Session():
        image = image.eval()

    # ... Normalize image values to be in [0, 1]
    image = image / 255.0
    
    return image
