import tensorflow as tf


def normalize(image, label):
    image = tf.cast(image, tf.float32)
    # image = (image / 127.5) - 1
    image /= 255.0
    return image


def preprocess(image, label):
    image = tf.image.resize(image, [286, 286], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    image = tf.image.random_crop(image, size=[256, 256, 3])
    image = tf.image.random_flip_left_right(image)
    image = normalize(image, label)
    return image
