from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf
from tensorflow.keras import constraints, layers, losses, optimizers, Sequential
import matplotlib.pyplot as plt


class InstanceNormalization(tf.keras.layers.Layer):
    """Instance Normalization Layer (https://arxiv.org/abs/1607.08022)."""
    def __init__(self, epsilon=1e-5):
        super().__init__()
        self.epsilon = epsilon

    def build(self, input_shape):
        self.scale = self.add_weight(
            name='scale',
            shape=input_shape[-1:],
            # initializer=tf.constant_initializer(0.01),
            initializer=tf.random_normal_initializer(0., 0.02),
            trainable=True)

        self.offset = self.add_weight(
            name='offset',
            shape=input_shape[-1:],
            initializer='zeros',
            trainable=True)

    def call(self, x):
        mean, variance = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        inv = tf.math.rsqrt(variance + self.epsilon)
        normalized = (x - mean) * inv
        return self.scale * normalized + self.offset


class AdaInsLayerNorm(layers.Layer):
    def __init__(self, epsilon=1e-5):
        super().__init__()
        self.epsilon = epsilon

    def build(self, input_shape):
        self.rho = self.add_weight(
            name='rho',
            shape=input_shape[-1],
            initializer=tf.constant_initializer(0.9),
            trainable=True,
            constraint=constraints.MinMaxNorm(0.0, 1.0))

    def call(self, x, gamma, beta):
        ins_mean, ins_sigma = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        x_ins = (x - ins_mean) / (tf.sqrt(ins_sigma + self.epsilon))

        ln_mean, ln_sigma = tf.nn.moments(x, axes=[1, 2, 3], keepdims=True)
        x_ln = (x - ln_mean) / (tf.sqrt(ln_sigma + self.epsilon))

        x_hat = self.rho * x_ins + (1 - self.rho) * x_ln
        return x_hat * gamma + beta


class LayerInsNorm(layers.Layer):
    def __init__(self):
        super().__init__()
        self.epsilon = 1e-5

    def build(self, input_shape):
        self.rho = self.add_weight(
            name='rho',
            shape=input_shape[-1],
            initializer=tf.constant_initializer(0.9),
            trainable=True,
            constraint=constraints.MinMaxNorm(0.0, 1.0))
        self.gamma = self.add_weight(
            name='gamma',
            shape=input_shape[-1],
            initializer=tf.constant_initializer(1.0),
            trainable=True)
        self.beta = self.add_weight(
            name='beta',
            shape=input_shape[-1],
            initializer=tf.constant_initializer(1.0),
            trainable=True)

    def call(self, x):
        ins_mean, ins_sigma = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        x_ins = (x - ins_mean) / (tf.sqrt(ins_sigma + self.epsilon))

        ln_mean, ln_sigma = tf.nn.moments(x, axes=[1, 2, 3], keepdims=True)
        x_ln = (x - ln_mean) / (tf.sqrt(ln_sigma + self.epsilon))

        x_hat = self.rho * x_ins + (1 - self.rho) * x_ln
        return x_hat * self.gamma + self.beta


class FullyConnectedWithWeight(tf.keras.layers.Layer):
    def __init__(self, use_bias=True):
        super().__init__()
        self.use_bias = use_bias

    def build(self, input_shape):
        self.weight = self.add_weight(
            name='weight',
            shape=[input_shape[-1], 1],
            # initializer=tf.constant_initializer(0.01),
            initializer=tf.random_normal_initializer(0.0, 0.02),
            regularizer=tf.keras.regularizers.l2(0.0001),
            trainable=True)
        self.bias = self.add_weight(
            name='bias',
            shape=[1],
            initializer='zero',
            trainable=True)

    def call(self, x):
        if self.use_bias:
            result = tf.matmul(x, self.weight) + self.bias
            weights = tf.gather(tf.transpose(self.weight+self.bias), 0)
        else:
            result = tf.matmul(x, self.weight)
            weights = tf.gather(tf.transpose(self.weight), 0)
        return result, weights


def conv(filters, kernel, strides, padding='same', use_bias=True):
    return tf.keras.layers.Conv2D(
        filters, kernel, strides, padding=padding, use_bias=use_bias,
        # kernel_initializer=tf.constant_initializer(0.01),
        kernel_initializer=tf.random_normal_initializer(0., 0.02),
        kernel_regularizer=tf.keras.regularizers.l2(0.0001))


def dconv(filters, kernel, strides, padding='same', use_bias=True):
    return tf.keras.layers.Conv2DTranspose(
        filters, kernel, strides, padding=padding, use_bias=use_bias,
        activation='tanh',
        # kernel_initializer=tf.constant_initializer(0.01),
        kernel_initializer=tf.random_normal_initializer(0., 0.02),
        kernel_regularizer=tf.keras.regularizers.l2(0.0001))


class DownSample(tf.keras.Model):
    def __init__(self, filters, size, strides, relu_type='relu',
                 apply_norm=True, use_bias=True):
        super().__init__()
        self.conv = conv(filters, size, strides, use_bias=use_bias)
        self.relu_type = relu_type
        self.apply_norm = apply_norm
        self.i_n = InstanceNormalization()

    def call(self, x_init):
        x = self.conv(x_init)
        if self.apply_norm:
            x = self.i_n(x)
        if self.relu_type.lower() == 'relu':
            return tf.nn.relu(x)
        elif self.relu_type.lower() == 'lrelu':
            return tf.nn.leaky_relu(x, 0.2)
        else:
            return x


class UpSample(tf.keras.Model):
    def __init__(self, filters, size, strides, use_bias=True):
        super().__init__()
        self.dconv = dconv(filters, size, strides, use_bias=use_bias)
        self.lin = LayerInsNorm()

    def call(self, x_init):
        x = self.dconv(x_init)
        x = self.lin(x)
        return tf.nn.relu(x)


class ResidualNetwork(tf.keras.Model):
    def __init__(self, filters, use_bias=True):
        super().__init__()
        self.part1 = DownSample(filters, 3, 1, use_bias=use_bias)
        self.part2 = DownSample(filters, 3, 1, relu_type='none', use_bias=use_bias)

    def call(self, x_init):
        x = self.part1(x_init)
        x = self.part2(x)
        return x + x_init


class AdaILResblock(tf.keras.Model):
    def __init__(self, filters, use_bias=True):
        super().__init__()
        self.conv1 = conv(filters, 3, 1, use_bias=use_bias)
        self.adailn1 = AdaInsLayerNorm()
        self.conv2 = conv(filters, 3, 1, use_bias=use_bias)
        self.adailn2 = AdaInsLayerNorm()
    
    def call(self, x_init, gamma, beta):
        x = self.conv1(x_init)
        x = self.adailn1(x, gamma, beta)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = self.adailn2(x, gamma, beta)
        return x + x_init


class MultilayerPerceptron(tf.keras.Model):
    def __init__(self, units, light=True, use_bias=True):
        super().__init__()
        self.units = units
        self.light = light
        self.gap = layers.GlobalAveragePooling2D()
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(units, activation='relu', use_bias=use_bias)
        self.dense2 = layers.Dense(units, activation='relu', use_bias=use_bias)
        self.dense_g = layers.Dense(units, use_bias=use_bias)
        self.dense_b = layers.Dense(units, use_bias=use_bias)

    def call(self, x_init):
        if self.light:
            x = self.gap(x_init)
        else:
            x = self.flatten(x_init)
        x = self.dense1(x)
        x = self.dense2(x)
        gamma = self.dense_g(x)
        beta = self.dense_b(x)
        gamma = tf.reshape(gamma, shape=[1, 1, 1, self.units])
        beta = tf.reshape(beta, shape=[1, 1, 1, self.units])
        return gamma, beta


def L1_loss(x, y):
    loss = tf.reduce_mean(tf.abs(x - y))
    return loss


def cam_loss(source, non_source):
    identity_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        tf.ones_like(source), source))
    non_identity_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        tf.zeros_like(non_source), non_source))
    return identity_loss + non_identity_loss


def discriminator_loss(real, fake):
    loss = []
    for i in range(2):
        real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            tf.ones_like(real[i]), real[i]))
        fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            tf.zeros_like(fake[i]), fake[i]))
        loss.append(real_loss + fake_loss)
    return sum(loss)


def generator_loss(fake):
    loss = []
    for i in range(2):
        fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            tf.ones_like(fake[i]), fake[i]))
        loss.append(fake_loss)
    return sum(loss)


def plot_images(test_images, models):
    to_zebra, _, _ = models[0](test_images[0])
    to_horse, _, _ = models[1](test_images[1])
    plt.figure(figsize=(10, 10))
    contrast = 1
    images = [test_images[0], to_zebra, test_images[1], to_horse]
    title = ['Horse', 'To Zebra', 'Zebra', 'To Horse']
    for i in range(len(images)):
        plt.subplot(2, 2, i + 1)
        plt.title(title[i])
        if i % 2 == 0:
            plt.imshow(images[i][0] * 0.5 + 0.5)
        else:
            plt.imshow(images[i][0] * 0.5 * contrast + 0.5)
    plt.show()


def normalize(image, label):
    image = tf.cast(image, tf.float32)
    image = (image / 127.5) - 1
    return image


def preprocess_image_train(image, label):
    image = tf.image.resize(image, [286, 286],
                            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    image = tf.image.random_crop(image, size=[256, 256, 3])
    image = tf.image.random_flip_left_right(image)
    image = normalize(image, label)
    return image
