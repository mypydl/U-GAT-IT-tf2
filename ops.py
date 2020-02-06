import tensorflow as tf
from tensorflow import keras

weight_initializer = tf.random_normal_initializer(0., 0.02)
weight_regularizer = keras.regularizers.l2(0.0002)


class InstanceNormalization(keras.layers.Layer):
    def __init__(self, epsilon=1e-5, name=None):
        super(InstanceNormalization, self).__init__(name=name)
        self.epsilon = epsilon

    def build(self, input_shape):
        self.scale = self.add_weight(
            name='scale',
            shape=input_shape[-1:],
            initializer=weight_initializer,
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


class AdaInsLayerNorm(keras.layers.Layer):
    def __init__(self, epsilon=1e-5, name=None):
        super(AdaInsLayerNorm, self).__init__(name=name)
        self.epsilon = epsilon

    def build(self, input_shape):
        self.rho = self.add_weight(
            name='rho',
            shape=input_shape[-1],
            initializer=tf.constant_initializer(1.0),
            trainable=True,
            constraint=keras.constraints.MinMaxNorm(0.0, 1.0))

    def call(self, x, gamma, beta):
        ins_mean, ins_sigma = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        x_ins = (x - ins_mean) / (tf.sqrt(ins_sigma + self.epsilon))
        ln_mean, ln_sigma = tf.nn.moments(x, axes=[1, 2, 3], keepdims=True)
        x_ln = (x - ln_mean) / (tf.sqrt(ln_sigma + self.epsilon))
        x_hat = self.rho * x_ins + (1 - self.rho) * x_ln
        return x_hat * gamma + beta


class LayerInsNorm(keras.layers.Layer):
    def __init__(self, name=None):
        super(LayerInsNorm, self).__init__(name=name)
        self.epsilon = 1e-5

    def build(self, input_shape):
        self.rho = self.add_weight(
            name='rho',
            shape=input_shape[-1],
            initializer=tf.constant_initializer(0.0),
            trainable=True,
            constraint=keras.constraints.MinMaxNorm(0.0, 1.0))
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


def spectral_norm(w, iteration=1):
    init = tf.random_normal_initializer()
    w_shape = w.shape
    w = tf.reshape(w, [-1, w_shape[-1]])
    u = tf.Variable(initial_value=init(shape=[1, w_shape[-1]]), name='u',
                    trainable=False)
    u_hat = u
    v_hat = None
    for i in range(iteration):
        v_ = tf.matmul(u_hat, tf.transpose(w))
        v_hat = tf.nn.l2_normalize(v_)
        u_ = tf.matmul(v_hat, w)
        u_hat = tf.nn.l2_normalize(u_)
    u_hat = tf.stop_gradient(u_hat)
    v_hat = tf.stop_gradient(v_hat)
    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))
    with tf.control_dependencies([u.assign(u_hat)]):
        w_norm = w / sigma
        w_norm = tf.reshape(w_norm, w_shape)
    return w_norm


class CAM(keras.layers.Layer):
    def __init__(self, use_bias=True, sn=False, name=None):
        super(CAM, self).__init__(name=name)
        self.use_bias = use_bias
        self.sn = sn

    def build(self, input_shape):
        self.weight = self.add_weight(
            name='weight',
            shape=[input_shape[-1], 1],
            initializer=weight_initializer,
            regularizer=weight_regularizer,
            trainable=True)
        self.bias = self.add_weight(
            name='bias',
            shape=[1],
            initializer='zero',
            trainable=True)

    def call(self, inputs):
        gap = tf.reduce_mean(inputs, axis=[1, 2])   # global_avg_pooling
        gmp = tf.reduce_max(inputs, axis=[1, 2])    # global_max_pooling
        if self.sn:
            self.weight = spectral_norm(self.weight)
        if self.use_bias:
            cam_gap_logit = tf.matmul(gap, self.weight) + self.bias
            cam_gap_weight = tf.gather(tf.transpose(self.weight + self.bias), 0)
            cam_gmp_logit = tf.matmul(gmp, self.weight) + self.bias
            cam_gmp_weight = tf.gather(tf.transpose(self.weight + self.bias), 0)
        else:
            cam_gap_logit = tf.matmul(gap, self.weight)
            cam_gap_weight = tf.gather(tf.transpose(self.weight), 0)
            cam_gmp_logit = tf.matmul(gmp, self.weight)
            cam_gmp_weight = tf.gather(tf.transpose(self.weight), 0)
        x_gap = tf.multiply(inputs, cam_gap_weight)
        x_gmp = tf.multiply(inputs, cam_gmp_weight)
        cam_logit = tf.concat([cam_gap_logit, cam_gmp_logit], axis=-1)
        outputs = tf.concat([x_gap, x_gmp], axis=-1)
        return cam_logit, outputs


def L1_loss(x, y):
    return tf.reduce_mean(tf.abs(x - y))


def cam_loss(source, non_source):
    identity_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(tf.ones_like(source), source))
    non_identity_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(tf.zeros_like(non_source), non_source))
    return identity_loss + non_identity_loss


def discriminator_loss(real, fake):
    loss = []
    for i in range(2):
        real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(tf.ones_like(real[i]), real[i]))
        fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(tf.zeros_like(fake[i]), fake[i]))
        loss.append(real_loss + fake_loss)
    return sum(loss)


def generator_loss(fake):
    loss = []
    for i in range(2):
        fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(tf.ones_like(fake[i]), fake[i]))
        loss.append(fake_loss)
    return sum(loss)
