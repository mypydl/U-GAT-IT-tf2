from ops import *

from tensorflow import keras


class MLP(keras.Model):
    def __init__(self, use_bias=True, name='MLP'):
        super(MLP, self).__init__(name=name)
        self.filters = filters = 64*4
        self.use_bias = use_bias
        self.gap = keras.layers.GlobalAvgPool2D()
        self.dense_l1 = keras.layers.Dense(filters, use_bias=use_bias, name='MLP_linear1')
        self.dense_l2 = keras.layers.Dense(filters, use_bias=use_bias, name='MLP_linear2')
        self.dense_g = keras.layers.Dense(filters, use_bias=use_bias, name='MLP_gamma')
        self.dense_b = keras.layers.Dense(filters, use_bias=use_bias, name='MLP_beta')

    def call(self, inputs, training=True):
        filters = self.filters
        x = self.gap(inputs)
        x = self.dense_l1(x)
        x = tf.nn.relu(x)
        x = self.dense_l2(x)
        x = tf.nn.relu(x)
        gamma = self.dense_g(x)
        gamma = tf.reshape(gamma, shape=[1, 1, 1, filters])
        beta = self.dense_b(x)
        beta = tf.reshape(beta, shape=[1, 1, 1, filters])
        return gamma, beta


class Generator(keras.Model):
    def __init__(self, filters=64, n_res=4, name='Generator'):
        super(Generator, self).__init__(name=name)
        self.n_res = n_res
        self.Conv0 = keras.layers.Conv2D(filters, kernel_size=7, strides=1,
                                         kernel_initializer=weight_initializer,
                                         kernel_regularizer=weight_regularizer,
                                         use_bias=True, name='conv_0')
        self.Ins_norm0 = InstanceNormalization(name='ins_norm_0')
        self.Conv_down = [
            keras.layers.Conv2D(filters*2**i, kernel_size=3, strides=2,
                                kernel_initializer=weight_initializer,
                                kernel_regularizer=weight_regularizer,
                                use_bias=True,
                                name='conv_'+str(i)) for i in range(1, 3)
        ]
        self.Ins_norm_down = [
            InstanceNormalization(name='ins_norm_'+str(i)) for i in range(1, 3)
        ]
        self.Conv1_res_down = [
            keras.layers.Conv2D(filters*4, kernel_size=3, strides=1,
                                kernel_initializer=weight_initializer,
                                kernel_regularizer=weight_regularizer,
                                use_bias=True,
                                name='res_down_' + str(i) + '_conv1') for i in
            range(n_res)
        ]
        self.Ins_norm_res1 = [
            InstanceNormalization(name='res_down_' + str(i) + '_ins1') for i in
            range(4)
        ]
        self.Conv2_res_down = [
            keras.layers.Conv2D(filters*4, kernel_size=3, strides=1,
                                kernel_initializer=weight_initializer,
                                kernel_regularizer=weight_regularizer,
                                use_bias=True,
                                name='res_down_' + str(i) + '_conv2') for i in
            range(n_res)
        ]
        self.Ins_norm_res2 = [
            InstanceNormalization(name='res_down_' + str(i) + '_ins2') for i in
            range(4)
        ]
        self.Cam = CAM()
        self.Conv_1x1 = keras.layers.Conv2D(filters*4, kernel_size=1, strides=1,
                                            kernel_initializer=weight_initializer,
                                            kernel_regularizer=weight_regularizer,
                                            use_bias=True, name='conv_1x1')
        self.Mlp = MLP()
        self.Conv1_res_up = [
            keras.layers.Conv2D(filters*4, kernel_size=3, strides=1,
                                kernel_initializer=weight_initializer,
                                kernel_regularizer=weight_regularizer,
                                use_bias=True,
                                name='res_up_' + str(i) + '_conv1') for i in
            range(n_res)
        ]
        self.Ail_norm1 = [
            AdaInsLayerNorm(name='res_up_' + str(i) + 'AIL_norm_1') for i in
            range(n_res)
        ]
        self.Conv2_res_up = [
            keras.layers.Conv2D(filters*4, kernel_size=3, strides=1,
                                kernel_initializer=weight_initializer,
                                kernel_regularizer=weight_regularizer,
                                use_bias=True,
                                name='res_up_' + str(i) + '_conv2') for i in
            range(n_res)
        ]
        self.Ail_norm2 = [
            AdaInsLayerNorm(name='res_up_' + str(i) + 'AIL_norm_2') for i in
            range(n_res)
        ]
        self.Up_sample = [keras.layers.UpSampling2D(2) for _ in range(2)]
        self.Conv_up = [
            keras.layers.Conv2D(filters//2**i, kernel_size=3, strides=1,
                                kernel_initializer=weight_initializer,
                                kernel_regularizer=weight_regularizer,
                                use_bias=True,
                                name='up_'+str(i-1)+'conv') for i in range(1, 3)
        ]
        self.Li_norm = [
            LayerInsNorm(name='up_' + str(i)+'li_norm') for i in range(2)
        ]
        self.Conv_final = keras.layers.Conv2D(filters=3, kernel_size=7, strides=1,
                                              kernel_initializer=weight_initializer,
                                              kernel_regularizer=weight_regularizer,
                                              use_bias=True,
                                              name='G_logit')

    def call(self, inputs, training=None, mask=None):
        # x = keras.Input(shape=(None, 256, 256, 3))(inputs)
        # x = tf.pad(x, [[0, 0], [3, 3], [3, 3], [0, 0]], mode='REFLECT')
        x = tf.pad(inputs, [[0, 0], [3, 3], [3, 3], [0, 0]], mode='REFLECT')
        x = self.Conv0(x)
        x = self.Ins_norm0(x)
        x = tf.nn.relu(x)
        # Down-Sampling
        for i in range(2):
            x = tf.pad(x, [[0, 0], [1, 0], [1, 0], [0, 0]], mode='REFLECT')
            x = self.Conv_down[i](x)
            x = self.Ins_norm_down[i](x)
            x = tf.nn.relu(x)
        # Down-Sampling Bottleneck
        for i in range(self.n_res):
            x_init = x
            x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
            x = self.Conv1_res_down[i](x)
            x = self.Ins_norm_res1[i](x)
            x = tf.nn.relu(x)
            x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
            x = self.Conv2_res_down[i](x)
            x = self.Ins_norm_res2[i](x)
            x = keras.layers.add([x, x_init])
        # Class Activation Map
        cam_logit, x = self.Cam(x)
        x = self.Conv_1x1(x)
        x = tf.nn.relu(x)
        heatmap = tf.squeeze(tf.reduce_sum(x, axis=-1))
        # Gamma, Beta block
        gamma, beta = self.Mlp(x)
        # Up-Sampling Bottleneck
        for i in range(self.n_res):
            x_init = x
            x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
            x = self.Conv1_res_up[i](x)
            x = self.Ail_norm1[i](x, gamma, beta)
            x = tf.nn.relu(x)
            x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
            x = self.Conv2_res_up[i](x)
            x = self.Ail_norm2[i](x, gamma, beta)
            x = keras.layers.add([x, x_init])
        # Up-Sampling
        for i in range(2):
            x = self.Up_sample[i](x)
            x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
            x = self.Conv_up[i](x)
            x = self.Li_norm[i](x)
            x = tf.nn.relu(x)
        x = tf.pad(x, [[0, 0], [3, 3], [3, 3], [0, 0]], mode='REFLECT')
        x = self.Conv_final(x)
        outputs = tf.nn.tanh(x)
        return outputs, cam_logit, heatmap


class SubDiscriminator(keras.Model):
    def __init__(self, filters=64, n_dis=6):
        super(SubDiscriminator, self).__init__()
        self.n_dis = n_dis
        self.Conv0 = keras.layers.Conv2D(filters, kernel_size=4, strides=2,
                                         kernel_initializer=weight_initializer,
                                         kernel_regularizer=weight_regularizer,
                                         use_bias=True, name='conv_0')
        self.Conv_ls = [
            keras.layers.Conv2D(filters*2**i, kernel_size=4, strides=2,
                                kernel_initializer=weight_initializer,
                                kernel_regularizer=weight_regularizer,
                                use_bias=True,
                                name='conv_'+str(i)) for i in range(1, n_dis-1)
        ]
        self.Conv_last = \
            keras.layers.Conv2D(filters*2**(n_dis-1), kernel_size=4, strides=1,
                                kernel_initializer=weight_initializer,
                                kernel_regularizer=weight_regularizer,
                                use_bias=True, name='conv_last')
        self.Cam = CAM(sn=True)
        self.Conv_1x1 = keras.layers.Conv2D(filters*2**(n_dis-1),
                                            kernel_size=1, strides=1,
                                            kernel_initializer=weight_initializer,
                                            kernel_regularizer=weight_regularizer,
                                            use_bias=True, name='conv_1x1')
        self.Conv_final = keras.layers.Conv2D(1, kernel_size=4, strides=1,
                                              kernel_initializer=weight_initializer,
                                              kernel_regularizer=weight_regularizer,
                                              use_bias=True, name='D_logit')

    def call(self, inputs, training=None, mask=None):
        n_dis = self.n_dis
        # x = keras.layers.Input([-1, 256, 256, 3])(inputs)
        # x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
        x = tf.pad(inputs, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
        if not self.Conv0.get_weights():
            self.Conv0.build(x.shape)
        weights = self.Conv0.get_weights()
        weights[0] = spectral_norm(weights[0])
        self.Conv0.set_weights(weights)
        x = self.Conv0(x)
        x = tf.nn.leaky_relu(x, 0.2)
        for i in range(n_dis-2):
            x = tf.pad(x, [[0, 0], [3, 3], [3, 3], [0, 0]], mode='REFLECT')
            if not self.Conv_ls[i].get_weights():
                self.Conv_ls[i].build(x.shape)
            weights = self.Conv_ls[i].get_weights()
            weights[0] = spectral_norm(weights[0])
            self.Conv_ls[i].set_weights(weights)
            x = self.Conv_ls[i](x)
            x = tf.nn.leaky_relu(x, 0.2)
        x = tf.pad(x, [[0, 0], [1, 2], [1, 2], [0, 0]], mode='REFLECT')
        if not self.Conv_last.get_weights():
            self.Conv_last.build(x.shape)
        weights = self.Conv_last.get_weights()
        weights[0] = spectral_norm(weights[0])
        self.Conv_last.set_weights(weights)
        x = self.Conv_last(x)
        x = tf.nn.leaky_relu(x, 0.2)
        cam_logit, x = self.Cam(x)
        x = self.Conv_1x1(x)
        x = tf.nn.leaky_relu(x, 0.2)
        heatmap = tf.squeeze(tf.reduce_sum(x, axis=-1))
        x = tf.pad(x, [[0, 0], [1, 2], [1, 2], [0, 0]], mode='REFLECT')
        if not self.Conv_final.get_weights():
            self.Conv_final.build(x.shape)
        weights = self.Conv_final.get_weights()
        weights[0] = spectral_norm(weights[0])
        self.Conv_final.set_weights(weights)
        outputs = self.Conv_final(x)
        return outputs, cam_logit, heatmap


class Discriminator:
    def __init__(self, filters=64, n_dis=6):
        self.Local = SubDiscriminator(filters, n_dis-2)
        self.Global = SubDiscriminator(filters, n_dis)

    def __call__(self, inputs, training=None):
        D_logit = []
        D_CAM_logit = []
        local_x, local_cam, local_heatmap = self.Local(inputs, training=training)
        global_x, global_cam, global_heatmap = self.Global(inputs, training=training)
        D_logit.extend([local_x, global_x])
        D_CAM_logit.extend([local_cam, global_cam])
        return D_logit, D_CAM_logit, local_heatmap, global_heatmap
