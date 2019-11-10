from CycleGan.try_2.ops import *
import tensorflow as tf
import tensorflow_datasets as tfds
from datetime import datetime
import time


class Generator(tf.keras.Model):
    def __init__(self, filters=64, n_res=4, smoothing=True, light=True,
                 use_bias=True):
        super().__init__()
        self.filters = filters
        self.smoothing = smoothing
        self.light = light
        self.use_bias = use_bias

        self.down_stack = [
            DownSample(self.filters, 7, 1),
            DownSample(self.filters * 2, 3, 2),
            DownSample(self.filters * 4, 3, 2),
            DownSample(self.filters * 4, 1, 1, apply_norm=False)]
        self.res_stack = [ResidualNetwork(self.filters * 4)] * n_res
        self.gap = tf.keras.layers.GlobalAveragePooling2D()
        self.gmp = tf.keras.layers.GlobalMaxPool2D()
        self.fully_connected_with_w = FullyConnectedWithWeight()
        self.mlp = MultilayerPerceptron(self.filters * 4, light, use_bias)
        self.ada_IL_resblock = [AdaILResblock(self.filters * 4)] * n_res
        self.up_stack = [
            UpSample(self.filters * 2, 3, 2, 'layernorm'),
            UpSample(self.filters, 3, 2, 'layernorm')]
        self.dconv = dconv(3, 7, 1)

    def call(self, x_init):
        # Start
        x = self.down_stack[0](x_init)

        '''编码器E'''
        # Down-Sampling
        for i in range(1, 3):
            x = self.down_stack[i](x)

        # Down-Sampling Bottleneck
        for res in self.res_stack:
            x = res(x)

        '''辅助分类器'''
        # Class Activation Map
        cam_x = self.gap(x)
        cam_gap_logit, cam_x_weight = self.fully_connected_with_w(cam_x)
        x_gap = tf.multiply(x, cam_x_weight)

        cam_x = self.gmp(x)
        cam_gmp_logit, cam_x_weight = self.fully_connected_with_w(cam_x)
        x_gmp = tf.multiply(x, cam_x_weight)

        cam_logit = tf.concat([cam_gap_logit, cam_gmp_logit], axis=-1)
        x = tf.concat([x_gap, x_gmp], axis=-1)

        x = self.down_stack[3](x)

        heat_map = tf.squeeze(tf.reduce_sum(x, axis=-1))

        '''解码器G'''
        # Gamma, Beta block
        gamma, beta = self.mlp(x)

        # Up-Sampling Bottleneck
        for res in self.ada_IL_resblock:
            x = res(gamma, beta, x)

        # Up-Sample
        for i in range(2):
            x = self.up_stack[i](x)

        # Finish
        outputs = self.dconv(x)
        return outputs, cam_logit, heat_map


class SubDiscriminator(tf.keras.Model):
    def __init__(self, filters, n_dis=4, d_type='global'):
        super().__init__()
        if d_type.lower() == 'local':
            n_dis -= 2
        self.down_stack = [
            DownSample(filters * 2**i, 4, 2, 'lrelu', False) for i in range(n_dis - 1)
        ] + [DownSample(filters * 2**(n_dis - 1), 4, 1, 'lrelu', False)]
        self.fully_connected_with_w = FullyConnectedWithWeight()
        self.gap = tf.keras.layers.GlobalAveragePooling2D()
        self.gmp = tf.keras.layers.GlobalMaxPool2D()
        self.last = [
            DownSample(filters * 2**(n_dis - 1), 1, 1, 'lrelu', False),
            DownSample(1, 4, 1, 'none', False)
        ]

    def call(self, x_init):
        x = x_init
        for down in self.down_stack:
            x = down(x)
        cam_x = self.gap(x)
        cam_gap_logit, cam_x_weight = self.fully_connected_with_w(cam_x)
        x_gap = tf.multiply(x, cam_x_weight)

        cam_x = self.gmp(x)
        cam_gmp_logit, cam_x_weight = self.fully_connected_with_w(cam_x)
        x_gmp = tf.multiply(x, cam_x_weight)

        cam_logit = tf.concat([cam_gap_logit, cam_gmp_logit], axis=-1)
        x = tf.concat([x_gap, x_gmp], axis=-1)

        x = self.last[0](x)
        heat_map = tf.squeeze(tf.reduce_sum(x, axis=-1))
        outputs = self.last[1](x)

        return outputs, cam_logit, heat_map


class Discriminator(tf.keras.Model):
    def __init__(self, filters=64, n_dis=4):
        super().__init__()
        self.Local = SubDiscriminator(filters, n_dis, 'local')
        self.Global = SubDiscriminator(filters, n_dis, 'global')

    def call(self, inputs):
        D_logit = []
        D_CAM_logit = []
        local_x, local_cam, local_heatmap = self.Local(inputs)
        global_x, global_cam, global_heatmap = self.Global(inputs)
        D_logit.extend([local_x, global_x])
        D_CAM_logit.extend([local_cam, global_cam])
        return D_logit, D_CAM_logit, local_heatmap, global_heatmap


class UGATIT:
    def __init__(self):                             # , args
        self.light = True                           # args.light
        if self.light:
            self.model_name = 'UGATIT_light'
        else:
            self.model_name = 'UGATIT'
        self.filters = 64                           # args.filters
        self.n_res = 4                              # args.n_res
        self.n_dis = 4                              # args.n_dis
        self.epochs = 200                           # args.epochs
        self.init_lr = 0.0001                       # args.init_lr
        self.decay_flag = True                      # args.decay_flag
        self.decay_epoch = 100                      # args.decay_epoch

        # weight
        self.gan_weight = 1                         # args.gan_weight
        self.cycle_weight = 10                      # args.cycle_weight
        self.identity_weight = 100                  # args.identity_weight
        self.cam_weight = 1000                      # args.cam_weight
        self.ld = 1                                 # args.GP_ld
        self.smoothing = True                       # args.smoothing

        # data
        self.data_dir = "cycle_gan/horse2zebra"     # args.data_dir
        self.train_horses = None
        self.train_zebras = None
        self.test_horses = None
        self.test_zebras = None
        self.iter_horse = None
        self.iter_zebra = None

        # models
        self.gen_a2b = Generator(self.filters, self.n_res, self.smoothing)
        self.gen_b2a = Generator(self.filters, self.n_res, self.smoothing)
        self.dis_a = Discriminator(self.filters, self.n_dis)
        self.dis_b = Discriminator(self.filters, self.n_dis)

        # optimizers
        self.opt_gen_a2b = optimizers.Adam(2e-4, beta_1=0.5)
        self.opt_gen_b2a = optimizers.Adam(2e-4, beta_1=0.5)
        self.opt_dis_a = optimizers.Adam(2e-4, beta_1=0.5)
        self.opt_dis_b = optimizers.Adam(2e-4, beta_1=0.5)

        # checkpoint
        self.ckpt_path = "./checkpoints/horse2zebra"     # args.ckpt_path
        self.ckpt = tf.train.Checkpoint(
            already_step=tf.Variable(0, dtype=tf.int32),
            generator_a2b=self.gen_a2b,
            generator_b2A=self.gen_b2a,
            discriminator_a=self.dis_a,
            discriminator_b=self.dis_b,
            generator_a2b_optimizer=self.opt_gen_a2b,
            generator_b2a_optimizer=self.opt_gen_b2a,
            discriminator_a_optimizer=self.opt_dis_a,
            discriminator_b_optimizer=self.opt_dis_b)
        self.ckpt_manager = tf.train.CheckpointManager(
            self.ckpt, self.ckpt_path, max_to_keep=5)

        # summary
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.log_dir = self.ckpt_path + '-log-' + stamp
        self.writer = tf.summary.create_file_writer(self.log_dir)

        # other
        self.load()

    @tf.function
    def train_step(self, domain_A, domain_B):
        with tf.GradientTape(persistent=True) as tape:
            """ Define Generator, Discriminator """
            x_ab, cam_ab, _ = self.gen_a2b(domain_A)
            x_ba, cam_ba, _ = self.gen_b2a(domain_B)
            x_aba, _, _ = self.gen_b2a(x_ab)
            x_bab, _, _ = self.gen_a2b(x_ba)
            x_aa, cam_aa, _ = self.gen_b2a(domain_A)
            x_bb, cam_bb, _ = self.gen_a2b(domain_B)
            real_A_logit, real_A_cam_logit, _, _ = self.dis_a(domain_A)
            real_B_logit, real_B_cam_logit, _, _ = self.dis_b(domain_B)
            fake_A_logit, fake_A_cam_logit, _, _ = self.dis_b(x_ba)
            fake_B_logit, fake_B_cam_logit, _, _ = self.dis_a(x_ab)

            """ Define Loss """
            G_ad_loss_A = generator_loss(fake_A_logit) + \
                generator_loss(fake_A_cam_logit)
            G_ad_loss_B = generator_loss(fake_B_logit) + \
                generator_loss(fake_B_cam_logit)
            D_ad_loss_A = discriminator_loss(real_A_logit, fake_A_logit) + \
                discriminator_loss(real_A_cam_logit, fake_A_cam_logit)
            D_ad_loss_B = discriminator_loss(real_B_logit, fake_B_logit) + \
                discriminator_loss(real_B_cam_logit, fake_B_cam_logit)
            reconstruction_A = L1_loss(x_aba, domain_A)
            reconstruction_B = L1_loss(x_bab, domain_B)
            identity_A = L1_loss(x_aa, domain_A)
            identity_B = L1_loss(x_bb, domain_B)
            cam_A = cam_loss(source=cam_ba, non_source=cam_aa)
            cam_B = cam_loss(source=cam_ab, non_source=cam_bb)
            Generator_A_gan = self.gan_weight * G_ad_loss_A
            Generator_A_cycle = self.cycle_weight * reconstruction_B
            Generator_A_identity = self.identity_weight * identity_A
            Generator_A_cam = self.cam_weight * cam_A
            Generator_B_gan = self.gan_weight * G_ad_loss_B
            Generator_B_cycle = self.cycle_weight * reconstruction_A
            Generator_B_identity = self.identity_weight * identity_B
            Generator_B_cam = self.cam_weight * cam_B
            Generator_A_loss = Generator_A_gan + Generator_A_cycle + \
                Generator_A_identity + Generator_A_cam
            Generator_B_loss = Generator_B_gan + Generator_B_cycle + \
                Generator_B_identity + Generator_B_cam
            Discriminator_A_loss = self.gan_weight * D_ad_loss_A
            Discriminator_B_loss = self.gan_weight * D_ad_loss_B
            Generator_loss = Generator_A_loss + Generator_B_loss
            Discriminator_loss = Discriminator_A_loss + Discriminator_B_loss

            """ Summary """
            tf.summary.scalar("Generator_loss", Generator_loss)
            tf.summary.scalar("Discriminator_loss", Discriminator_loss)
            tf.summary.scalar("G_A_loss", Generator_A_loss)
            tf.summary.scalar("G_A_gan", Generator_A_gan)
            tf.summary.scalar("G_A_cycle", Generator_A_cycle)
            tf.summary.scalar("G_A_identity", Generator_A_identity)
            tf.summary.scalar("G_A_cam", Generator_A_cam)
            tf.summary.scalar("G_B_loss", Generator_B_loss)
            tf.summary.scalar("G_B_gan", Generator_B_gan)
            tf.summary.scalar("G_B_cycle", Generator_B_cycle)
            tf.summary.scalar("G_B_identity", Generator_B_identity)
            tf.summary.scalar("G_B_cam", Generator_B_cam)
            tf.summary.scalar("D_A_loss", Discriminator_A_loss)
            tf.summary.scalar("D_B_loss", Discriminator_B_loss)
            # self.rho_var = []
            # for var in tf.trainable_variables():
            #     if 'rho' in var.name:
            #         self.rho_var.append(tf.summary.histogram(var.name, var))
            #         self.rho_var.append(tf.summary.scalar(var.name + "_min", tf.reduce_min(var)))
            #         self.rho_var.append(tf.summary.scalar(var.name + "_max", tf.reduce_max(var)))
            #         self.rho_var.append(tf.summary.scalar(var.name + "_mean", tf.reduce_mean(var)))
            # g_summary_list = [self.G_A_loss, self.G_A_gan, self.G_A_cycle, self.G_A_identity, self.G_A_cam,
            #                   self.G_B_loss, self.G_B_gan, self.G_B_cycle, self.G_B_identity, self.G_B_cam,
            #                   self.all_G_loss]
            # g_summary_list.extend(self.rho_var)
            # d_summary_list = [self.D_A_loss, self.D_B_loss, self.all_D_loss]
            # self.G_loss = tf.summary.merge(g_summary_list)
            # self.D_loss = tf.summary.merge(d_summary_list)

        """ Training """
        grad_dis_a = tape.gradient(Discriminator_A_loss, self.dis_a.trainable_variables)
        grad_dis_b = tape.gradient(Discriminator_B_loss, self.dis_b.trainable_variables)
        grad_gen_a = tape.gradient(Generator_A_loss, self.gen_a2b.trainable_variables)
        grad_gen_b = tape.gradient(Generator_B_loss, self.gen_b2a.trainable_variables)
        self.opt_dis_a.apply_gradients(zip(grad_dis_a, self.dis_a.trainable_variables))
        self.opt_dis_b.apply_gradients(zip(grad_dis_b, self.dis_b.trainable_variables))
        self.opt_gen_a2b.apply_gradients(zip(grad_gen_a, self.gen_a2b.trainable_variables))
        self.opt_gen_b2a.apply_gradients(zip(grad_gen_b, self.gen_b2a.trainable_variables))

    def train(self):
        # plot_images([next(self.iter_horse), next(self.iter_zebra)], [self.gen_a2b, self.gen_b2a])
        for epoch in range(self.ckpt.already_step.numpy(), self.epochs):
            start = time.time()
            n = 0
            # 启用高级分析器, 隐式启用图收集,会导致高内存开销。
            # tf.summary.trace_on(graph=True, profiler=True)
            tf.summary.trace_on(graph=True)
            for horse, zebra in tf.data.Dataset.zip((self.train_horses,
                                                     self.train_zebras)):
                self.train_step(horse, zebra)
                if n % 20 == 0:
                    print('.', end='')
                n += 1
            with self.writer.as_default():
                tf.summary.trace_export('cyclegan', 0, self.log_dir)
                self.writer.flush()
            if (epoch + 1) % 3 == 0:
                self.ckpt.already_step.assign_add(3)
                ckpt_save_path = self.ckpt_manager.save()
                print('Saved ckpt for epoch {} at {}'.format(epoch + 1,
                                                             ckpt_save_path))
            plot_images([next(self.iter_horse), next(self.iter_zebra)],
                        [self.gen_a2b, self.gen_b2a])
            print('Epoch {} took {}s'.format(epoch + 1, time.time() - start))

    def load(self):
        dataset, _ = tfds.load(self.data_dir, with_info=True,
                               as_supervised=True)
        train_horses, train_zebras = dataset['trainA'], dataset['trainB']
        test_horses, test_zebras = dataset['testA'], dataset['testB']
        AUTOTUNE = tf.data.experimental.AUTOTUNE
        self.train_horses = train_horses.map(
            preprocess_image_train, AUTOTUNE).shuffle(1000).batch(1)
        self.train_zebras = train_zebras.map(
            preprocess_image_train, AUTOTUNE).shuffle(1000).batch(1)
        self.test_horses = test_horses.map(
            normalize, AUTOTUNE).shuffle(1000).batch(1)
        self.test_zebras = test_zebras.map(
            normalize, AUTOTUNE).shuffle(1000).batch(1)
        self.iter_horse = iter(self.test_horses)
        self.iter_zebra = iter(self.test_zebras)
        print("[*] Reading checkpoints...")
        if self.ckpt_manager.latest_checkpoint:
            self.ckpt.restore(self.ckpt_manager.latest_checkpoint)
            print("[*] Success to read " + self.ckpt_manager.latest_checkpoint)
        else:
            print("[*] Failed to find a checkpoint")


if __name__ == '__main__':
    tfds.disable_progress_bar()
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)

    gan = UGATIT()
    gan.train()
