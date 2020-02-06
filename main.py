from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from utils import *
from models import *

import tensorflow as tf
import tensorflow_datasets as tfds
from datetime import datetime
import time
import matplotlib.pyplot as plt

# 想用GPU跑就注释掉下面两行
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

tf.config.experimental_run_functions_eagerly(True)
tfds.disable_progress_bar()
# gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
# tf.config.experimental.set_memory_growth(gpus[0], True)


class UGATIT:
    def __init__(self):
        self.light = True               # [U-GAT-IT full version / U-GAT-IT light version]
        if self.light:
            self.model_name = 'UGATIT_light'
        else:
            self.model_name = 'UGATIT'
        self.epochs = 100               # The number of epochs to run
        self.already_epoch = tf.Variable(0, dtype=tf.int64)
        self.already_step = tf.Variable(0, dtype=tf.int64)
        self.step_per_epoch = 100
        # steps
        self.print_freq = 1000          # The number of image_print_freq
        self.save_freq = 1              # The number of ckpt_save_freq
        self.init_lr = 0.0003           # The learning rate
        # weight
        # self.GP_ld = 10                 # The gradient penalty lambda
        self.gan_weight = 1             # Weight about GAN
        self.cycle_weight = 10          # Weight about Cycle
        self.identity_weight = 100      # Weight about Identity
        self.cam_weight = 1000          # Weight about CAM
        # self.gan_type = 'gan'           # [gan / lsgan / wgan-gp / wgan-lp / dragan / hinge]
        self.smoothing = True           # AdaLIN smoothing effect
        #
        self.filters = 64               # base filter number per layer
        self.n_res = 4                  # The number of resblock
        self.n_dis = 6                  # The number of discriminator layer
        # self.n_critic = 1               # The number of critic
        self.sn = True                  # using spectral norm
        # self.img_size = 256             # The size of image
        # self.img_ch = 3                 # The size of image channel
        # data
        self.data_dir = "cycle_gan/horse2zebra"     # dataset_name
        self.batch_size = 1             # The size of batch size
        self.buffer_size = 300
        self.domain_A = None
        self.domain_B = None
        self.test_A = None
        self.test_B = None
        # models
        self.gen_a2b = Generator(self.filters, self.n_res, name='generator_a2b')
        self.gen_b2a = Generator(self.filters, self.n_res, name='generator_b2a')
        self.dis_a = Discriminator(self.filters, self.n_dis)
        self.dis_b = Discriminator(self.filters, self.n_dis)
        # optimizers
        self.G_A_optim = tf.keras.optimizers.Adam(self.init_lr, beta_1=0.5)
        self.G_B_optim = tf.keras.optimizers.Adam(self.init_lr, beta_1=0.5)
        self.D_A_optim = tf.keras.optimizers.Adam(self.init_lr, beta_1=0.5)
        self.D_B_optim = tf.keras.optimizers.Adam(self.init_lr, beta_1=0.5)
        # checkpoint
        self.ckpt_path = "./checkpoints/horse2zebra"     # args.ckpt_path
        self.ckpt = tf.train.Checkpoint(
            already_epoch=self.already_epoch,
            generator_a2b=self.gen_a2b,
            generator_b2A=self.gen_b2a,
            discriminator_a_local=self.dis_a.Local,
            discriminator_a_global=self.dis_a.Global,
            discriminator_b_local=self.dis_b.Local,
            discriminator_b_global=self.dis_b.Global,
            generator_A_optimizer=self.G_A_optim,
            generator_B_optimizer=self.G_B_optim,
            discriminator_A_optimizer=self.D_A_optim,
            discriminator_B_optimizer=self.D_B_optim)
        self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, self.ckpt_path, max_to_keep=3)
        # summary
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.log_dir = self.ckpt_path + '-log-' + stamp
        self.writer = tf.summary.create_file_writer(self.log_dir)
        # other
        self.load()

    @tf.function
    def train_step(self, domain_A, domain_B, step):
        with tf.GradientTape(persistent=True) as tape:
            """ Define Generator, Discriminator """
            x_ab, cam_ab, _ = self.gen_a2b(domain_A, training=False)
            x_ba, cam_ba, _ = self.gen_b2a(domain_B, training=False)
            x_aba, _, _ = self.gen_b2a(x_ab)
            x_bab, _, _ = self.gen_a2b(x_ba)
            x_aa, cam_aa, _ = self.gen_b2a(domain_A)
            x_bb, cam_bb, _ = self.gen_a2b(domain_B)
            real_A_logit, real_A_cam_logit, _, _ = self.dis_a(domain_A, training=False)
            real_B_logit, real_B_cam_logit, _, _ = self.dis_b(domain_B, training=False)
            fake_A_logit, fake_A_cam_logit, _, _ = self.dis_b(x_ba)
            fake_B_logit, fake_B_cam_logit, _, _ = self.dis_a(x_ab)

            """ Define Loss """
            G_ad_loss_A = generator_loss(fake_A_logit) + generator_loss(fake_A_cam_logit)
            G_ad_loss_B = generator_loss(fake_B_logit) + generator_loss(fake_B_cam_logit)
            D_ad_loss_A = discriminator_loss(real_A_logit, fake_A_logit) + discriminator_loss(real_A_cam_logit, fake_A_cam_logit)
            D_ad_loss_B = discriminator_loss(real_B_logit, fake_B_logit) + discriminator_loss(real_B_cam_logit, fake_B_cam_logit)
            cycle_loss_A = L1_loss(x_aba, domain_A)
            cycle_loss_B = L1_loss(x_bab, domain_B)
            identity_A = L1_loss(x_aa, domain_A)
            identity_B = L1_loss(x_bb, domain_B)
            cam_A = cam_loss(source=cam_ba, non_source=cam_aa)
            cam_B = cam_loss(source=cam_ab, non_source=cam_bb)

            Generator_A_gan = self.gan_weight * G_ad_loss_A
            Generator_A_cycle = self.cycle_weight * cycle_loss_B
            Generator_A_identity = self.identity_weight * identity_A
            Generator_A_cam = self.cam_weight * cam_A
            Generator_B_gan = self.gan_weight * G_ad_loss_B
            Generator_B_cycle = self.cycle_weight * cycle_loss_A
            Generator_B_identity = self.identity_weight * identity_B
            Generator_B_cam = self.cam_weight * cam_B

            Generator_A_loss = Generator_A_gan + Generator_A_cycle + Generator_A_identity + Generator_A_cam
            Generator_B_loss = Generator_B_gan + Generator_B_cycle + Generator_B_identity + Generator_B_cam

            Discriminator_A_loss = self.gan_weight * D_ad_loss_A
            Discriminator_B_loss = self.gan_weight * D_ad_loss_B

            Generator_loss = Generator_A_loss + Generator_B_loss
            Discriminator_loss = Discriminator_A_loss + Discriminator_B_loss

        """ Training """
        grad_gen_a = tape.gradient(Generator_A_loss, self.gen_a2b.trainable_variables)
        grad_gen_b = tape.gradient(Generator_B_loss, self.gen_b2a.trainable_variables)
        grad_dis_a = tape.gradient(Discriminator_loss, self.dis_a.Local.trainable_variables+self.dis_a.Global.trainable_variables)
        grad_dis_b = tape.gradient(Discriminator_loss, self.dis_b.Local.trainable_variables+self.dis_b.Global.trainable_variables)
        self.G_A_optim.apply_gradients(zip(grad_gen_a, self.gen_a2b.trainable_variables))
        self.G_B_optim.apply_gradients(zip(grad_gen_b, self.gen_b2a.trainable_variables))
        self.D_A_optim.apply_gradients(zip(grad_dis_a, self.dis_a.Local.trainable_variables+self.dis_a.Global.trainable_variables))
        self.D_B_optim.apply_gradients(zip(grad_dis_b, self.dis_b.Local.trainable_variables+self.dis_b.Global.trainable_variables))

        """ Summary """
        with self.writer.as_default():
            tf.summary.scalar("Generator_loss", Generator_loss, step)
            # tf.summary.scalar("G_A_0_loss", Generator_A_loss, step)
            # tf.summary.scalar("G_A_1_gan", Generator_A_gan, step)
            # tf.summary.scalar("G_A_2_cycle", Generator_A_cycle, step)
            # tf.summary.scalar("G_A_3_identity", Generator_A_identity, step)
            # tf.summary.scalar("G_A_4_cam", Generator_A_cam, step)
            # tf.summary.scalar("G_B_0_loss", Generator_B_loss, step)
            # tf.summary.scalar("G_B_1_gan", Generator_B_gan, step)
            # tf.summary.scalar("G_B_2_cycle", Generator_B_cycle, step)
            # tf.summary.scalar("G_B_3_identity", Generator_B_identity, step)
            # tf.summary.scalar("G_B_4_cam", Generator_B_cam, step)
            tf.summary.scalar("Discriminator_loss", Discriminator_loss, step)
            # tf.summary.scalar("D_A_loss", Discriminator_A_loss, step)
            # tf.summary.scalar("D_B_loss", Discriminator_B_loss, step)
            self.writer.flush()

    @tf.function
    def train(self):
        start = time.time()
        while self.already_step.value() < self.epochs * self.step_per_epoch:
            self.already_step.assign_add(1)
            tf.summary.trace_on()  # , profiler=True
            self.train_step(next(self.domain_A), next(self.domain_B), self.already_step)
            with self.writer.as_default():
                tf.summary.trace_export('U-GAT-IT', self.already_step, self.log_dir)
                self.writer.flush()
            if self.already_step.value() % 1 == 0:
                tf.print('.', end='')
            if self.already_step.value() % self.epochs == 0:
                self.already_epoch.assign_add(1)
                if self.already_epoch.value() % self.save_freq == 0:
                    ckpt_save_path = self.ckpt_manager.save()
                    tf.print('Saved ckpt for epoch {} at {}'.format(
                        self.already_epoch.value(), ckpt_save_path))
                # self.plot_images()
                tf.print('Epoch {} took {}s'.format(self.already_epoch.value(),
                                                    time.time() - start))
                start = time.time()

    def test(self):
        for _ in range(10):
            self.plot_images()

    def plot_images(self):
        real_A = next(self.test_A)[0]
        real_B = next(self.test_B)[0]
        fake_B, _, _ = self.gen_a2b(real_A)
        fake_A, _, _ = self.gen_b2a(real_B)
        plt.figure(figsize=(8, 8))
        contrast = 1
        images = [real_A, fake_B, real_B, fake_A]
        title = ['real_A', 'fake_B', 'real_B', 'fake_A']
        for i in range(4):
            plt.subplot(2, 2, i + 1)
            plt.title(title[i])
            if i % 2 == 0:
                plt.imshow(images[i][0])
            else:
                plt.imshow(images[i][0] * contrast)
        plt.show()

    def load(self):
        dataset, _ = tfds.load(self.data_dir, with_info=True, as_supervised=True)
        AUTOTUNE = tf.data.experimental.AUTOTUNE
        train_A, train_B = dataset['trainA'], dataset['trainB']
        # test_A, test_B = dataset['testA'], dataset['testB']
        train_A = train_A.map(preprocess, AUTOTUNE).repeat().shuffle(self.buffer_size).batch(self.batch_size)
        train_B = train_B.map(preprocess, AUTOTUNE).repeat().shuffle(self.buffer_size).batch(self.batch_size)
        # self.test_A = test_A.map(normalize, AUTOTUNE).batch(self.batch_size)
        # self.test_B = test_B.map(normalize, AUTOTUNE).batch(self.batch_size)
        self.domain_A = iter(train_A)
        self.domain_B = iter(train_B)
        # self.test_A = iter(test_A)
        # self.test_B = iter(test_B)
        print("[*] Reading checkpoints...")
        if self.ckpt_manager.latest_checkpoint:
            self.ckpt.restore(self.ckpt_manager.latest_checkpoint)
            print("[*] Success to read " + self.ckpt_manager.latest_checkpoint)
            self.already_step.assign(self.already_epoch.value() * self.step_per_epoch)
        else:
            print("[*] Failed to find a checkpoint")

    def save(self):
        self.gen_a2b.save('gen_a2b')
        self.gen_b2a.save('gen_b2a')


if __name__ == '__main__':
    gan = UGATIT()
    gan.train()
    gan.save()
