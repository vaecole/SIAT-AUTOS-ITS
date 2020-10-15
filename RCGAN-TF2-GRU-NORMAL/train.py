# coding=gbk
import os
import random

from data_process import get_data, write_data
from model import Generator, Discriminator
from image_utils import save_images

import time
import tensorflow as tf
import numpy as np

epochs = 800
batch_size = 96
save_interval = 5


class Train:
    """
    ...
    """

    def __init__(self):
        self.gen_optimizer = tf.keras.optimizers.Adam(0.0002, 0.5)
        self.disc_optimizer = tf.keras.optimizers.Adam(0.0002, 0.5)
        self.generator = Generator()
        self.discriminator = Discriminator()
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.condition_weekend = np.array([[1, 0]]).repeat(96, axis=0)  # weekend
        self.condition_workday = np.array([[0, 1]]).repeat(96, axis=0)  # workday

    def call(self, data_path, save_path):
        """
        ...
        """
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        monthly_parking_rate = get_data(data_path)
        seed = tf.random.normal([96, 1], 0.5, 0.2)
        time_consumed_total = 0.
        avg_weekend, avg_workday = self.get_average(monthly_parking_rate)
        for epoch in range(1, epochs + 1):
            start = time.time()
            total_gen_loss = 0
            total_disc_loss = 0
            for daily_parking_rate, condition in monthly_parking_rate:
                gen_loss, disc_loss = self.train_step(daily_parking_rate, condition)
                total_gen_loss += gen_loss
                total_disc_loss += disc_loss

            time_consumed = time.time() - start
            time_consumed_total += time_consumed
            time_consumed_agv = time_consumed_total / epoch
            epochs_last = epochs - epoch
            estimate_time_last = epochs_last * time_consumed_agv
            print('Time for epoch {}/{} is {} sec - gen_loss = {}, disc_loss = {}, time estimated to finish: {}'
                  .format(epoch, epochs, time.time() - start,
                          total_gen_loss / batch_size,
                          total_disc_loss / batch_size,
                          estimate_time_last))

            if epoch % save_interval == 0:
                save_images(save_path, epoch, self.generator, seed, avg_weekend, avg_workday)
                if epoch > 199:
                    gen_data = np.concatenate((np.reshape(self.generator(seed, self.condition_weekend, False),
                                                          (1, 96)),
                                               np.reshape(self.generator(seed, self.condition_workday, False),
                                                          (1, 96))), axis=0)
                    write_data(save_path, gen_data, 96, 2, epoch)

        gen_data = np.concatenate((np.reshape(self.generator(seed, self.condition_weekend, False), (1, 96)),
                                   np.reshape(self.generator(seed, self.condition_workday, False), (1, 96))), axis=0)
        write_data(save_path, gen_data, 96, 2, 'final')
        self.generator.save_weights(save_path + '/model_generator_weight')
        self.discriminator.save_weights(save_path + '/model_discriminator_weight')
        print('models saved into path: ' + save_path + ', total time consumed: %s' % time_consumed_total)

    def start_training(self, data_path, save_path, load_model=False):
        """
        ...
        """
        if load_model:
            self.generator.load_weights(save_path + '/model_generator_weight')
            self.discriminator.load_weights(save_path + '/model_discriminator_weight')
            print('models from ' + save_path + ' recovered. ')
        self.call(data_path, save_path)

    def train_step(self, inputs, condition_):
        """
        ...
        """
        with tf.GradientTape(persistent=True) as tape:
            noise_ = tf.random.normal([96, 1], 0.5, 0.2)
            generated_image = self.generator(noise_, condition_)
            real_output = self.discriminator(inputs, condition_)
            generated_output = self.discriminator(generated_image, condition_)

            loss_g = self.generator_loss(self.cross_entropy, generated_output)
            loss_d = self.discriminator_loss(self.cross_entropy, real_output, generated_output)

        grad_gen = tape.gradient(loss_g, self.generator.trainable_variables)
        grad_disc = tape.gradient(loss_d, self.discriminator.trainable_variables)

        self.gen_optimizer.apply_gradients(zip(grad_gen, self.generator.trainable_variables))
        self.disc_optimizer.apply_gradients(zip(grad_disc, self.discriminator.trainable_variables))

        return loss_g, loss_d

    @staticmethod
    def discriminator_loss(loss_object, real_output, fake_output):
        """
        ...
        """
        real_loss = loss_object(tf.ones_like(real_output), real_output)
        fake_loss = loss_object(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    @staticmethod
    def generator_loss(loss_object, fake_output):
        """
        ...
        """
        return loss_object(tf.ones_like(fake_output), fake_output)

    @staticmethod
    def get_average(monthly_parking_rate):
        """
        ...
        """
        weekends, workdays = 0, 0
        weekend_total, workday_total = tf.zeros((96, 1)), tf.zeros((96, 1))
        for daily_parking_rate, condition in monthly_parking_rate:
            if condition[0][0] == 1:
                weekend_total += daily_parking_rate
                weekends += 1
            else:
                workday_total += daily_parking_rate
                workdays += 1
        weekend_total /= weekends
        workday_total /= workdays
        return weekend_total, workday_total


if __name__ == "__main__":
    # disable GPU
    # tf.config.set_visible_devices([], 'GPU')

    # enable CPU
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

    # save_path_ = './generated/images1602599186'
    save_path_ = "./generated/%s" % int(time.time())
    Train().start_training('./data', save_path_, load_model=False)
