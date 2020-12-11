# coding=gbk
import os

from data_process import get_data, write_data
from model import Generator, Discriminator
from image_utils import save_images, save_images_batch
from image_utils import sample_noise

import time
import tensorflow as tf
import numpy as np

batch_size = 96
save_interval = 5


class Train:
    """
    ...
    """

    def __init__(self, data_path, epochs=210):
        self.epochs = epochs
        self.gen_optimizer = tf.keras.optimizers.Adam(0.0002, 0.5)
        self.disc_optimizer = tf.keras.optimizers.Adam(0.0002, 0.5)
        self.generator = Generator()
        self.discriminator = Discriminator()
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.condition_weekend = np.array([[1, 0]]).repeat(batch_size, axis=0)  # weekend
        self.condition_workday = np.array([[0, 1]]).repeat(batch_size, axis=0)  # workday
        self.monthly_parking_rate = get_data(data_path)
        self.seed = sample_noise(batch_size)  # tf.random.normal([batch_size, 1], 0.5, 0.2)
        self.avg_weekend, self.avg_workday = self.get_average(self.monthly_parking_rate)

    def __call__(self, save_path):
        """
        ...
        """
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        time_consumed_total = 0.
        for epoch in range(1, self.epochs + 1):
            start = time.time()
            total_gen_loss = 0
            total_disc_loss = 0
            for daily_parking_rate, condition in self.monthly_parking_rate:
                gen_loss, disc_loss = self.train_step(daily_parking_rate, condition)
                total_gen_loss += gen_loss
                total_disc_loss += disc_loss

            time_consumed = time.time() - start
            time_consumed_total += time_consumed
            time_consumed_agv = time_consumed_total / epoch
            self.epochs_last = self.epochs - epoch
            estimate_time_last = self.epochs_last * time_consumed_agv
            print('Time for epoch {}/{} is {} sec - gen_loss = {}, disc_loss = {}, time estimated to finish: {}'
                  .format(epoch, self.epochs, time.time() - start,
                          total_gen_loss / batch_size,
                          total_disc_loss / batch_size,
                          estimate_time_last))

            if epoch % save_interval == 0:
                save_images(save_path, epoch, self.generator, self.seed, self.avg_weekend, self.avg_workday)
                if epoch > 150:
                    self.generate(save_path, epoch)
                    self.save_model(save_path, time_consumed_total)

        self.save_model(save_path, time_consumed_total)

    def load_model(self, save_path):
        self.generator.load_weights(save_path + '/model_generator_weight')
        self.discriminator.load_weights(save_path + '/model_discriminator_weight')
        print('models from ' + save_path + ' recovered. ')

    def save_model(self, save_path, time_consumed_total):
        gen_data = np.concatenate(
            (np.reshape(self.generator(self.seed, self.condition_weekend, False), (1, batch_size)),
             np.reshape(self.generator(self.seed, self.condition_workday, False), (1, batch_size))),
            axis=0)
        write_data(save_path, gen_data, batch_size, 2, 'final')
        self.generator.save_weights(save_path + '/model_generator_weight')
        self.discriminator.save_weights(save_path + '/model_discriminator_weight')
        print('models saved into path: ' + save_path + ', total time consumed: %s' % time_consumed_total)

    def generate(self, save_path, epoch='', cases=3):
        """
        ...
        """
        for i in range(cases):
            seed = sample_noise(batch_size)
            save_images_batch(save_path, 'batch' + str(epoch), self.generator, seed, self.avg_weekend, self.avg_workday,
                              i == cases - 1)

    def train_step(self, inputs, condition_):
        """
        ...
        """
        with tf.GradientTape(persistent=True) as tape:
            noise_ = sample_noise(batch_size)
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
        weekend_total, workday_total = tf.zeros((batch_size, 1)), tf.zeros((batch_size, 1))
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

    # todo: this method may be over-fitting, try split into training and testing data to confirm
    #  (complex models may cause this)
    
    # disable GPU
    tf.config.set_visible_devices([], 'GPU')

    # enable GPU
    # physical_devices = tf.config.list_physical_devices('GPU')
    # tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

    # save_path_ = './generated/1602738298_256_cpu'
    save_path_ = "./generated/%s" % int(time.time())
    train = Train('./data', 500)
    train(save_path_)
    # train.load_model(save_path_)
    # train.generate(save_path_, 'final')
