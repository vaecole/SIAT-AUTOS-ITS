# coding=gbk

from data_process import get_data, write_data
import time
import tensorflow as tf  # TF 2.0
from model import Generator, Discriminator
from image_utils import save_images

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

    def call(self, data_path, save_path):
        """
        ...
        """
        monthly_parking_rate = get_data(data_path)
        seed = tf.random.normal([96, 1], 0.5, 0.2)
        time_consumed_total = 0.
        for epoch in range(1, epochs + 1):
            start = time.time()
            total_gen_loss = 0
            total_disc_loss = 0
            real_weekend = monthly_parking_rate[0][0]
            real_workday = monthly_parking_rate[1][0]
            for daily_parking_rate, condition in monthly_parking_rate:
                gen_loss, disc_loss = self.train_step(daily_parking_rate, condition)
                total_gen_loss += gen_loss
                total_disc_loss += disc_loss

            time_consumed = time.time() - start
            time_consumed_total += time_consumed
            time_consumed_agv = time_consumed_total / epoch
            epochs_last = epochs - epoch
            estimate_time_last = epochs_last * time_consumed_agv
            print('Time for epoch {} is {} sec - gen_loss = {}, disc_loss = {}, time estimated to finish: {}'
                  .format(epoch, time.time() - start,
                          total_gen_loss / batch_size,
                          total_disc_loss / batch_size,
                          estimate_time_last))

            if epoch % save_interval == 0:
                save_images(save_path, epoch, self.generator, seed, real_weekend, real_workday)
        # write_data(save_path, num_gen_once, sample_size, cond_dim, num_run)
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


if __name__ == "__main__":
    # disable GPU
    tf.config.set_visible_devices([], 'GPU')
    # physical_devices = tf.config.list_physical_devices('GPU')
    # tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

    # save_path_ = 'generated_images1602599186'
    save_path_ = "./generated_images%s" % int(time.time())
    Train().start_training('./data', save_path_, load_model=False)
