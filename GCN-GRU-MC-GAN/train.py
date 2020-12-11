import os
from tensorflow.keras.optimizers import Adam
from spektral.utils import normalized_laplacian
from model import Generator, Discriminator
import time
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import utils

batch_size = 96 * 7
lr = 0.0001
adam_beta_1 = 0.5
save_week_interval = 10
save_all_interval = 100
dropout = 0.5
alpha = 0.2


class Train:
    def __init__(self, seqs, adj, nodes_features, epochs, key, use_gcn):
        self.epochs = epochs
        self.seqs = seqs.astype('float32')
        self.key = key

        self.gen_optimizer = Adam(lr, adam_beta_1)
        self.desc_optimizer = Adam(lr, adam_beta_1)

        self.adj = normalized_laplacian(adj.astype('float32'))
        self.nodes_features = nodes_features.astype('float32')
        self.generator = Generator.make_model(use_gcn, dropout, batch_size, self.adj, self.nodes_features)
        self.discriminator = Generator.make_model(use_gcn, dropout, batch_size, self.adj, self.nodes_features)
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    def __call__(self, epochs=None, save_path='generated/'):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if epochs is None:
            epochs = self.epochs

        time_len = self.seqs.shape[0]
        num_nodes = self.seqs.shape[1]
        total_batch = int(time_len / batch_size)  # 2976/96=31

        time_consumed_total = 0.
        for epoch in range(1, epochs + 1):
            start = time.time()
            total_gen_loss = 0
            total_disc_loss = 0

            for week in range(0, total_batch):
                current_seqs = self.seqs[week * batch_size:week * batch_size + batch_size]
                seqs_noised = current_seqs.copy()
                max_s = current_seqs[self.key].max()
                seqs_noised[self.key] = np.random.normal(max_s / 2.0, max_s / 10.0,
                                                         size=(current_seqs.shape[0])).astype('float32')
                # current_seqs.plot(figsize=(20,5))
                # seqs_noised.plot(figsize=(20,5))
                gen_loss, disc_loss = self.train_step(current_seqs, seqs_noised)
                total_gen_loss += gen_loss
                total_disc_loss += disc_loss

            time_consumed = time.time() - start
            time_consumed_total += time_consumed
            time_consumed_agv = time_consumed_total / epoch
            epochs_last = epochs - epoch
            estimate_time_last = epochs_last * time_consumed_agv
            print('epoch {}({})/{}({}) - gen_loss = {}, disc_loss = {}, estimated to finish: {}'
                  .format(epoch, round(time.time() - start, 2),
                          epochs, round(time_consumed_total, 2),
                          round(float(total_gen_loss / total_batch), 5),
                          round(float(total_disc_loss / total_batch), 5),
                          round(estimate_time_last, 2)))

            if epoch % save_week_interval == 0:
                self.compare_plot('week_' + str(epoch), save_path, int(total_batch / 2) * 7)
            if epoch % save_all_interval == 0:
                self.compare_plot('all_' + str(epoch), save_path, 0, total_batch)
        self.save_model(save_path, time_consumed_total)

    @tf.function
    def train_step(self, seqs, seqs_noised):
        with tf.GradientTape(persistent=True) as tape:
            real_output = tf.squeeze(self.discriminator(tf.expand_dims(seqs, axis=0)), axis=0)  # 评价高
            generated = tf.squeeze(self.generator(tf.expand_dims(seqs_noised, axis=0)), axis=0)
            left = tf.slice(seqs, [0, 0], [batch_size, self.key])
            right = tf.slice(seqs, [0, self.key + 1], [batch_size, -1])
            combined = tf.concat([left, generated, right], 1)
            generated_output = tf.squeeze(self.discriminator(tf.expand_dims(combined, axis=0)), axis=0)  # 初始评价低

            loss_g = self.generator_loss(self.cross_entropy, generated_output)
            loss_d = self.discriminator_loss(self.cross_entropy, real_output, generated_output)

        grad_gen = tape.gradient(loss_g, self.generator.trainable_variables)
        grad_disc = tape.gradient(loss_d, self.discriminator.trainable_variables)

        self.gen_optimizer.apply_gradients(zip(grad_gen, self.generator.trainable_variables))
        self.desc_optimizer.apply_gradients(zip(grad_disc, self.discriminator.trainable_variables))

        return loss_g, loss_d

    def generate(self, real_seqs):
        seqs_replace = real_seqs.copy()
        max_s = seqs_replace[self.key].max()
        seqs_replace[self.key] = np.random.normal(max_s / 2.0, max_s / 10.0, size=(seqs_replace.shape[0])).astype(
            'float32')
        gen_data = tf.squeeze(self.generator(tf.expand_dims(seqs_replace, axis=0), training=False), axis=0)
        return pd.DataFrame(gen_data.numpy())

    @staticmethod
    def discriminator_loss(loss_object, real_output, fake_output):
        real_loss = loss_object(tf.ones_like(real_output), real_output)
        fake_loss = loss_object(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    @staticmethod
    def generator_loss(loss_object, fake_output):
        return loss_object(tf.ones_like(fake_output), fake_output)

    def compare_plot(self, name, save_path, start_day=0, week=1):
        fig, ax = plt.subplots()
        fig.set_figheight(8)
        fig.set_figwidth(20)
        real_seqs = self.seqs[start_day:start_day + batch_size * week]
        generated_seqs = []
        for w in range(week):
            generated_seq = self.generate(real_seqs[start_day * w:start_day * w + batch_size])
            if len(generated_seqs) == 0:
                generated_seqs = generated_seq
            else:
                generated_seqs = generated_seqs.append(generated_seq, ignore_index=True)
        generated_seqs = utils.max_min_scale(generated_seqs)
        all_seqs = pd.concat([pd.DataFrame(real_seqs[self.key].values), generated_seqs], axis=1)
        all_seqs.plot(ax=ax)
        n = 2
        ax.legend(['real' + str(w) for w in range(1, n)] + ['gen' + str(w) for w in range(1, n)]);
        fig.savefig(save_path + "/compare_" + name + ".png")
        plt.close()

    def load_model(self, save_path):
        """
        ...
        """
        self.generator.load_weights(save_path + '/model_generator_weight')
        self.discriminator.load_weights(save_path + '/model_discriminator_weight')
        print('models from ' + save_path + ' recovered. ')

    def save_model(self, save_path, time_consumed_total):
        """
        ...
        """
        self.generator.save_weights(save_path + '/model_generator_weight')
        self.discriminator.save_weights(save_path + '/model_discriminator_weight')
        print('models saved into path: ' + save_path + ', total time consumed: %s' % time_consumed_total)


def start_train(epochs=10000, use_gcn=True, target_park='宝琳珠宝交易中心', start='2016-06-02', end='2016-07-07'):
    seqs_normal, adj, node_f, key = utils.init_data(target_park, start, end)
    # seqs_normal[key].plot(figsize=(20,10))
    name = target_park + '_GCN_' + str(use_gcn)
    print('Start ' + name, seqs_normal.shape, key)
    train = Train(seqs_normal, adj, node_f, epochs, key, use_gcn)
    print(train.generator.summary())
    save_path = 'generated/' + str(epochs) + name + str(time.time())
    train(epochs, save_path)
    print('Finished ' + name)
    return train


def resume_train(model_path, epochs=10000, use_gcn=True, target_park='宝琳珠宝交易中心',
                 start='2016-06-02', end='2016-07-07'):
    seqs_normal, adj, node_f, key = utils.init_data(target_park, start, end)
    # seqs_normal[key].plot(figsize=(20,10))
    name = target_park + '_GCN_' + str(use_gcn)
    print('Start ' + name, seqs_normal.shape, key)
    train = Train(seqs_normal, adj, node_f, epochs, key)
    train.load_model(model_path)
    train(epochs, model_path)
    print('Finished ' + name)
    return train


if __name__ == "__main__":
    # disable GPU
    # tf.config.set_visible_devices([], 'GPU')

    # enable GPU
    # physical_devices = tf.config.list_physical_devices('GPU')
    # tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

    train1 = start_train(10000, True, '宝琳珠宝交易中心', '2016-07-26', '2016-09-06')
    train2 = start_train(10000, False, '宝琳珠宝交易中心', '2016-07-26', '2016-09-06')
    # resume_train('./generated/1602738298_256_cpu')
