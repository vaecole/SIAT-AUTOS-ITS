import os
from tensorflow.keras.optimizers import Adam
from spektral.utils import normalized_laplacian
import model
import time
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import utils

lr = 0.0001
adam_beta_1 = 0.5
save_week_interval = 5
save_all_interval = 20
dropout = 0.5
alpha = 0.2


class Train:
    def __init__(self, seqs, adj, nodes_features, epochs, key, use_gcn, batch_size):
        self.epochs = epochs
        self.seqs = seqs.astype('float32')
        self.seqs_noised = seqs.copy().astype('float32')
        max_s = seqs[key].max()
        self.seqs_noised[key] = np.random.normal(max_s / 2.0, max_s / 10.0, size=(seqs.shape[0])).astype('float32')
        self.key = key

        self.gen_optimizer = Adam(lr, adam_beta_1)
        self.desc_optimizer = Adam(lr, adam_beta_1)

        self.adj = normalized_laplacian(adj.astype('float32'))
        self.adj_expanded = tf.expand_dims(normalized_laplacian(adj.astype('float32')), axis=0)
        self.nodes_features = nodes_features.astype('float32')
        self.nodes_f_expanded = tf.expand_dims(nodes_features.astype('float32'), axis=0)
        self.generator = model.make_model('generator', use_gcn, dropout, batch_size, self.adj, self.nodes_features)
        self.discriminator = model.make_model('discriminator', use_gcn, dropout, batch_size, self.adj,
                                              self.nodes_features)
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    def __call__(self, epochs=None, save_path='generated/', batch_size=96):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if epochs is None:
            epochs = self.epochs

        time_len = self.seqs.shape[0]
        total_batch = int(time_len / batch_size)  # 2976/96=31

        time_consumed_total = 0.
        for epoch in range(1, epochs + 1):
            start = time.time()
            total_gen_loss = 0
            total_disc_loss = 0

            for week in range(0, total_batch):
                current_seqs = self.seqs[week * batch_size:week * batch_size + batch_size]
                seqs_noised = self.seqs_noised[week * batch_size:week * batch_size + batch_size]
                max_s = current_seqs[self.key].max()
                seqs_noised[self.key] = np.random.normal(max_s / 2.0, max_s / 10.0,
                                                         size=(current_seqs.shape[0])).astype('float32')
                # current_seqs.plot(figsize=(20, 5))
                # seqs_noised.plot(figsize=(20, 5))
                gen_loss, disc_loss = self.train_step(current_seqs, seqs_noised, batch_size)
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
                self.compare_plot('week_' + str(epoch), save_path, int(total_batch / 2) * 7, 1, batch_size)
            if epoch % save_all_interval == 0:
                self.compare_plot('all_' + str(epoch), save_path, 0, total_batch, batch_size)
        self.save_model(save_path, time_consumed_total)

    @tf.function
    def train_step(self, seqs, seqs_noised, batch_size):
        with tf.GradientTape(persistent=True) as tape:
            real_output = self.call_model(self.discriminator, seqs)  # 评价高
            generated = self.call_model(self.generator, seqs_noised)
            left = tf.slice(seqs, [0, 0], [batch_size, self.key])
            right = tf.slice(seqs, [0, self.key + 1], [batch_size, -1])
            combined = tf.concat([left, generated[0], right], 1)
            generated_output = self.call_model(self.discriminator, combined)  # 初始评价低

            loss_g = self.generator_loss(self.cross_entropy, generated_output)
            loss_d = self.discriminator_loss(self.cross_entropy, real_output, generated_output)

        grad_gen = tape.gradient(loss_g, self.generator.trainable_variables)
        grad_disc = tape.gradient(loss_d, self.discriminator.trainable_variables)

        self.gen_optimizer.apply_gradients(zip(grad_gen, self.generator.trainable_variables))
        self.desc_optimizer.apply_gradients(zip(grad_disc, self.discriminator.trainable_variables))

        return loss_g, loss_d

    def call_model(self, model, seqs):
        return model(inputs=[tf.expand_dims(seqs, axis=0), self.nodes_f_expanded, self.adj_expanded])

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

    def compare_plot(self, name, save_path, start_day=0, week=1, batch_size=96):
        fig, ax = plt.subplots()
        fig.set_figheight(8)
        fig.set_figwidth(20)
        real_seqs = self.seqs[start_day:start_day + batch_size * week]
        noise_seq = self.seqs_noised[start_day:start_day + batch_size * week]
        generated_seqs = np.concatenate(
            [self.call_model(self.generator, noise_seq[start_day * w:start_day * w + batch_size]).numpy()[0]
             for w in range(week)])
        # generated_seqs = utils.max_min_scale(pd.DataFrame(generated_seqs))
        all_seqs = pd.concat([pd.DataFrame(real_seqs[self.key].values), pd.DataFrame(generated_seqs)], axis=1)
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


def start_train(epochs=10000, use_gcn=True, target_park='宝琳珠宝交易中心', start='2016-06-02', end='2016-07-07',
                graph_nodes_max_dis=0.5):
    print('Starting ' + target_park)
    seqs_normal, adj, node_f, nks, conns, _ = utils.init_data(target_park, start, end, graph_nodes_max_dis)
    take = 96*7*8
    batch_size = 96*7
    seqs_normal = seqs_normal.take(range(take))
    name = target_park + '_' + str(use_gcn) + '_' + str(len(adj)) + '_' + str(len(seqs_normal)) + '_'
    save_path = 'generated/week_dist/' + str(epochs) + name + str(time.time())
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    fig, ax = plt.subplots()
    fig.set_figheight(20)
    fig.set_figwidth(50)
    seqs_normal.plot(ax=ax)
    fig.savefig(save_path + '/raw_' + '_'.join(nks.keys()) + '_target.png')
    plt.close()
    train = Train(seqs_normal, adj, node_f, epochs, nks[target_park], use_gcn, batch_size)
    print(train.generator.summary())
    train(epochs, save_path, batch_size)
    return train


def search_data_pattern(epochs=10000, use_gcn=True, start='2016-01-02', end='2017-01-02', graph_nodes_max_dis=0.5):
    sites = utils.init_data_for_search(start, end, graph_nodes_max_dis)
    for site in sites:
        # seqs_normal[key].plot(figsize=(20,10))
        name = site[0] + '_GCN_' + str(use_gcn) + '_' + str(len(site[2])) + '_'
        print('Start ' + name, site[1].shape, site[-1])
        train = Train(site[1], site[2], site[3], epochs, site[-1], use_gcn)
        save_path = 'generated/' + str(epochs) + name + str(time.time())
        train(epochs, save_path)
        print('Finished ' + name)
    return train


if __name__ == "__main__":

# '翠景山庄', '东翠花园', '都市名园', '都心名苑', '丰园酒店', '桂龙家园', '红围坊停车场', '洪涛大厦', '华瑞大厦',
#              '化工大厦', '天元大厦', '同乐大厦', '万达丰大厦',
#     sites = ['万山珠宝工业园', '文锦广场', '新白马', '银都大厦',
#              '永新商业城', '武警生活区银龙花园', '中深石化大厦', '中信星光明庭管理处']
    # sites = ['都市名园', '翠景山庄', '华瑞大厦', '同乐大厦', '新白马', '银都大厦', '万山珠宝工业园', '桂龙家园']
    sites = ['东翠花园', '都心名苑', '丰园酒店', '红围坊停车场', '洪涛大厦', '化工大厦', '天元大厦', '万达丰大厦',
             '文锦广场', '永新商业城', '武警生活区银龙花园', '中深石化大厦', '中信星光明庭管理处', '都市名园',
             '翠景山庄', '华瑞大厦', '同乐大厦', '新白马', '银都大厦', '万山珠宝工业园', '桂龙家园']

    # todo: build some baseline code to compare performances, build metrics to measure our advancements
    # for epochs in ():
    for site in sites:
        start_train(1000, True, site, '2016-01-02', '2017-01-02')

    # search_data_pattern(epochs=200, graph_nodes_max_dis=0.5)
    # disable GPU
    # tf.config.set_visible_devices([], 'GPU')

    # enable GPU
    # physical_devices = tf.config.list_physical_devices('GPU')
    # tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

    # train1 = start_train(5000, False, '宝琳珠宝交易中心', '2016-07-26', '2016-09-06', 0.1)
    # train2 = start_train(5000, True, '宝琳珠宝交易中心', '2016-07-26', '2016-09-06')
    # train3 = start_train(5000, False, '宝琳珠宝交易中心', '2016-07-26', '2016-09-06')
    # resume_train('./generated/1602738298_256_cpu')
