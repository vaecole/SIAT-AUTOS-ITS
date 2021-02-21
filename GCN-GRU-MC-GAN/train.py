import itertools
import json
import os
from tensorflow.keras.optimizers import Adam
from spektral.utils import normalized_laplacian
from tqdm import tqdm

import math
import losses
import model
import time
import tensorflow as tf
import numpy as np
import pandas as pd
import utils
import metrics

EPOCHS = 2000
total_weeks = 5
fix_weeks = 5
use_gpu = True
root_path = 'generated/' + str(total_weeks) + 'weeks' + ('_gpu' if use_gpu else '') + '_wgan_%d/'% EPOCHS
lr = 0.0001
adam_beta_1 = 0.5
evaluate_interval = EPOCHS / 20


class Train:
    def __init__(self, seqs, adj, nodes_features, epochs, key, use_gcn, batch_size, use_gru=True):
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
        self.generator = model.make_generator('generator', batch_size, self.adj, self.nodes_features, use_gcn, use_gru)
        self.discriminator = model.make_discriminator('discriminator', batch_size, self.adj, self.nodes_features,
                                                      use_gcn,
                                                      use_gru)
        self.d_loss_fn, self.g_loss_fn = losses.get_wasserstein_losses_fn()

    def __call__(self, epochs=None, save_path='generated/', batch_size=96):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if epochs is None:
            epochs = self.epochs

        time_len = self.seqs.shape[0]
        total_batch = int(time_len / batch_size)

        time_consumed_total = 0.
        final_epoch = epochs
        stable = 0
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
                gen_loss, disc_loss = self.train_step(current_seqs, seqs_noised, batch_size)
                total_gen_loss += gen_loss
                total_disc_loss += disc_loss

            time_consumed = time.time() - start
            time_consumed_total += time_consumed
            time_consumed_agv = time_consumed_total / epoch
            epochs_last = epochs - epoch
            estimate_time_last = epochs_last * time_consumed_agv
            if epoch % evaluate_interval == 0:
                metrics_ = self.evaluate(save_path, batch_size, epoch)
                metrics_['gen_loss'] = round(float(total_gen_loss / total_batch), 3)
                metrics_['disc_loss'] = round(float(total_disc_loss / total_batch), 3)
                print('epoch {}/{}({})-{}, to finish： {}'
                      .format(epoch, epochs, round(time_consumed_total, 2),
                              json.dumps(metrics_, ensure_ascii=False),
                              round(estimate_time_last, 2)))
                if not metrics_.keys().__contains__('RMSE'):
                    metrics_['crash'] = True
                    break
                if metrics_['WSSTD'] < 0.03 and metrics_['RMSE'] < 0.1:
                    stable += 1
                    # if stable > 2:
                    #     final_epoch = epoch
                    #     break
                else:
                    stable = 0
        self.save_model(save_path, time_consumed_total)
        return time_consumed_total, final_epoch

    @tf.function
    def train_step(self, seqs, seqs_noised, batch_size):
        with tf.GradientTape(persistent=True) as tape:
            real_output = self.call_model(self.discriminator, seqs)
            generated = self.call_model(self.generator, seqs_noised)
            left = tf.slice(seqs, [0, 0], [batch_size, self.key])
            right = tf.slice(seqs, [0, self.key + 1], [batch_size, -1])
            combined = tf.concat([left, generated[0], right], 1)
            generated_output = self.call_model(self.discriminator, combined)
            loss_g = self.g_loss_fn(generated_output)
            loss_d = self.d_loss_fn(real_output, generated_output)
        grad_gen = tape.gradient(loss_g, self.generator.trainable_variables)
        grad_disc = tape.gradient(loss_d, self.discriminator.trainable_variables)
        self.gen_optimizer.apply_gradients(zip(grad_gen, self.generator.trainable_variables))
        self.desc_optimizer.apply_gradients(zip(grad_disc, self.discriminator.trainable_variables))

        return loss_g, loss_d

    def call_model(self, model_, seqs):
        return model_(inputs=[tf.expand_dims(seqs, axis=0), self.nodes_f_expanded, self.adj_expanded])

    def generate(self, real_seqs):
        seqs_replace = real_seqs.copy()
        max_s = seqs_replace[self.key].max()
        seqs_replace[self.key] = np.random.normal(max_s / 2.0, max_s / 10.0, size=(seqs_replace.shape[0])).astype(
            'float32')
        gen_data = tf.squeeze(self.generator(tf.expand_dims(seqs_replace, axis=0), training=False), axis=0)
        return pd.DataFrame(gen_data.numpy())

    def get_compare(self, start_day, batch_size):
        real_seqs = self.seqs[start_day:start_day + batch_size]
        noise_seq = self.seqs_noised[start_day:start_day + batch_size]
        generated_seqs = self.call_model(self.generator, noise_seq).numpy()[0]
        return real_seqs[self.key].values, generated_seqs

    def evaluate(self, save_path, batch_size, name=None, restore_model=False):
        if restore_model:
            self.try_restore(save_path)
        start_day = 0
        real, generated = self.get_compare(start_day, batch_size)
        if math.isnan(generated[0][0]):
            return dict()

        utils.compare_plot(name, save_path, real, generated)
        return metrics.get_common_metrics(real.reshape(1, -1)[0], generated.reshape(1, -1)[0])

    def try_restore(self, base_path):
        dirs = os.listdir(base_path)
        assert dirs
        load_path = os.path.join(base_path, dirs[0])
        self.load_model(load_path)

    def load_model(self, save_path):
        print('try recover models from ' + save_path + '. ')
        self.generator.load_weights(save_path + '/model_generator_weight')
        self.discriminator.load_weights(save_path + '/model_discriminator_weight')
        print('models from ' + save_path + ' recovered. ')

    def save_model(self, save_path, time_consumed_total):
        self.generator.save_weights(save_path + '/model_generator_weight')
        self.discriminator.save_weights(save_path + '/model_discriminator_weight')
        print('models saved into path: ' + save_path + ', total time consumed: %s' % time_consumed_total)


def start_train(epochs=10000, target_park='宝琳珠宝交易中心', start='2016-01-02', end='2017-01-02'):
    seqs_normal, adj, node_f, nks, conns, _ = utils.init_data(target_park, start, end)
    batch_size = 96 * 7 * fix_weeks
    seqs_normal = seqs_normal.take(range(96 * 7 * 0, 96 * 7 * total_weeks))
    use_gru_bag = [True, False]
    use_gcn_bag = [True, False]
    for (use_gcn, use_gru) in itertools.product(use_gcn_bag, use_gru_bag):
        name = target_park + ('_GCN' if use_gcn else '') + ('_GRU' if use_gru else '')
        print('Starting ' + name)
        site_path = root_path + name
        if not os.path.exists(site_path):
            os.makedirs(site_path)
        else:
            continue
        describe_site(nks, seqs_normal, site_path, target_park, node_f)
        train = Train(seqs_normal, adj, node_f, epochs, nks[target_park], use_gcn, batch_size, use_gru)
        # print(train.generator.summary())
        start = time.time()
        save_path = site_path + '/' + str(start)
        os.makedirs(save_path)
        train_time_consumed, final_epoch = train(epochs, save_path, batch_size)
        # evaluation
        metrics_ = train.evaluate(site_path, batch_size)
        metrics_['name'] = name
        metrics_['train_time_consumed'] = round(train_time_consumed, 2)
        metrics_['final_epoch'] = final_epoch
        metrics.write_metrics(root_path, metrics_)


def describe_site(nks, seqs_normal, site_path, target_park, node_f):
    corrs = seqs_normal.corr(method="pearson")[nks[target_park]]
    node_f.to_csv(site_path + '/metrics_desc.txt', encoding='utf_8_sig', index=False)
    metrics.write_metrics(site_path, {'avg_corrs': round(sum(corrs) / len(corrs), 3),
                                      'index': nks[target_park]}, '_desc')


if __name__ == "__main__":
    if use_gpu:
        # enable GPU
        physical_devices = tf.config.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
    else:
        # disable GPU
        tf.config.set_visible_devices([], 'GPU')

    # invalid: to many neighbors: '都心名苑', '新白马', '丰园酒店', '红围坊停车场', '天元大厦', '同乐大厦', '万达丰大厦', '文锦广场',
    # '银都大厦', '永新商业城', '中信星光明庭管理处',
    good_sites = ['都市名园', '华瑞大厦', '红围坊停车场', '银都大厦', '宝琳珠宝交易中心']

    sites = ['都市名园', '华瑞大厦', '东翠花园', '化工大厦', '武警生活区银龙花园', '中深石化大厦', '翠景山庄', '万山珠宝工业园', '桂龙家园']
    big_sites = ['都心名苑', '新白马', '丰园酒店', '红围坊停车场', '天元大厦', '同乐大厦', '万达丰大厦', '文锦广场', '银都大厦', '永新商业城', '中信星光明庭管理处']

    for site in tqdm(good_sites):
        start_train(EPOCHS, site)
