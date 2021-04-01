import itertools
import json
import os
from tensorflow.keras.optimizers import SGD
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

EPOCHS = [5000]
total_weeks = 2
fix_weeks = 2
use_gpu = True
lr = 0.0002
adam_beta_1 = 0.1


class Train:
    def __init__(self, seqs, adj, nodes_features, epochs, key, use_gcn, batch_size, use_gru=True):
        self.epochs = epochs
        self.seqs = seqs.astype('float32')
        self.seqs_noised = seqs.copy().astype('float32')
        self.max_s = seqs[key].max()
        self.seqs_noised[key] = np.random.normal(self.max_s / 2.0, self.max_s / 10.0, size=(seqs.shape[0])).astype(
            'float32')
        self.key = key

        self.gen_optimizer = SGD(lr, adam_beta_1)
        self.desc_optimizer = SGD(lr, adam_beta_1)

        self.adj = normalized_laplacian(adj.astype('float32'))
        self.adj_expanded = tf.expand_dims(normalized_laplacian(adj.astype('float32')), axis=0)
        self.nodes_features = nodes_features.astype('float32')
        self.nodes_f_expanded = tf.expand_dims(nodes_features.astype('float32'), axis=0)
        self.generator = model.make_generator('generator', batch_size, self.adj, self.nodes_features, use_gcn, use_gru)
        self.discriminator = model.make_discriminator('discriminator', batch_size, self.adj, self.nodes_features,
                                                      use_gcn,
                                                      use_gru)
        self.d_loss_fn, self.g_loss_fn = losses.get_wasserstein_losses_fn()
        self.wsst_hist = []
        self.var_hist = []
        self.rmse_hist = []
        self.mae_hist = []
        self.r2_hist = []
        self.g_loss_hist = []
        self.d_loss_hist = []

    def __call__(self, epochs=None, save_path='generated/', batch_size=96, monitor=True):
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
            if epoch % 10 == 0:
                metrics_ = self.evaluate(save_path, batch_size, epoch, plot_compare=False)
                self.wsst_hist.append(metrics_.get('WSSTD'))
                self.rmse_hist.append(metrics_.get('RMSE'))
                self.mae_hist.append(metrics_.get('MAE'))
                self.r2_hist.append(metrics_.get('R^2'))
                self.var_hist.append(metrics_.get('Var'))
                self.g_loss_hist.append(round(float(total_gen_loss / total_batch), 3))
                self.d_loss_hist.append(round(float(total_disc_loss / total_batch), 3))

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
                if monitor and metrics_['WSSTD'] < 0.02 and metrics_['RMSE'] < 0.09:
                    stable += 1
                    if stable > 2:
                        final_epoch = epoch
                        break
                else:
                    stable = 0

                matrix_hist = pd.DataFrame()
                matrix_hist['wsst_hist'] = self.wsst_hist
                matrix_hist['rmse_hist'] = self.rmse_hist
                matrix_hist['mae_hist'] = self.mae_hist
                matrix_hist['r2_hist'] = self.r2_hist
                matrix_hist['var_hist'] = self.var_hist
                matrix_hist['g_loss_hist'] = self.g_loss_hist
                matrix_hist['d_loss_hist'] = self.d_loss_hist
                matrix_hist.to_csv(save_path + '/matrix_hist.csv', encoding='utf_8_sig', index=False)

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
        loss_dif = abs(loss_d - loss_g)
        if loss_dif <= 10:
            self.apply_grad(self.generator, tape, loss_g)
            self.apply_grad(self.discriminator, tape, loss_d)
        else:
            if loss_d > loss_g:
                self.apply_grad(self.discriminator, tape, loss_d)
            else:
                self.apply_grad(self.generator, tape, loss_g)

        return loss_g, loss_d

    def apply_grad(self, d_or_g, tape, loss):
        grad = tape.gradient(loss, d_or_g.trainable_variables)
        self.desc_optimizer.apply_gradients(zip(grad, d_or_g.trainable_variables))

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
        self.seqs_noised[self.key] = np.random.normal(self.max_s / 2.0, self.max_s / 10.0,
                                                      size=(self.seqs.shape[0])).astype('float32')
        noise_seq = self.seqs_noised[start_day:start_day + batch_size]
        generated_seqs = self.call_model(self.generator, noise_seq).numpy()[0]
        return real_seqs[self.key].values, generated_seqs

    def evaluate(self, save_path, batch_size, name=None, restore_model=False, plot_compare=True):
        if restore_model:
            self.load_model(save_path)
        start_day = 0
        real, generated = self.get_compare(start_day, batch_size)
        if math.isnan(generated[0][0]):
            return dict()

        if plot_compare:
            utils.compare_plot(name, save_path, real, generated)
        return metrics.get_common_metrics(real.reshape(1, -1)[0], generated.reshape(1, -1)[0])

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
    monitor = False
    for (use_gru, use_gcn) in itertools.product(use_gru_bag, use_gcn_bag):
        name = target_park + '/GAN' + ('_GCN' if use_gcn else '') + ('_GRU' if use_gru else '')
        print('Starting ' + name)
        site_path = root_path + name
        if not os.path.exists(site_path):
            os.makedirs(site_path)
        else:
            continue
        describe_site(nks, seqs_normal, site_path, target_park, node_f)
        train = Train(seqs_normal, adj, node_f, epochs, nks[target_park], use_gcn, batch_size, use_gru)
        # print(train.generator.summary())
        train_time_consumed, final_epoch = train(epochs, site_path, batch_size, monitor)
        if final_epoch < epochs:
            epochs = final_epoch
            monitor = False
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
    sites = ['都市名园', '华瑞大厦', '东翠花园', '化工大厦', '武警生活区银龙花园', '中深石化大厦', '翠景山庄', '万山珠宝工业园', '桂龙家园', '宝琳珠宝交易中心']
    big_sites = ['都心名苑', '新白马', '丰园酒店', '红围坊停车场', '天元大厦', '同乐大厦', '万达丰大厦', '文锦广场', '银都大厦', '永新商业城', '中信星光明庭管理处',
                 '孙逸仙心血管医院停车场', '华润万象城']
    # compare_sites = ['都市名园', '华瑞大厦', '孙逸仙心血管医院停车场', '名都大厦', '公路局大院', '合作金融大厦','国家珠宝检测中心','蔡屋围发展大厦','金园花园','雅园宾馆']
    compare_sites = ['国家珠宝检测中心', '雅园宾馆', '都市名园', '华瑞大厦', '孙逸仙心血管医院停车场', '名都大厦', '公路局大院', '合作金融大厦', '蔡屋围发展大厦', '金园花园']
    gru_advanced = ['公路局大院']

    all_sites = ['万山珠宝工业园', '万达丰大厦', '世濠大厦', '东悦名轩', '东方华都', '东方大厦', '东方颐园', '东晓综合市场', '东翠花园', '东门天地大厦', '中信星光明庭管理处',
                 '中深石化大厦', '中航凯特公寓', '丰园酒店', '丽晶大厦', '俊园大厦', '信托大厦', '公路局大院', '化工大厦', '半岛大厦', '华丽园大厦', '华安国际大酒店',
                 '华润万象城', '华瑞大厦', '华通大厦', '华隆园', '合作金融大厦', '合正星园', '同乐大厦', '名都大厦', '嘉年华名苑', '国家珠宝检测中心', '国都花园', '大信大厦',
                 '天元大厦', '孙逸仙心血管医院停车场', '宝丽大厦', '宝琳珠宝交易中心', '工人文化宫', '建设集团大院', '惠名花园', '振业大厦', '文锦广场', '新白马', '朝花大厦',
                 '柏丽花园', '桂花大厦', '桂龙家园', '武警生活区银龙花园', '永新商业城', '永通大厦', '沁芳名苑', '洪涛大厦', '深业大厦', '湖景大厦', '湖臻大厦', '物资大厦',
                 '电影大厦', '百仕达花园一期', '百仕达花园二期', '百汇大厦', '红围坊停车场', '红桂大厦', '缤纷时代家园', '翠拥华庭', '翠景山庄', '翡翠公寓', '联兴大厦',
                 '荔景大厦',
                 '蔡屋围发展大厦', '都市名园', '都心名苑', '金丰城', '金园花园', '金山大厦', '金湖文化中心', '金碧苑', '金融中心', '银都大厦', '长虹大厦', '雅园宾馆',
                 '鲲田商贸停车场', '鸿园居', '鸿基大厦', '鸿景翠峰花园', '鸿翠苑', '鹏兴花园', '鹏兴花园三期', '鹏兴花园二期', '鹏兴花园六期', '鹤围村', '龙园山庄']

    graph_good_sites = ['中信星光明庭管理处','中航凯特公寓','丽晶大厦','华丽园大厦','华安国际大酒店','华瑞大厦', '合作金融大厦','名都大厦','国家珠宝检测中心','国都花园','永新商业城','湖景大厦','物资大厦','翡翠公寓','蔡屋围发展大厦','都市名园','金丰城','长虹大厦','雅园宾馆','鸿基大厦','鹏兴花园六期']

    for EPOCH in EPOCHS:
        global root_path
        root_path = 'generated/' + str(total_weeks) + 'weeks' + (
            '_gpu' if use_gpu else '') + '_wgan_compare_less_gcn_%d_' % EPOCH + str(time.time()) + '/'
        global evaluate_interval
        evaluate_interval = EPOCH / 20
        for site in tqdm(graph_good_sites):
            start_train(EPOCH, site)
