import pandas as pd
import tensorflow as tf
import os

pd.options.mode.chained_assignment = None
from sklearn import preprocessing
from graph_utils import build_graph

min_max_scaler = preprocessing.MinMaxScaler()


def build_area_seqs(target_area, nks, start='2016-08-01', end='2017-01-01'):
    # 整合到一个文件中
    area_df_list = []
    max_start = start
    min_end = end
    for name in target_area.parking_name:
        file_name = 'generated/data/seqs/' + name + '_seq.csv'
        file_df = pd.read_csv(file_name)
        file_df['parking'] = nks[name]
        cols = file_df.columns.tolist()
        cols = [cols[0], cols[2], cols[1]]
        file_df = file_df[cols]
        avg = file_df.occupy.max() / 4
        current_start = file_df.loc[file_df.occupy >= avg].iloc[0].date
        if current_start > max_start:
            max_start = current_start
        current_end = file_df.iloc[-1].date
        if current_end < min_end:
            min_end = current_end
        area_df_list.append(file_df)
        # print(current_start, current_end, max_start, min_end)
        if max_start >= min_end:
            print(name + ' is causing invalid date range.')
            raise AssertionError(name + ' is causing invalid date range.', current_start, current_end, max_start,
                                 min_end)

    print(max_start, min_end, )
    area_df = pd.DataFrame()
    for site_df in area_df_list:
        out_bound_indexes = site_df[(site_df['date'] < max_start) | (site_df['date'] > min_end)].index
        site_df.drop(out_bound_indexes, inplace=True)
        if len(area_df) > 0:
            area_df = pd.concat([area_df, site_df])
        else:
            area_df = site_df

    return area_df.pivot_table('occupy', ['date'], 'parking')


def get_nodes_features(area_df):
    node_f = area_df[['total_space', 'monthly_fee', 'building_type']]
    node_f.loc[:, ['total_space', 'monthly_fee']] = min_max_scaler.fit_transform(node_f[['total_space', 'monthly_fee']])
    building_type_one_hot = pd.get_dummies(node_f['building_type'])
    node_f = node_f.drop('building_type', axis=1)
    node_f = node_f.join(building_type_one_hot)
    return node_f


def max_min_scale(raw):
    raw[raw.columns.values] = min_max_scaler.fit_transform(raw[raw.columns.values])
    return raw

def init_data(target_park='宝琳珠宝交易中心', start='2016-01-02', end='2017-01-02', graph_nodes_max_dis=0.5):
    basic_info_df = pd.read_csv('generated/data/parkings_info.csv')
    basic_info_df['lat_long'] = list(zip(basic_info_df['latitude'], basic_info_df['longitude']))
    target_area, adj, target_map, nks, kns, conns = build_graph(basic_info_df, target_park, max_dis=graph_nodes_max_dis)
    # target_park_basic_info = basic_info_df.loc[basic_info_df.parking_name == target_park].iloc[0]
    node_f = get_nodes_features(target_area)
    seqs_raw = build_area_seqs(target_area, nks, start, end)
    seqs_normal = seqs_raw.fillna(0)
    seqs_normal = max_min_scale(seqs_normal)
    return seqs_normal, adj, node_f, nks, conns, target_map


import matplotlib.pyplot as plt


def init_data_for_search(start='2016-01-02', end='2017-01-02', graph_nodes_max_dis=0.5, take_week=2):
    plt.rc('font', size=14)
    basic_info_df = pd.read_csv('generated/data/parkings_info.csv')
    basic_info_df['lat_long'] = list(zip(basic_info_df['latitude'], basic_info_df['longitude']))
    result = []
    for target_park in basic_info_df.parking_name.values:
        fig, ax = plt.subplots()
        fig.set_figheight(8)
        fig.set_figwidth(20)
        print('searching ' + target_park)
        target_area, adj, target_map, nks, kns, conns = build_graph(basic_info_df, target_park, graph_nodes_max_dis)
        print(nks)
        key = nks[target_park]
        node_f = get_nodes_features(target_area)
        try:
            seqs_raw = build_area_seqs(target_area, nks, start, end).take(range(96 * 7 * 0, 96 * 7 * take_week))
            seqs_normal = seqs_raw.fillna(0)
            seqs_normal = max_min_scale(seqs_normal)
            result.append([conns, target_park, seqs_normal, adj, node_f, key])
            # seqs_normal[key].plot(ax=ax)
            # fig.savefig("generated/data/2/raw_" + target_park + "_target.png")
            # plt.close()
            fig, ax = plt.subplots()
            fig.set_figheight(30)
            fig.set_figwidth(100)
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.legend(fontsize=20)
            seqs_normal.plot(ax=ax, linewidth=2, alpha=0.7)
            seqs_normal[key].plot(ax=ax, linewidth=3, alpha=0.8)
            # ax.legend(['Ground Truth'] + ['Generated'], fontsize=16)
            ax.set_ylabel('Occupied Parking Space Rate', fontsize=18)
            ax.set_xlabel('Date Time', fontsize=18)
            fig.savefig("generated/data/2/raw_" + target_park + "_area.png")
            plt.close()
        except AssertionError as ae:
            print(ae)
            continue
    return result


def compare_plot(name, save_path, real, generated):
    plt.rc('font', size=14)
    # plt.xticks(fontsize=20)
    # plt.yticks(fontsize=20)
    # plt.title('Real Data and Generated Data' + ' label ', fontsize=22)
    # plt.ylabel('Occupied Parking Space Rate', fontsize=22)
    # plt.xlabel('Time Point', fontsize=22)
    # plt.legend(fontsize=20)
    plt.grid(True)
    fig, ax = plt.subplots()
    fig.set_figheight(8)
    fig.set_figwidth(20)
    # fig.suptitle('This is the figure title', fontsize=15)
    all_seqs = pd.concat([pd.DataFrame(real), pd.DataFrame(generated)], axis=1)
    pd.DataFrame(all_seqs).plot(ax=ax, linewidth=2, alpha=0.7)
    n = 2
    # ax.legend(['real' + str(w) for w in range(1, n)] + ['gen' + str(w) for w in range(1, n)])
    ax.legend(['Ground Truth'] + ['Generated'], fontsize=16)
    ax.set_ylabel('Occupied Parking Space Rate', fontsize=18)
    ax.set_xlabel('Time Point', fontsize=18)
    fig.savefig(save_path + "/compare_" + str(name) + ".png")
    plt.close()


def dense_to_sparse(dense):
    zero = tf.constant(0, dtype=tf.float32)
    where = tf.not_equal(dense, zero)
    indices = tf.where(where)
    values = tf.gather_nd(dense, indices)
    return tf.SparseTensor(indices, values, dense.shape)


def matrix_array_to_csv(save_path, colum_name, ary):
    filename = '/matrix_hist.csv'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        pd.DataFrame().to_csv(save_path + filename, encoding='utf_8_sig', index=False)

    df = pd.read_csv(save_path + filename)
    df[colum_name] = ary
    df.to_csv(save_path + filename, encoding='utf_8_sig', index=False)
