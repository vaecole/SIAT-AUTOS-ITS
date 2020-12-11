
import pandas as pd
pd.options.mode.chained_assignment = None
from sklearn import preprocessing
from graph_utils import build_graph

min_max_scaler = preprocessing.MinMaxScaler()

def build_area_seqs(target_area, nks, start='2016-08-01', end='2017-01-01'):
    # 整合到一个文件中
    area_df = pd.DataFrame()
    for name in target_area.parking_name:
        file_name = 'generated/data/seqs/'+name+'_seq.csv'
        file_df = pd.read_csv(file_name)
        file_df['parking'] = nks[name]
        cols = file_df.columns.tolist()
        cols = [cols[0], cols[2], cols[1]]
        file_df = file_df[cols]
        if len(area_df)>0:
            area_df = pd.concat([area_df, file_df])
        else:
            area_df = file_df

    out_bound_indexes = area_df[(area_df['date'] < start) | (area_df['date'] >= end)].index
    area_df.drop(out_bound_indexes, inplace = True)
    return area_df.pivot_table('occupy', ['date'], 'parking')

def get_nodes_features(area_df):
    node_f = area_df[['total_space','monthly_fee','building_type']]
    node_f.loc[:,['total_space', 'monthly_fee']] = min_max_scaler.fit_transform(node_f[['total_space', 'monthly_fee']])
    building_type_oneHot = pd.get_dummies(node_f['building_type'])
    node_f = node_f.drop('building_type',axis = 1)
    node_f = node_f.join(building_type_oneHot)
    return node_f

def max_min_scale(raw):
    raw[raw.columns.values] = min_max_scaler.fit_transform(raw[raw.columns.values])
    return raw

def init_data(target_park = '宝琳珠宝交易中心', start='2016-06-02', end='2016-07-07', graph_nodes_max_dis = 0.5):
    basic_info_df = pd.read_csv('generated/data/parkings_info.csv')
    basic_info_df['lat_long'] = list(zip(basic_info_df['latitude'], basic_info_df['longitude']))
    target_area, adj, target_map, nks, kns = build_graph(basic_info_df, target_park, max_dis=graph_nodes_max_dis)
    # target_park_basic_info = basic_info_df.loc[basic_info_df.parking_name == target_park].iloc[0]
    key = nks[target_park]
    node_f = get_nodes_features(target_area)
    seqs_raw = build_area_seqs(target_area, nks, start, end)
    seqs_normal = seqs_raw.fillna(0)
    seqs_normal = max_min_scale(seqs_normal)
    return seqs_normal, adj, node_f, key