from geopy import distance
import pandas as pd
import folium
import numpy as np


# build area for points near than max_dist to target
def get_area(target, max_dis, all_df):
    area_data = []
    for index, row in all_df.iterrows():
        if distance.distance(row.lat_long, target.lat_long).km < max_dis:
            area_data.append(row.values.tolist())

    return pd.DataFrame(area_data, columns=all_df.columns.values.tolist())


# build connection between points with length ---- not near than max_dist
def build_adj_matrix(area, max_dist=0.5):
    adj_matrix = []
    name_key_dict = dict()
    key_name_dict = dict()
    conns = []
    key = 0
    for lat1, long1, name1 in zip(area.latitude, area.longitude, area.parking_name):
        name_key_dict[name1] = key
        key_name_dict[key] = name1
        key += 1
        one_line = []
        for lat2, long2, name2 in zip(area.latitude, area.longitude, area.parking_name):
            if name1 != name2:  # and distance.distance((lat1, long1), (lat2, long2)).km < max_dist:
                dist = distance.distance((lat1, long1), (lat2, long2)).km
                one_line.append(dist)
                conns.append([name1, name2, dist])
            else:
                one_line.append(0)
        adj_matrix.append(one_line)
    return adj_matrix, name_key_dict, key_name_dict, conns


def build_adj_map(target, area_, adj_mat, key_names, name_keys, conn_dis):
    # Instantiate a feature group for the parkings in the dataframe
    parkings = folium.map.FeatureGroup()

    # Loop through and add each to the parkings feature group
    for lat, lng, name in zip(area_.latitude, area_.longitude, area_.parking_name):
        parkings.add_child(
            folium.CircleMarker(
                [lat, lng],
                radius=8,  # define how big you want the circle markers to be
                color='yellow',
                fill=True,
                fill_color=('red' if name == target.parking_name else 'blue'),
                fill_opacity=0.6,
                popup=name,
                encode='uft-8'))
        neighbor_index = 0
        for is_neighbor in adj_mat[name_keys[name]]:
            if 0 < is_neighbor < conn_dis:  # and target.parking_name == name:
                start = (lat, lng)
                end_parking = area_.loc[area_["parking_name"] == key_names[neighbor_index]].iloc[0]
                end = (end_parking.latitude, end_parking.longitude)
                parkings.add_child(folium.PolyLine([start, end], color="green", weight=3, opacity=0.2))
            neighbor_index += 1

    luohu_map = folium.Map(location=[target.latitude, target.longitude], zoom_start=16)
    luohu_map.add_child(parkings)
    return luohu_map


def build_graph(basic_info_df, parking_name, max_dis=0.5, conn_coe=8 / 10):
    target_parking = basic_info_df.loc[basic_info_df.parking_name == parking_name].iloc[0]
    area = get_area(target_parking, max_dis, basic_info_df)
    adj_mat, nks, kns, conns = build_adj_matrix(area, max_dis * conn_coe)
    target_map = build_adj_map(target_parking, area, adj_mat, kns, nks, max_dis * conn_coe)
    return area, np.mat(adj_mat), target_map, nks, kns, conns
