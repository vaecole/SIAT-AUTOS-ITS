# coding=gbk
import numpy as np
import xlwt
import os
import pandas as pd


class OneHot(object):
    """
    将输入标签转换为one_hot
    """

    def __init__(self, batch_size, total_types=2):
        self.batch_size = batch_size
        self.total_types = total_types

    def one_hot_day_type(self, day_types):
        """
        ...
        """

        count = len(day_types)
        encoded_day_types = np.zeros(shape=[count, self.batch_size, self.total_types])  # 30*96*2
        for i in range(count):
            label = day_types[i]
            encoded_day_types[i, :, label - 1] = 1
        return encoded_day_types  # 1->[1,0], 2->[0,1]

    def encode(self, day_type):
        """
        ...
        """

        encoded_day_types = np.zeros(shape=[self.total_types])
        encoded_day_types[day_type - 1] = 1
        return encoded_day_types.tolist()

    def one_hot_label(self, lab):
        """
        ...
        """

        lab_ = np.zeros(shape=[1, self.batch_size, self.total_types])
        for i in range(self.total_types):
            lab_[:, :, lab] = 1
        return lab_


def get_data(data_path, batch_size=96):
    """
    Read data
    """
    total_batch = 0
    file_names = os.listdir(data_path)
    day_parking_rate = []
    one_hot = OneHot(batch_size)  # 30, 96, 2
    for file_name in file_names:
        if os.path.isdir(file_name):
            continue
        df = pd.read_excel(data_path + "/" + file_name)
        total_batch += int(len(df) / batch_size)
        # 96 points for each day
        day_parking_rate.append(
            [list(map(lambda x: [x[0]] + one_hot.encode(int(x[1])),
                      df.values[day * batch_size:day * batch_size + batch_size]))
             for day in range(total_batch)])
    return day_parking_rate, total_batch


def write_data(g_data, num_gen_once, sample_size, cond_dim, num_run):
    """
    数据输出到excel
    """
    write_xls = xlwt.Workbook()
    # g_data_=[]
    g_data_ = np.reshape(g_data, newshape=[num_run, cond_dim, num_gen_once, sample_size])
    for p in range(cond_dim):  # 条件
        sheet = write_xls.add_sheet('label' + str(p))
        for k in range(num_run):
            for i in range(num_gen_once):
                for j in range(sample_size):
                    sheet.write(i * sample_size + j, k, str(g_data_[k][p][i][j]))
    xls_name = 'generated_data' + '_num_run' + str(num_run) + '.xls'
    write_xls.save(xls_name)
