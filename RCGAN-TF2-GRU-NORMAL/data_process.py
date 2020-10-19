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
    Read monthly and daily(96*15min=24H) 96 units parking rate which in (0,1) per 15 minutes and encoded day type(weekend=[1,0], workday=[0,1])
    """
    file_names = os.listdir(data_path)
    monthly_parking_rate = []
    one_hot = OneHot(batch_size)  # 30, 96, 2
    for file_name in file_names:
        if os.path.isdir(file_name):
            continue
        dataframe = pd.read_excel(data_path + "/" + file_name)
        current_batch = int(len(dataframe) / batch_size)
        # 1 monthly days, 96 points for each day
        for day in range(current_batch):
            monthly_parking_rate.append(
                [list(map(lambda x: [x[0]],
                          dataframe.values[day * batch_size:day * batch_size + batch_size])),
                 list(map(lambda x: one_hot.encode(int(x[1])),
                          dataframe.values[day * batch_size:day * batch_size + batch_size]))])

    return monthly_parking_rate


def write_data(save_path, g_data, sample_size, cond_dim, epoch):
    """
    数据输出到excel
    """
    write_xls = xlwt.Workbook()
    for p in range(cond_dim):  # 条件
        sheet = write_xls.add_sheet('label' + str(p))
        for j in range(sample_size):
            sheet.write(j, 0, str(g_data[p][j]))
    xls_name = save_path + '/' + str(epoch) + '.xls'
    write_xls.save(xls_name)

# todo: residual analysis
