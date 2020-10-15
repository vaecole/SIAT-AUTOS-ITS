import numpy as np
import xlwt
import os
import pandas as pd


# 将输入标签转换为one_hot
class OneHot(object):
    def __init__(self, num, sample_size, condim):
        self.num = num
        self.sample_size = sample_size
        self.condim = condim

    def one_hot_index(self, index):
        self.index = index
        index_ = np.zeros(shape=[self.num, self.sample_size, self.condim])  # 30天数据
        # p = np.array(index).reshape(num,288,1)
        for i in range(self.num):
            label = self.index[i]
            index_[i, :, label - 1] = 1
        return index_

    def one_hot_label(self, lab):
        self.lab = lab
        lab_ = np.zeros(shape=[1, self.sample_size, self.condim])
        for i in range(self.condim):
            lab_[:, :, lab] = 1
        return lab_


# 数据读取和数据转换
def get_data(data, sample_size):
    sample_set_ = []
    index_ = []
    list_name = os.listdir(data)
    for j in list_name:
        name = j
        sample_set = []
        index = []
        df = pd.read_excel(data + "/" + name, header=None)
        df_1 = [n[0] for n in df.values.tolist()]  # values 返回值为内部多维数据组成的 多维数组（ndarray）
        df_2 = [n[1] for n in df.values.tolist()]
        df_2 = [int(i) for i in df_2]
        num = int(len(df) / sample_size)
        for i in range(num):
            sample_set.append(df_1[i * sample_size:i * sample_size + sample_size])  # 0到288 每天的288个数 30*288
            index.append(df_2[i * sample_size])
        sample_set_.append(sample_set)
        index_.append(index)
    index_ = np.array(index_)
    return sample_set_, index_, num


# 数据输出到excel
def write_data(g_data, num_gen_once, sample_size, cond_dim, num_run):
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
