
# coding=gbk
import numpy as np
import xlwt
import os
import pandas as pd


class OneHot(object):
    """
    �������ǩת��Ϊone_hot
    """
    def __init__(self, num, sample_size, condim):
        self.num = num
        self.sample_size = sample_size
        self.condim = condim

    def one_hot_index(self, index):
        self.index = index
        index_ = np.zeros(shape=[self.num, self.sample_size, self.condim])  # 30������
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


def get_data(data, sample_size):
    """
    ���ݶ�ȡ������ת��
    """
    sample_set_ = []
    index_ = []
    list_name = os.listdir(data)
    for j in list_name:
        name = j
        sample_set = []
        index = []
        df = pd.read_excel(data + "/" + name, header=None)
        df_1 = [n[0] for n in df.values.tolist()]  # values ����ֵΪ�ڲ���ά������ɵ� ��ά���飨NDArray��
        df_2 = [n[1] for n in df.values.tolist()]
        df_2 = [int(i) for i in df_2]
        num = int(len(df) / sample_size)
        for i in range(num):
            sample_set.append(df_1[i * sample_size:i * sample_size + sample_size])  # 0��288 ÿ���288���� 30*288
            index.append(df_2[i * sample_size])
        sample_set_.append(sample_set)
        index_.append(index)
    index_ = np.array(index_)
    return sample_set_, index_, num


def write_data(g_data, num_gen_once, sample_size, cond_dim, num_run):
    """
    ���������excel
    """
    write_xls = xlwt.Workbook()
    # g_data_=[]
    g_data_ = np.reshape(g_data, newshape=[num_run, cond_dim, num_gen_once, sample_size])
    for p in range(cond_dim):  # ����
        sheet = write_xls.add_sheet('label' + str(p))
        for k in range(num_run):
            for i in range(num_gen_once):
                for j in range(sample_size):
                    sheet.write(i * sample_size + j, k, str(g_data_[k][p][i][j]))
    xls_name = 'generated_data' + '_num_run' + str(num_run) + '.xls'
    write_xls.save(xls_name)
