
# coding=gbk
import datetime
import os
import numpy as np
from data_process import get_data, write_data
from image_utils import plot_show
from utils import train

if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # ʹ��CPU

    all_start = datetime.datetime.now()  # ����ʼ����

    # �������趨
    sample_size = 96  # һ����ѧϰ��������С
    hidden = 100  # LSTM�����Ԫ����
    batch_size = 1  # LSTM���batch_size
    time_step = 96  # LSTM���time_step
    learn_set = 1000  # ���������б�
    cond_dim = 2  # ����ֵ
    latent_dim = 1  # latent space ά��
    num_run = 1  # ���еĴ���
    num_gen_once = 1  # ������������
    LR = 0.001  # ѧϰ��
    # ���ݴ�����
    data = './data'  # ��ȡdata�ļ����ڵ����ݼ������ݼ���һ�����ڵ�ͣ������ ÿʮ�����ȡһ���㣬ÿ��96����
    sample_set, index, num_seq = get_data(data, sample_size)
    index_list = [i for i in range(num_seq)]

    # �������ɲ���
    g_data_ = []
    for loop in range(num_run):
        begin = datetime.datetime.now()
        g_data = train(sample_set, index, sample_size, learn_set, batch_size, hidden, time_step, num_seq,
                       num_gen_once, LR, latent_dim, cond_dim)
        g_data_.append(g_data)
        end = datetime.datetime.now()
        print('��ʱ: ', end - begin)
        end_end = datetime.datetime.now()
        print('����ʱ��', end_end - all_start)
    g_data_ = np.array(g_data_)
    print(g_data_.shape)

    # ���ݱ��漰���ӻ�
    plot_show(learn_set, g_data_, index, cond_dim, num_run, num_gen_once)
    write_data(g_data_, num_gen_once, sample_size, cond_dim, num_run)
