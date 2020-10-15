import datetime
import os
import numpy as np
from data_process import get_data, write_data
from image_utils import plot_show
from utils import train

if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 使用CPU

    all_start = datetime.datetime.now()  # 程序开始运行

    # 超参数设定
    sample_size = 96  # 一次性学习的样本大小
    hidden = 100  # LSTM层的神经元个数
    batch_size = 1  # LSTM层的batch_size
    time_step = 96  # LSTM层的timestep
    learn_set = 1700  # 迭代次数列表
    cond_dim = 2  # 条件值
    latent_dim = 1  # latent space 维度
    num_run = 1  # 运行的次数
    num_gen_once = 1  # 单次生成序列
    LR = 0.001  # 学习率
    # 数据处理部分
    data = '../data'  # 读取data文件夹内的数据集，数据集是一个月内的停车数据 每十五分钟取一个点，每天96个点
    sample_set, index, num_seq = get_data(data, sample_size)
    index_list = [i for i in range(num_seq)]

    # 数据生成部分
    g_data_ = []
    for loop in range(num_run):
        begin = datetime.datetime.now()
        g_data = train(sample_set, index, sample_size, learn_set, batch_size, hidden, time_step, num_seq,
                       num_gen_once, LR, latent_dim, cond_dim)
        g_data_.append(g_data)
        end = datetime.datetime.now()
        print('用时: ', end - begin)
        end_end = datetime.datetime.now()
        print('总用时：', end_end - all_start)
    g_data_ = np.array(g_data_)
    print(g_data_.shape)

    # 数据保存及可视化
    plot_show(learn_set, g_data_, g_data, cond_dim, num_run, num_gen_once)
    write_data(g_data_, num_gen_once, sample_size, cond_dim, num_run)
