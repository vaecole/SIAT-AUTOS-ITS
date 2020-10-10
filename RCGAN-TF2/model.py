# coding=gbk
from abc import ABC

import tensorflow as tf
import numpy as np


class Generator(tf.keras.Model, ABC):
    """
    ...
    """

    def __init__(self):
        super(Generator, self).__init__()

        self.dense_z = tf.keras.layers.Dense(10, activation='relu')
        self.dropout_z = tf.keras.layers.Dropout(0.5)

        self.dense_y = tf.keras.layers.Dense(100, activation='relu')
        self.dropout_y = tf.keras.layers.Dropout(0.5)

        self.lstm = tf.keras.layers.LSTM(100, activation='relu', return_sequences=True)
        self.final_dense = tf.keras.layers.Dense(1)

    def call(self, inputs):
        """
        ...
        """
        z = self.dense_z(np.array(inputs))  # 96*10
        y = self.dense_y(z)  # 96*100
        y = tf.expand_dims(y, axis=0)  # 1*96*100
        x = self.lstm(y)  # 1*96*100
        x = tf.squeeze(x, axis=0)  # 96*100
        x = self.final_dense(x)
        return tf.expand_dims(x, axis=0)  # 1*96*1


def generator(x, y, n_hidden, batch_size, time_step):
    """
    生成器网络
    """
    # 权值和偏置初始化
    w_init = tf.compat.v1.truncated_normal_initializer(stddev=2)
    b_init = tf.compat.v1.constant_initializer(0.)
    n_hidden_g = n_hidden  # 100
    n_hidden_1 = 10
    lstm_layers = 1

    # 合并x数据，并调整格式
    inputs = tf.concat([x, y], axis=2)  # 1*96*3
    # inputs =x
    x_1 = tf.reshape(inputs, shape=[-1, inputs.get_shape()[2]])  # 96*3

    w0_ = tf.compat.v1.get_variable('w0_', [inputs.get_shape()[2], n_hidden_1], initializer=w_init)  # 3*10, normal
    b0_ = tf.compat.v1.get_variable('b0_', [n_hidden_1], initializer=b_init)  # 10, all zero
    h0_ = tf.nn.relu(tf.matmul(x_1, w0_) + b0_)  # [96*3]x[3*10] + [10] = [96*10] + [10] = [96*10]

    w0 = tf.compat.v1.get_variable('w0', [h0_.get_shape()[1], n_hidden_g], initializer=w_init)  # 10*100, normal
    b0 = tf.compat.v1.get_variable('b0', [n_hidden_g], initializer=b_init)  # 100, all zero
    h0 = tf.nn.relu(tf.matmul(h0_, w0) + b0)  # [96*10]x[10*100] + [100] = [96*100]
    h0 = tf.reshape(h0, [-1, time_step, n_hidden_g])  # [1*96*100]

    # LSTM输入格式为[batch_size,max_time,n_inputs]
    # LSTM层
    # cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden)
    cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell(
        [tf.compat.v1.nn.rnn_cell.BasicLSTMCell(n_hidden) for i in range(lstm_layers)])
    init_state = cell.zero_state(batch_size=batch_size, dtype=tf.float32)
    output_rnn, final_states = tf.compat.v1.nn.dynamic_rnn(cell, h0, initial_state=init_state, dtype=tf.float32)
    output_rnn = tf.nn.relu(tf.reshape(output_rnn, [-1, n_hidden]))  # [96,100] 将LSTM层计算的结果还原为二维张量

    # 输出层
    w1 = tf.compat.v1.get_variable('g_w1', [output_rnn.get_shape()[1], 1], initializer=w_init)  # [100,1]
    b1 = tf.compat.v1.get_variable('g_b1', [1], initializer=b_init)
    h1 = tf.matmul(output_rnn, w1) + b1  # [96*100]x[100*1] + [1] = [96*1]
    h1 = tf.reshape(h1, [-1, time_step, 1])  # 1 * 96 * 1
    return h1


class Discriminator(tf.keras.Model, ABC):
    """
    ...
    """

    def __init__(self):
        super(Discriminator, self).__init__()

        self.dense_z = tf.keras.layers.Dense(10, activation='relu')
        self.dropout_z = tf.keras.layers.Dropout(0.5)

        self.dense_y = tf.keras.layers.Dense(100, activation='relu')
        self.dropout_y = tf.keras.layers.Dropout(0.5)

        self.lstm = tf.keras.layers.LSTM(100, activation='relu', return_sequences=True)
        self.final_dense = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, inputs):
        """
        ...
        """
        z = self.dense_z(np.array(inputs))  # 96*10
        y = self.dense_y(z)  # 96*100
        y = tf.expand_dims(y, axis=0)  # 1*96*100
        x = self.lstm(y)  # 1*96*100
        x = tf.squeeze(x, axis=0)  # 96*100
        x = self.final_dense(x)
        return tf.expand_dims(x, axis=0)  # 1*96*1


def discriminator(x, y, n_hidden, batch_size, time_step):
    """
    鉴别器网络
    """
    # 权值和偏置初始化
    w_init = tf.compat.v1.keras.initializers.VarianceScaling(scale=2.0)
    b_init = tf.compat.v1.constant_initializer(0.)
    lstm_layers = 1
    # 数据合并，并调整格式
    inputs = tf.concat([x, y], axis=2)  # 1*96*3
    n_hidden_d = n_hidden  # 100
    n_hidden_1 = 10
    # inputs =x
    x_1 = tf.reshape(inputs, shape=[-1, inputs.get_shape()[2]])  # 96*3

    w0_ = tf.compat.v1.get_variable('w0_', [inputs.get_shape()[2], n_hidden_1], initializer=w_init)  # 3*10
    b0_ = tf.compat.v1.get_variable('b0_', [n_hidden_1], initializer=b_init)  # 10*n
    h0_ = tf.nn.relu(tf.matmul(x_1, w0_) + b0_)  # [96*3]x[3*10]+[10] = [96*10]

    # 输入层
    w0 = tf.compat.v1.get_variable('w0', [h0_.get_shape()[1], n_hidden_d], initializer=w_init)  # 10*100
    b0 = tf.compat.v1.get_variable('b0', [n_hidden_d], initializer=b_init)  # 100*n
    h0 = tf.nn.relu(tf.matmul(h0_, w0) + b0)  # [96*10]x[10*100] + [100] = 96*100
    h0 = tf.reshape(h0, [-1, time_step, n_hidden_d])  # 1*96*100

    # LSTM层
    # cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden)
    cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell(
        [tf.compat.v1.nn.rnn_cell.BasicLSTMCell(n_hidden) for i in range(lstm_layers)])  # 1*100
    init_state = cell.zero_state(batch_size=batch_size, dtype=tf.float32)  # 1
    output_rnn, final_states = tf.compat.v1.nn.dynamic_rnn(cell, h0, initial_state=init_state,
                                                           dtype=tf.float32)  # 96*100
    output_rnn = tf.nn.relu(tf.reshape(output_rnn, [-1, n_hidden]))  # 96*100

    # 输出层
    w1 = tf.compat.v1.get_variable('d_w1', [output_rnn.get_shape()[1], 1], initializer=w_init)  # [100,1]
    b1 = tf.compat.v1.get_variable('d_b1', [1], initializer=b_init)  # 1*n
    h1 = tf.matmul(output_rnn, w1) + b1  # 96*1
    h1 = tf.nn.sigmoid(h1)  # [96,1]
    return h1


if __name__ == "__main__":
    pass
