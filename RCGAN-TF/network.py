import tensorflow as tf


# 生成器网络
def generator(x, y, n_hidden, batch_size, time_step):
    # 权值和偏置初始化
    w_init = tf.truncated_normal_initializer(stddev=2)
    b_init = tf.constant_initializer(0.)
    n_hidden_g = n_hidden
    n_hidden_1 = 10
    lstm_layers = 1

    # 合并x数据，并调整格式
    inputs = tf.concat([x, y], axis=2)
    # inputs =x
    x_1 = tf.reshape(inputs, shape=[-1, inputs.get_shape()[2]])

    w0_ = tf.get_variable('w0_', [inputs.get_shape()[2], n_hidden_1], initializer=w_init)
    b0_ = tf.get_variable('b0_', [n_hidden_1], initializer=b_init)
    h0_ = tf.nn.relu(tf.matmul(x_1, w0_) + b0_)

    w0 = tf.get_variable('w0', [h0_.get_shape()[1], n_hidden_g], initializer=w_init)
    b0 = tf.get_variable('b0', [n_hidden_g], initializer=b_init)
    h0 = tf.nn.relu(tf.matmul(h0_, w0) + b0)
    h0 = tf.reshape(h0, [-1, time_step, n_hidden_g])

    # LSTM输入格式为[batch_size,max_time,n_inputs]
    # LSTM层
    # cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden)
    cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.BasicLSTMCell(n_hidden) for i in range(lstm_layers)])
    init_state = cell.zero_state(batch_size=batch_size, dtype=tf.float32)
    output_rnn, final_states = tf.nn.dynamic_rnn(cell, h0, initial_state=init_state, dtype=tf.float32)
    output_rnn = tf.nn.relu(tf.reshape(output_rnn, [-1, n_hidden]))  # [288,100] 将LSTM层计算的结果还原为二维张量

    # 输出层
    w1 = tf.get_variable('g_w1', [output_rnn.get_shape()[1], 1], initializer=w_init)  # [288,1]
    b1 = tf.get_variable('g_b1', [1], initializer=b_init)
    h1 = tf.matmul(output_rnn, w1) + b1
    h1 = tf.reshape(h1, [-1, time_step, 1])
    return h1


# 鉴别器网络
def discriminator(x, y, n_hidden, batch_size, time_step):
    # 权值和偏置初始化
    w_init = tf.contrib.layers.variance_scaling_initializer()
    b_init = tf.constant_initializer(0.)
    lstm_layers = 1
    # 数据合并，并调整格式
    inputs = tf.concat([x, y], axis=2)
    n_hidden_d = n_hidden
    n_hidden_1 = 10
    # inputs =x
    x_1 = tf.reshape(inputs, shape=[-1, inputs.get_shape()[2]])

    w0_ = tf.get_variable('w0_', [inputs.get_shape()[2], n_hidden_1], initializer=w_init)
    b0_ = tf.get_variable('b0_', [n_hidden_1], initializer=b_init)
    h0_ = tf.nn.relu(tf.matmul(x_1, w0_) + b0_)

    # 输入层
    w0 = tf.get_variable('w0', [h0_.get_shape()[1], n_hidden_d], initializer=w_init)
    b0 = tf.get_variable('b0', [n_hidden_d], initializer=b_init)
    h0 = tf.nn.relu(tf.matmul(h0_, w0) + b0)

    h0 = tf.reshape(h0, [-1, time_step, n_hidden_d])

    # LSTM层
    # cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden)
    cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.BasicLSTMCell(n_hidden) for i in range(lstm_layers)])
    init_state = cell.zero_state(batch_size=batch_size, dtype=tf.float32)
    output_rnn, final_states = tf.nn.dynamic_rnn(cell, h0, initial_state=init_state, dtype=tf.float32)
    output_rnn = tf.nn.relu(tf.reshape(output_rnn, [-1, n_hidden]))

    # 输出层
    w1 = tf.get_variable('d_w1', [output_rnn.get_shape()[1], 1], initializer=w_init)  # [288,1]
    b1 = tf.get_variable('d_b1', [1], initializer=b_init)
    h1 = tf.matmul(output_rnn, w1) + b1
    h1 = tf.nn.sigmoid(h1)  # [288,1]
    return h1
