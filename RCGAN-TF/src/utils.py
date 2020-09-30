import gc
import tensorflow as tf
import numpy as np

from network import generator, discriminator

force_gc = True
RANGE = 2


# 垃圾回收
def collect_gc():
    if force_gc:
        gc.collect()


# 优化器
def optimizer(loss, var_list, num_decay_steps=400, initial_learning_rate=0.03):
    decay = 0.95
    batch = tf.Variable(0)

    # 指数型衰减学习率
    learning_rate = tf.train.exponential_decay(
        initial_learning_rate,
        batch,
        num_decay_steps,
        decay,
        staircase=True)
    # 优化器，梯度下降
    optimizer_ = tf.train.GradientDescentOptimizer(learning_rate).minimize(
        loss,
        global_step=batch,
        var_list=var_list
    )
    return optimizer_


# 训练函数
def train(sample_set, index, sample_size, TRAIN_ITERS, batch_size, n_hidden, timestep, num_seq,
          num_gen_once, LR, latent_dim, cond_dim):
    """ 构建图 """
    x = tf.placeholder(tf.float32, shape=(batch_size, sample_size, 1))
    y = tf.placeholder(tf.float32, shape=(batch_size, sample_size, cond_dim))
    z = tf.placeholder(tf.float32, shape=(batch_size, sample_size, latent_dim))
    # keep_prob = tf.placeholder(tf.float32)
    # networks : generator
    with tf.variable_scope('Gen'):  # 定义变量域
        G_z = generator(z, y, n_hidden, batch_size, timestep)

    # networks : discriminator
    with tf.variable_scope('Disc') as scope:  # 定义变量域
        D_real = discriminator(x, y, n_hidden, batch_size, timestep)
        scope.reuse_variables()  # 复用变量，避免重复生成两个变量，目的是产生一组变量，这样训练才有意义
        D_fake = discriminator(G_z, y, n_hidden, batch_size, timestep)

    loss_d = tf.reduce_mean(-tf.log(D_real) - tf.log(1 - D_fake))
    loss_g = tf.reduce_mean(-tf.log(D_fake))

    # trainable variables for each network      #收集两个变量域的变量，方便后面提交到优化器当中进行训练
    d_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Disc')
    g_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Gen')

    # 优化器设定
    from src.utils import optimizer
    opt_d = optimizer(loss_d, d_params, 400, LR)  #
    opt_g = optimizer(loss_g, g_params, 400, LR)  #

    """ 开始训练 """
    # training-loop
    # sess = tf.InteractiveSession()
    # tf.global_variables_initializer().run()
    from src.NoiseDistribution import NoiseDistribution
    p_z = NoiseDistribution(RANGE)
    # 转换为one_hot形式
    from src.data_process import OneHot
    lab_to_one = OneHot(num_seq, sample_size, cond_dim)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for step in range(TRAIN_ITERS):  # 训练次数
            for i in range(len(sample_set)):  # 每个训练集有30天的数据
                for j in range(num_seq):
                    index_ = lab_to_one.one_hot_index(index[i])
                    x_ = np.reshape(sample_set[i][j], newshape=[batch_size, sample_size, 1])  #
                    y_ = np.reshape(index_[j], newshape=[batch_size, sample_size, cond_dim])  #
                    z_ = p_z.sample(sample_size)
                    loss_d_, _ = sess.run([loss_d, opt_d], {x: np.reshape(x_, (batch_size, sample_size, 1)),
                                                            z: np.reshape(z_, (batch_size, sample_size, latent_dim)),
                                                            y: np.reshape(y_, (batch_size, sample_size, cond_dim))})

                    z_ = p_z.sample(sample_size)
                    loss_g_, _ = sess.run([loss_g, opt_g], {z: np.reshape(z_, (batch_size, sample_size, latent_dim)),
                                                            y: np.reshape(y_, (batch_size, sample_size, cond_dim))})

            if step % 100 == 0:
                print('[%d/%d]: loss_d : %.3f, loss_g : %.3f' % (step, TRAIN_ITERS, loss_d_, loss_g_))

        # generate

        g_seq_label = []
        for i in range(cond_dim):
            label_onehot = lab_to_one.one_hot_label(i)[0]
            print(label_onehot[0][0])
            g_seq = []
            for j in range(num_gen_once):
                zs = p_z.sample(sample_size)
                g_data = sess.run(G_z, {z: np.reshape(zs, (batch_size, sample_size, latent_dim)),
                                        y: np.reshape(label_onehot, (batch_size, sample_size, cond_dim))})
                g_data = np.reshape(g_data, [sample_size])
                g_seq.append(g_data)
            print(np.array(g_seq).shape)
            # g_seq_=np.array(g_seq)
            g_seq_label.append(g_seq)
        g_seq_label = np.array(g_seq_label)

    from src.utils import collect_gc
    collect_gc()
    tf.reset_default_graph()  # 第二次执行的时候避免新建已经有的值
    return g_seq_label
