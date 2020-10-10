# coding=gbk
import tensorflow as tf
import numpy as np

from data_process import get_data
from model import generator, discriminator
from utils import optimizer

import os
import time

import tensorflow as tf  # TF 2.0

from model import Generator, Discriminator

# from utils import generator_loss, discriminator_loss, save_imgs, preprocess_image


RANGE = 2


def train(data_path):
    # 1*30*96(1month*30day*96[15min]), 1*30(1: weekend, 2: workday)
    monthly_parking_rate, days = get_data(data_path)

    if not os.path.exists('./images'):
        os.makedirs('./images')

    latent_dim = 100
    epochs = 800
    batch_size = 96
    save_interval = 50

    img_shape = (28, 28, 1)
    # 1: weekend, 2: workday
    cond_dim = 2

    generator = Generator()
    discriminator = Discriminator()

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        0.001,
        decay_steps=400,
        decay_rate=0.95,
        staircase=True)
    gen_optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule)
    disc_optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule)

    # gen_optimizer = tf.keras.optimizers.Adam(0.0002, 0.5)
    # disc_optimizer = tf.keras.optimizers.Adam(0.0002, 0.5)

    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    @tf.function
    def train_step(inputs):
        """
        ...
        """
        # noise = tf.random.normal([batch_size, latent_dim])
        from NoiseDistribution import NoiseDistribution
        noise = NoiseDistribution(2)
        with tf.GradientTape(persistent=True) as tape:
            # todo: combine noise sample with conditions? really necessary
            generated_images = generator(noise.sample(batch_size))
            real_output = discriminator(inputs)
            generated_output = discriminator(generated_images)

            gen_loss = tf.reduce_mean(input_tensor=-tf.math.log(generated_output))
            disc_loss = tf.reduce_mean(input_tensor=-tf.math.log(real_output) - tf.math.log(1 - generated_output))

            # generator_loss(cross_entropy, generated_output)
            # disc_loss = discriminator_loss(cross_entropy, real_output, generated_output)

        grad_gen = tape.gradient(gen_loss, generator.trainable_variables)
        grad_disc = tape.gradient(disc_loss, discriminator.trainable_variables)

        gen_optimizer.apply_gradients(zip(grad_gen, generator.trainable_variables))
        disc_optimizer.apply_gradients(zip(grad_disc, discriminator.trainable_variables))

        return gen_loss, disc_loss

    seed = tf.random.normal([16, latent_dim])

    # condition_is_workday = list(map(lambda x: int(x[0][1]), monthly_parking_rate))
    for epoch in range(1, epochs + 1):
        start = time.time()
        total_gen_loss = 0
        total_disc_loss = 0
        for daily_parking_rate in monthly_parking_rate[0]:
            gen_loss, disc_loss = train_step(daily_parking_rate)
            total_gen_loss += gen_loss
            total_disc_loss += disc_loss

        print('Time for epoch {} is {} sec - gen_loss = {}, disc_loss = {}'.format(epoch, time.time() - start,
                                                                                   total_gen_loss / batch_size,
                                                                                   total_disc_loss / batch_size))
        if epoch % save_interval == 0:
            save_imgs(epoch, generator, seed)


if __name__ == "__main__":
    train()


def train(sample_set, index, sample_size, TRAIN_ITERS, batch_size, n_hidden, timestep, num_seq,
          num_gen_once, LR, latent_dim, cond_dim):
    """
    训练函数
    """
    # 构建图
    x = tf.compat.v1.placeholder(tf.float32, shape=(batch_size, sample_size, 1))  # 1*96*1, real input
    y = tf.compat.v1.placeholder(tf.float32, shape=(batch_size, sample_size, cond_dim))  # 1*96*2,
    z = tf.compat.v1.placeholder(tf.float32, shape=(batch_size, sample_size, latent_dim))  # 1*96*1
    # keep_prob = tf.placeholder(tf.float32)
    # networks : generator
    with tf.compat.v1.variable_scope('Gen'):  # 定义变量域
        G_z = generator(z, y, n_hidden, batch_size, timestep)

    # networks : discriminator
    with tf.compat.v1.variable_scope('Disc') as scope:  # 定义变量域
        D_real = discriminator(x, y, n_hidden, batch_size, timestep)
        scope.reuse_variables()  # 复用变量，避免重复生成两个变量，目的是产生一组变量，这样训练才有意义
        D_fake = discriminator(G_z, y, n_hidden, batch_size, timestep)

    loss_d = tf.reduce_mean(input_tensor=-tf.math.log(D_real) - tf.math.log(1 - D_fake))
    loss_g = tf.reduce_mean(input_tensor=-tf.math.log(D_fake))

    # trainable variables for each network      #收集两个变量域的变量，方便后面提交到优化器当中进行训练
    d_params = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope='Disc')
    g_params = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope='Gen')

    # 优化器设定
    from utils import optimizer
    opt_d = optimizer(loss_d, d_params, 400, LR)  #
    opt_g = optimizer(loss_g, g_params, 400, LR)  #

    """ 开始训练 """
    # training-loop
    # sess = tf.InteractiveSession()
    # tf.global_variables_initializer().run()
    from NoiseDistribution import NoiseDistribution
    p_z = NoiseDistribution(RANGE)  # 2
    # 转换为one_hot形式

    lab_to_one = OneHot(num_seq, sample_size, cond_dim)  # 30, 96, 2

    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        for step in range(TRAIN_ITERS):  # 训练次数 1000
            for i in range(len(sample_set)):  # 每个训练集有30天的数据 1
                for j in range(num_seq):  # 30
                    index_ = lab_to_one.one_hot_index(index[i])
                    x_ = np.reshape(sample_set[i][j], newshape=[batch_size, sample_size, 1])  # 1*96*1
                    y_ = np.reshape(index_[j], newshape=[batch_size, sample_size, cond_dim])  # 1*96*2
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

    tf.compat.v1.reset_default_graph()  # 第二次执行的时候避免新建已经有的值
    return g_seq_label


def optimizer(loss, var_list, num_decay_steps=400, initial_learning_rate=0.03):
    """
    优化器
    """
    decay = 0.95
    batch = tf.Variable(0)

    # 指数型衰减学习率
    learning_rate = tf.compat.v1.train.exponential_decay(
        initial_learning_rate,
        batch,
        num_decay_steps,
        decay,
        staircase=True)
    # 优化器，梯度下降
    optimizer_ = tf.compat.v1.train.GradientDescentOptimizer(learning_rate).minimize(
        loss,
        global_step=batch,
        var_list=var_list
    )
    return optimizer_
