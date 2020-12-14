import tensorflow as tf
from abc import ABC
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dropout, GRU, Dense, Reshape, Input, Dot, Concatenate
import tensorflow.keras.layers as layers
from spektral.layers import GraphConv

l2_reg = 5e-4 / 2  # L2 regularization rate
base = 16


def make_model(name, use_gcn, dropout, S, adj, node_f):
    N = len(adj)
    input_s = Input(shape=(S, N))
    input_f = Input(shape=(N, node_f.shape[1]))
    input_g = Input(shape=(N, N))
    if use_gcn:
        gcov1 = GraphConv(2 * base,
                          activation='elu', kernel_regularizer=l2(l2_reg), use_bias=False)([input_f, input_g])
        gcov2 = GraphConv(base,
                          activation='elu', kernel_regularizer=l2(l2_reg), use_bias=False)([gcov1, input_g])
        # N*Cov2
        input_s1 = Dot(axes=(2, 1))([input_s, gcov2])
    else:
        input_s1 = Dropout(dropout)(Dense(4 * base, activation='relu', input_shape=(N,))(input_s))

    fc1 = Dropout(dropout)(Dense(4 * base, activation='relu', input_shape=(N,))(input_s1))
    fc2 = Dropout(dropout)(Dense(8 * base, activation='relu', input_shape=(N,))(fc1))
    # S*D2

    gru = GRU(2 * base, return_sequences=True)(fc2)
    out = Dense(1, activation='tanh')(gru)
    return Model(name=name, inputs=[input_s, input_f, input_g], outputs=out)


class Generator(Model, ABC):

    def __init__(self, adj, nodes_features, use_gcn, dropout):
        super(Generator, self).__init__()
        self.adj = adj
        self.nodes_features = nodes_features

        self.dropout = Dropout(dropout)
        if use_gcn:
            self.graph_conv_1 = GraphConv(2 * base,
                                          activation='elu',
                                          kernel_regularizer=l2(l2_reg),
                                          use_bias=False)
            self.graph_conv_2 = GraphConv(base,
                                          activation='elu',
                                          kernel_regularizer=l2(l2_reg),
                                          use_bias=False)
        else:
            self.dense_0 = Dense(4 * base, activation='relu')

        self.dense_1 = Dense(4 * base, activation='relu')
        self.dense_2 = Dense(8 * base, activation='relu')
        self.gru = GRU(2 * base, return_sequences=True)
        self.final_dense = Dense(1, activation='tanh')

    def call(self, seq, use_gcn, dropout, training=True):
        s = tf.convert_to_tensor(seq)  # S*N

        if use_gcn:
            f = tf.convert_to_tensor(self.nodes_features)  # N*F
            g = tf.convert_to_tensor(self.adj)  # N*N
            c = self.graph_conv_1([f, g])  # N*Cov1
            c = self.graph_conv_2([c, g])  # N*Cov2
            s = tf.matmul(s, c)  # S*N x N*Cov2
        else:
            s = self.dense_0(s)  # S*D1
            s = self.dropout(s, training=training)

        fc = self.dense_1(s)  # S*D1
        fc = self.dropout(fc, training=training)
        fc = self.dense_2(fc)  # S*D2
        fc = self.dropout(fc, training=training)

        fc = tf.expand_dims(fc, axis=0)
        ro = self.gru(fc)
        ro = tf.squeeze(ro, axis=0)  # S*R
        return self.final_dense(ro)  # S*1


class Discriminator(Model, ABC):

    def __init__(self, adj, nodes_features, use_gcn, dropout):
        super(Discriminator, self).__init__()
        self.adj = adj
        self.nodes_features = nodes_features

        self.dropout = Dropout(dropout)
        if use_gcn:
            self.graph_conv_1 = GraphConv(2 * base,
                                          activation='elu',
                                          kernel_regularizer=l2(l2_reg),
                                          use_bias=False)
            self.graph_conv_2 = GraphConv(base,
                                          activation='elu',
                                          kernel_regularizer=l2(l2_reg),
                                          use_bias=False)
        else:
            self.dense_0 = Dense(4 * base, activation='relu')

        self.dense_1 = Dense(4 * base, activation='relu')
        self.dense_2 = Dense(8 * base, activation='relu')
        self.gru = GRU(2 * base, return_sequences=True)
        self.final_dense = Dense(1, activation='tanh')

    def call(self, seq, use_gcn, dropout, training=True):
        s = tf.convert_to_tensor(seq)  # S*N

        if use_gcn:
            f = tf.convert_to_tensor(self.nodes_features)  # N*F
            g = tf.convert_to_tensor(self.adj)  # N*N
            c = self.graph_conv_1([f, g])  # N*Cov1
            c = self.graph_conv_2([c, g])  # N*Cov2
            s = tf.matmul(s, c)  # S*N x N*Cov2
        else:
            s = self.dense_0(s)  # S*D1
            s = self.dropout(s, training=training)

        fc = self.dense_1(s)  # S*D1
        fc = self.dropout(fc, training=training)
        fc = self.dense_2(fc)  # S*D2
        fc = self.dropout(fc, training=training)

        fc = tf.expand_dims(fc, axis=0)
        ro = self.gru(fc)
        ro = tf.squeeze(ro, axis=0)  # S*R
        return self.final_dense(ro)  # S*1

    @staticmethod
    def make_model(use_gcn, dropout, N):
        model = tf.keras.Sequential()
        if use_gcn:
            model.add(GraphConv(2 * base, activation='elu', kernel_regularizer=l2(l2_reg), use_bias=False),
                      input_shape=(N,))  # N*Cov1
            model.add(GraphConv(base, activation='elu', kernel_regularizer=l2(l2_reg), use_bias=False))
            model.add(layers.Dot(input_shape=(N,)))
        else:
            model.add(Dense(4 * base, activation='relu', input_shape=(N,)))
            model.add(Dropout(dropout))
        model.add(Dense(4 * base, activation='relu'))
        model.add(Dropout(dropout))
        model.add(Dense(8 * base, activation='relu'))
        model.add(Dropout(dropout))
        model.add(Reshape((1, N, -1)))
        model.add(GRU(2 * base, return_sequences=True))
        model.add(Reshape((N, -1)))
        model.add(Dense(1, activation='tanh'))
        return model
