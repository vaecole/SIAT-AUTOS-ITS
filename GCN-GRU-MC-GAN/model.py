import tensorflow as tf
import utils
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dropout, Dense, Input, Dot, LeakyReLU, Flatten, InputSpec
from spektral.layers import TAGConv, GraphConv, DiffPool
from tensorflow.python.keras.layers import GRU, LSTM, CuDNNLSTM, CuDNNGRU, Attention
import tensorflow.keras.backend as K

l2_reg = 5e-4 / 2  # L2 regularization rate
base = 16
dropout = 0.3


def make_generator(name, s, adj, node_f, use_gcn=True, use_gru=True):
    n = node_f.shape[0]  # number of nodes
    input_s = Input(shape=(s, n))
    input_f = Input(shape=(n, node_f.shape[1]))
    input_g = Input(shape=(n, n))
    if use_gcn:
        gcov1 = GraphConv(2 * base)([input_f, input_g])
        gcov2 = GraphConv(2 * base)([gcov1, input_g])
        input_s1 = Dot(axes=(2, 1))([input_s, gcov2])  # dot product: element by element multiply
    else:
        input_s1 = Dense(4 * base, activation='relu', input_shape=(n,))(input_s)

    fc2 = Dense(8 * base, activation='relu', input_shape=(n,))(input_s1)
    # S*D2

    if use_gru:
        rnn1 = Dropout(dropout)(CuDNNGRU(2 * base, return_sequences=True)(fc2))
    else:
        rnn1 = Dense(16 * base, activation='relu', input_shape=(n,))(fc2)
    out = Dense(1)(rnn1)
    return Model(name=name, inputs=[input_s, input_f, input_g], outputs=out)


def make_discriminator(name, s, adj, node_f, use_gcn=True, use_gru=True):
    n = node_f.shape[0]  # number of nodes
    input_s = Input(shape=(s, n))
    input_f = Input(shape=(n, node_f.shape[1]))
    input_g = Input(shape=(n, n))
    if use_gcn:
        gcov1 = GraphConv(2 * base)([input_f, input_g])
        gcov2 = GraphConv(2 * base)([gcov1, input_g])
        input_s1 = Dot(axes=(2, 1))([input_s, gcov2])  # dot product: element by element multiply
    else:
        input_s1 = Dense(4 * base, activation='relu', input_shape=(n,))(input_s)

    fc2 = Dense(8 * base, activation='relu', input_shape=(n,))(input_s1)
    # S*D2

    if use_gru:
        rnn1 = Dropout(dropout)(CuDNNGRU(2 * base, return_sequences=True)(fc2))
    else:
        rnn1 = Dense(16 * base, activation='relu', input_shape=(n,))(fc2)
    out = Dense(1)(Flatten()(rnn1))
    return Model(name=name, inputs=[input_s, input_f, input_g], outputs=out)
