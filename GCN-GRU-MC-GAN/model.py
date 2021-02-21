from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dropout, Dense, Input, Dot, LeakyReLU, Flatten
from spektral.layers import GCNConv
from tensorflow.python.keras.layers import GRU, LSTM, CuDNNLSTM, CuDNNGRU

l2_reg = 5e-4 / 2  # L2 regularization rate
base = 16
dropout = 0.5


def make_generator(name, s, adj, node_f, use_gcn=True, use_gru=True):
    n = len(adj)
    input_s = Input(shape=(s, n))
    input_f = Input(shape=(n, node_f.shape[1]))
    input_g = Input(shape=(n, n))
    if use_gcn:
        gcov1 = GCNConv(2 * base,
                        activation='elu', kernel_regularizer=l2(l2_reg), use_bias=False)([input_f, input_g])
        gcov2 = GCNConv(2 * base,
                        activation='elu', kernel_regularizer=l2(l2_reg), use_bias=False)([gcov1, input_g])
        # N*Cov2
        input_s1 = Dot(axes=(2, 1))([input_s, gcov2])
    else:
        input_s1 = Dropout(dropout)(LeakyReLU()(Dense(4 * base, input_shape=(n,))(input_s)))

    fc1 = Dropout(dropout)(LeakyReLU()(Dense(4 * base, input_shape=(n,))(input_s1)))
    fc2 = Dropout(dropout)(LeakyReLU()(Dense(8 * base, input_shape=(n,))(fc1)))
    # S*D2

    if use_gru:
        rnn1 = CuDNNGRU(2 * base, return_sequences=True)(fc2)
    else:
        rnn1 = CuDNNLSTM(2 * base, return_sequences=True)(fc2)
    out = Dense(1)(rnn1)
    return Model(name=name, inputs=[input_s, input_f, input_g], outputs=out)


def make_discriminator(name, s, adj, node_f, use_gcn=True, use_gru=True):
    n = len(adj)
    input_s = Input(shape=(s, n))
    input_f = Input(shape=(n, node_f.shape[1]))
    input_g = Input(shape=(n, n))
    if use_gcn:
        gcov1 = GCNConv(2 * base,
                        activation='elu', kernel_regularizer=l2(l2_reg), use_bias=False)([input_f, input_g])
        gcov2 = GCNConv(2 * base,
                        activation='elu', kernel_regularizer=l2(l2_reg), use_bias=False)([gcov1, input_g])
        # N*Cov2
        input_s1 = Dot(axes=(2, 1))([input_s, gcov2])
    else:
        input_s1 = Dropout(dropout)(LeakyReLU()(Dense(4 * base, input_shape=(n,))(input_s)))

    fc1 = Dropout(dropout)(LeakyReLU()(Dense(4 * base, input_shape=(n,))(input_s1)))
    fc2 = Dropout(dropout)(LeakyReLU()(Dense(8 * base, input_shape=(n,))(fc1)))
    # S*D2

    if use_gru:
        rnn1 = CuDNNGRU(2 * base, return_sequences=True)(fc2)
    else:
        rnn1 = CuDNNLSTM(2 * base, return_sequences=True)(fc2)
    out = Dense(1)(Flatten()(rnn1))
    return Model(name=name, inputs=[input_s, input_f, input_g], outputs=out)
