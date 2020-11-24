# coding=gbk
from abc import ABC

import tensorflow as tf
from spektral.layers import GraphConv
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dropout

class Generator(tf.keras.Model, ABC):
    """
    ...
    """

    def __init__(self, graph_features=11):
        super(Generator, self).__init__()
        self.gcn = GraphConv(shape=(graph_features, ))
        self.dense_z = tf.keras.layers.Dense(256, activation='relu')
        self.dropout_z = tf.keras.layers.Dropout(0.5)

        self.dense_y = tf.keras.layers.Dense(256, activation='relu')
        self.dropout_y = tf.keras.layers.Dropout(0.5)

        self.combined_dense = tf.keras.layers.Dense(512, activation='relu')
        self.dropout_x = tf.keras.layers.Dropout(0.5)

        self.rnn = tf.keras.layers.GRU(128, return_sequences=True)

        self.final_dense = tf.keras.layers.Dense(1, activation='relu')

    def call(self, inputs, condition, training=True):
        """
        ...
        """
        z = self.dense_z(tf.convert_to_tensor(inputs)) # 96*256
        z = self.dropout_z(z, training)

        y = self.dense_y(tf.convert_to_tensor(condition)) # 2*256
        y = self.dropout_y(y, training)

        combined_x = self.combined_dense(tf.concat([z, y], axis=-1)) # 98*256
        combined_x = self.dropout_x(combined_x, training)

        x = tf.squeeze(self.rnn(tf.expand_dims(combined_x, axis=0)), axis=0) # 98*256
        return self.final_dense(x)


class Discriminator(tf.keras.Model, ABC):
    """
    ...
    """

    def __init__(self):
        super(Discriminator, self).__init__()

        self.dense_z = tf.keras.layers.Dense(256, activation='relu')
        self.dropout_z = tf.keras.layers.Dropout(0.5)

        self.dense_y = tf.keras.layers.Dense(256, activation='relu')
        self.dropout_y = tf.keras.layers.Dropout(0.5)

        self.combined_dense = tf.keras.layers.Dense(512, activation='relu')
        self.dropout_x = tf.keras.layers.Dropout(0.5)

        self.rnn = tf.keras.layers.GRU(128, return_sequences=True)

        self.final_dense = tf.keras.layers.Dense(1, activation='tanh')

    def call(self, inputs, condition, training=True):
        """
        ...
        """
        z = self.dense_z(tf.convert_to_tensor(inputs))
        z = self.dropout_z(z, training)

        y = self.dense_y(tf.convert_to_tensor(condition))
        y = self.dropout_y(y, training)

        combined_x = self.combined_dense(tf.concat([z, y], axis=-1))
        combined_x = self.dropout_x(combined_x, training)

        x = tf.squeeze(self.rnn(tf.expand_dims(combined_x, axis=0)), axis=0)  # 96*100
        return self.final_dense(x)


if __name__ == "__main__":
    pass
