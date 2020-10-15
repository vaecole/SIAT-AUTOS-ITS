# coding=gbk
from abc import ABC

import tensorflow as tf


class Generator(tf.keras.Model, ABC):
    """
    ...
    """

    def __init__(self):
        super(Generator, self).__init__()

        self.dense_z = tf.keras.layers.Dense(512, activation='relu')
        self.dropout_z = tf.keras.layers.Dropout(0.5)

        self.dense_y = tf.keras.layers.Dense(512, activation='relu')
        self.dropout_y = tf.keras.layers.Dropout(0.5)

        self.combined_dense = tf.keras.layers.Dense(1024, activation='relu')
        self.dropout_x = tf.keras.layers.Dropout(0.5)

        self.rnn = tf.keras.layers.GRU(256, return_sequences=True)

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

        x = tf.squeeze(self.rnn(tf.expand_dims(combined_x, axis=0)), axis=0)
        return self.final_dense(x)


class Discriminator(tf.keras.Model, ABC):
    """
    ...
    """

    def __init__(self):
        super(Discriminator, self).__init__()

        self.dense_z = tf.keras.layers.Dense(512, activation='relu')
        self.dropout_z = tf.keras.layers.Dropout(0.5)

        self.dense_y = tf.keras.layers.Dense(512, activation='relu')
        self.dropout_y = tf.keras.layers.Dropout(0.5)

        self.combined_dense = tf.keras.layers.Dense(1024, activation='relu')
        self.dropout_x = tf.keras.layers.Dropout(0.5)

        self.rnn = tf.keras.layers.GRU(256, return_sequences=True)

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
