import tensorflow as tf


def get_wasserstein_losses_fn():
    def d_loss_fn(real_output, fake_output):
        real_loss = - tf.reduce_mean(real_output)
        fake_loss = tf.reduce_mean(fake_output)
        return fake_loss + real_loss

    def g_loss_fn(fake_output):
        fake_loss = - tf.reduce_mean(fake_output)
        return fake_loss

    return d_loss_fn, g_loss_fn


def get_cross_entropy_loss_fn():
    def discriminator_loss(real_output, fake_output):
        real_loss = tf.losses.binary_crossentropy(tf.ones_like(real_output), real_output, from_logits=True)
        fake_loss = tf.losses.binary_crossentropy(tf.zeros_like(fake_output), fake_output, from_logits=True)
        total_loss = real_loss + fake_loss
        return total_loss

    def generator_loss(fake_output):
        return tf.losses.binary_crossentropy(tf.ones_like(fake_output), fake_output, from_logits=True)

    return discriminator_loss, generator_loss
