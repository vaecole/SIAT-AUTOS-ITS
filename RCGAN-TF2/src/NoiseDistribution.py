
# coding=gbk
import numpy as np


class NoiseDistribution(object):
    """
    ���ɾ���һ���������Ƶ�����
    """
    def __init__(self, range_):
        self.range = range_

    def sample(self, n):
        offset = np.random.random(n) * (float(self.range) / n)
        samples = np.linspace(-self.range, self.range, n) + offset  # [-5,5]���ּ�ƫ��ֵ
        samples = np.reshape(samples, newshape=[1, n, 1])
        return samples

    @staticmethod
    def sample_z(batch_size, seq_length, latent_dim):
        sample = np.float32(np.random.normal(size=[batch_size, seq_length, latent_dim]))
        return sample
