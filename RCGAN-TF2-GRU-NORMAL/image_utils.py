# coding=gbk
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def save_images(save_path, epoch, generator, noise, avg_weekend, avg_workday):
    """
    ...
    """
    condition_weekend = np.array([[1, 0]]).repeat(96, axis=0)  # weekend
    condition_workday = np.array([[0, 1]]).repeat(96, axis=0)  # workday
    image_weekend = generator(noise, condition_weekend, False)
    image_workday = generator(noise, condition_workday, False)

    plt.clf()
    plt.plot(noise, 'grey', linewidth=1, label='Noise')
    plt.plot(image_weekend, 'blue', linewidth=3, label='Generated Weekend')
    plt.plot(image_workday, 'red', linewidth=3, label='Generated Workday')
    plt.plot(avg_weekend, 'green', linewidth=2, label='Weekend AVG')
    plt.plot(avg_workday, 'purple', linewidth=2, label='Workday AVG')
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.title('Real AVG Data and Generated Data', fontsize=22)
    plt.ylabel('Unoccupied Parking Space Rate', fontsize=22)
    plt.xlabel('Time Point(96x15 minutes)', fontsize=22)
    plt.legend(fontsize=20)
    plt.grid(True)
    fig = plt.gcf()
    fig.set_size_inches(15, 8)
    fig.savefig(save_path + '/%d' % epoch + '.png', dpi=100, bbox_inches='tight')


def save_images_batch(save_path, epoch, generator, noise, avg_weekend, avg_workday, clear, save):
    """
    ...
    """
    condition_weekend = np.array([[1, 0]]).repeat(96, axis=0)  # weekend
    condition_workday = np.array([[0, 1]]).repeat(96, axis=0)  # workday
    image_weekend = generator(noise, condition_weekend, False)
    image_workday = generator(noise, condition_workday, False)

    if clear:
        plt.clf()
    # plt.plot(noise, 'grey', linewidth=1)  # , label='Noise')
    plt.plot(image_weekend, 'blue', linewidth=3)  # , label='Generated Weekend')
    plt.plot(image_workday, 'red', linewidth=3)  # , label='Generated Workday')
    plt.plot(avg_weekend, 'green', linewidth=2)  # , label='Weekend AVG')
    plt.plot(avg_workday, 'purple', linewidth=2)  # , label='Workday AVG')
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.title('Real AVG Data and Generated Data', fontsize=22)
    plt.ylabel('Unoccupied Parking Space Rate', fontsize=22)
    plt.xlabel('Time Point(96x15 minutes)', fontsize=22)
    plt.legend(fontsize=20)
    plt.grid(True)
    fig = plt.gcf()
    fig.set_size_inches(15, 8)
    if save:
        fig.savefig(save_path + '/%s' % epoch + '.png', dpi=100, bbox_inches='tight')


def sample_noise(n):
    """
    ...
    """
    return tf.random.normal([n, 1], 0.5, 0.2)
