# coding=gbk
import os

import matplotlib.pyplot as plt
import numpy as np


def save_images(save_path, epoch, generator, noise, real_weekend, real_workday):
    """
    ...
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    condition_weekend = np.array([[1, 0]]).repeat(96, axis=0)  # weekend
    condition_workday = np.array([[0, 1]]).repeat(96, axis=0)  # workday
    image_weekend = generator(noise, condition_weekend, False)
    image_workday = generator(noise, condition_workday, False)

    plt.clf()
    plt.plot(noise, 'grey', linewidth=1, label='Noise')
    plt.plot(image_weekend, 'blue', linewidth=3, label='Generated Weekend')
    plt.plot(image_workday, 'red', linewidth=3, label='Generated Workday')
    plt.plot(real_weekend, 'green', linewidth=2, label='Real Weekend')
    plt.plot(real_workday, 'purple', linewidth=2, label='Real Workday')
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.title('Noise Data and Generated Data' + ' weekend ', fontsize=22)
    plt.ylabel('Unoccupied Parking Space Rate', fontsize=22)
    plt.xlabel('Time Point', fontsize=22)
    plt.legend(fontsize=20)
    plt.grid(True)
    fig = plt.gcf()
    fig.set_size_inches(15, 8)
    fig.savefig(save_path + '/parking_rate_%d' % epoch + '.png', dpi=100, bbox_inches='tight')
    # plt.show()
