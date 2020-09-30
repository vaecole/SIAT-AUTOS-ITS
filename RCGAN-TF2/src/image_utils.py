
# coding=gbk
import os
import matplotlib.pyplot as plt


def plot_show(learn_set, g_data, index, con_dim, num_run, num_gen_once):
    """
    绘制图像
    """
    if not os.path.exists('generation' + str(learn_set)):
        os.makedirs('generation' + str(learn_set))
    for i in range(con_dim):
        for j in range(num_run):
            for k in range(num_gen_once):
                plt.plot(g_data[j][i][k], 'lightcoral', linewidth=3, label='generated data')
                plt.xticks(fontsize=20)
                plt.yticks(fontsize=20)
                plt.title('Real Data and Generated Data' + ' label ', fontsize=22)
                plt.ylabel('Unoccupied Parking Space Rate', fontsize=22)
                plt.xlabel('Time Point', fontsize=22)
                plt.legend(fontsize=20)
                plt.grid(True)
                fig = plt.gcf()
                fig.set_size_inches(15, 8)
                fig.savefig('generation' + str(learn_set) + '/' + 'label' + str(i) + '_' + 'num_run'
                            + str(j) + '_number' + str(k) + '.png', dpi=100, bbox_inches='tight')
                plt.show()
