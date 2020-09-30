
import os
import matplotlib.pyplot as plt

#绘制图像
def plot_show(learn_set,g_data,sample_data,con_dim,num_run,num_gen_once):
    if not os.path.exists('generation' + str(learn_set)):
        os.makedirs('generation' + str(learn_set))
        plt.plot(sample_data, 'red', linewidth=3, label='sample data')
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.title('Real Data and Generated Data' + ' label ', fontsize=22)
        plt.ylabel('Unoccupied Parking Space Rate', fontsize=22)
        plt.xlabel('Time Point', fontsize=22)
        plt.legend(fontsize=20)
        plt.grid(True)
        fig = plt.gcf()
        fig.set_size_inches(15, 8)
        #fig.savefig('generation' + str(learn_set) + '/' + 'label' + str(i) + '_' + 'num_run'
         #           + str(j) +'_number'+str(k)+ '.png',dpi=100, bbox_inches='tight')
        plt.show()
