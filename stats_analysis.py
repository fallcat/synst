import pickle
import torch
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from models.utils import MODEL_STATS, STATS_TYPES


def print_matrix(matrix, output_file):
    for row in matrix:
        for item in row:
            output_file.write(f"{item} \t")
        output_file.write("\n")


def visualize(mean, std, num_layers, num_heads, fig_path, fig_name):
    fig, ax = plt.subplots()
    width = 0.35
    ind = np.arange(num_layers)

    def autolabel(rects):
        """
        Attach a text label above each bar displaying its height
        """
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2., 1.05 * height,
                    '%d' % int(height),
                    ha='center', va='bottom')

    p = [ax.bar(ind + width * i, mean[:, i], width, bottom=0, yerr=std[:, i]) for i in range(num_heads)]
    ax.set_title(fig_name)
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(np.arange(1, num_layers + 1))
    ax.set_xlabel('Layer')
    for rects in p:
        autolabel(rects)

    ax.legend((pn[0] for pn in p), ("Head " + str(i) for i in range(1, num_heads + 1)))
    ax.autoscale_view()

    plt.savefig(fig_path)
    plt.close(fig)


splits = ['train', 'test']


def main():
    base_path = 'experiments/iwslt01/'
    stats_path = 'stats_100000'
    with open(base_path + stats_path + '.pickle', 'rb') as stats_file:
        with open(base_path + stats_path + '.txt', 'wt') as output_file:
            stats = pickle.load(stats_file)
            for split in splits:
                num_layers = stats[split + '_stats'][MODEL_STATS[0]][STATS_TYPES[0]]['mean'].size()[0]
                num_heads = stats[split + '_stats'][MODEL_STATS[0]][STATS_TYPES[0]]['mean'].size()[1]
                for model_stat in MODEL_STATS:
                    output_file.write("========== " + model_stat + " ==========\n")
                    for stats_type in STATS_TYPES:
                        output_file.write("---------- " + stats_type + " ----------\n")
                        output_file.write("mean:\n")
                        mean = stats[split + '_stats'][model_stat][stats_type]['mean'].cpu().numpy()
                        print_matrix(mean, output_file)
                        output_file.write("std:\n")
                        std = torch.sqrt(stats[split + '_stats'][model_stat][stats_type]['var']).cpu().numpy()
                        print_matrix(std, output_file)
                        fig_path = base_path + stats_path + '_' + split + '_' + model_stat + '_' + stats_type + '.pdf'
                        fig_name = split.capitalize() + ' - ' + model_stat.capitalize() + ' - ' + stats_type.capitalize()
                        visualize(mean, std, num_layers, num_heads, fig_path, fig_name)


if __name__ == "__main__":
    main()
