import pickle
import torch
from models.utils import MODEL_STATS, STATS_TYPES


def print_matrix(matrix, output_file):
    for row in matrix:
        for item in row:
            output_file.write(f"{item.item()} \t")
        output_file.write("\n")


def main():
    with open('/mnt/nfs/work1/miyyer/wyou/synst/experiments/iwslt01/stats_100000.pickle', 'rb') as stats_file:
        with open('/mnt/nfs/work1/miyyer/wyou/synst/experiments/iwslt01/stats_100000.txt', 'wt') as output_file:
            stats = pickle.load(stats_file)
            for model_stat in MODEL_STATS:
                output_file.write("========== " + model_stat + " ==========\n")
                for stats_type in STATS_TYPES:
                    output_file.write("---------- " + stats_type + " ----------\n")
                    output_file.write("mean:\n")
                    print_matrix(stats['stats'][model_stat][stats_type]['mean'], output_file)
                    output_file.write("std:\n")
                    print_matrix(torch.sqer(stats['stats'][model_stat][stats_type]['var']), output_file)


if __name__ == "__main__":
    main()
