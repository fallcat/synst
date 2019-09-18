import pickle
import numpy as np

split_portion = 100

with open('../iwslt/train.tok.en', 'rt') as file_en:
    with open('../iwslt/train.tok.de', 'rt') as file_de:
        with open('../iwslt/forward.align', 'rt') as file_fa:
            with open('../iwslt/forward.align.pickle', 'wb') as file_fas:
                final_count = {}
                stats = {}
                for x, y, z in zip(file_en, file_de, file_fa):
                    z_dict = {}
                    len_x = len(x.split())
                    for z_element in z.split():
                        a, b = z_element.split('-')
                        z_dict[round(int(b) + 1 / len_x * split_portion) - 1] = round(int(a) + 1 / len_x * split_portion) - 1
                    for i, y_word in enumerate(y.split()):
                        new_i = round(int(i) + 1 / len_x * split_portion) - 1
                        if new_i in z_dict:
                            if y_word in final_count:
                                if new_i in final_count[y_word]:
                                    final_count[y_word][new_i].append(z_dict[new_i])
                                else:
                                    final_count[y_word][new_i] = [z_dict[new_i]]
                            else:
                                final_count[y_word] = {}
                                final_count[y_word][new_i] = [z_dict[new_i]]
                for key in final_count:
                    stats[key] = {}
                    mean_offsets = []
                    stds = []
                    for num in final_count[key]:
                        mean = np.mean(final_count[key][num])
                        std = np.std(final_count[key][num])
                        stats[key][num] = {}
                        stats[key][num]['mean'] = mean
                        stats[key][num]['std'] = std
                        mean_offsets.append(mean - num)
                        stds.append(std)
                    stats[key]['mean_offsets_mean'] = np.mean(mean_offsets)
                    stats[key]['stds_mean'] = np.mean(stds)
                pickle.dump(stats, file_fas)
