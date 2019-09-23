import pickle
import math
import numpy as np

split_portion = 4
WORD_COUNT = (1.0360595565014956, 1)

with open('../iwslt/train.tok.en', 'rt') as file_en:
    with open('../iwslt/train.tok.de', 'rt') as file_de:
        with open('../iwslt/forward.align', 'rt') as file_fa:
            with open('../iwslt/forward.align.4.pickle', 'wb') as file_fas:
                final_count = {}
                stats = {}
                for x, y, z in zip(file_en, file_de, file_fa):
                    z_dict = {}
                    len_x = len(x.split())
                    len_y = len(y.split())
                    for z_element in z.split():
                        a, b = z_element.split('-')
                        a, b = int(a), int(b)
                        key = math.ceil((b + 0.5) / len_y * split_portion)  # round((b + 1) / len_x * split_portion) - 1
                        if key in z_dict:
                            print("z_dict", z_dict)
                            print("key", key)
                            print("z_dict[key]", z_dict[key])
                            z_dict[key].append(a - b * 1.0360595565014956)  # round((int(a) + 1) / len_x * split_portion) - 1
                        else:
                            z_dict = [a - b * WORD_COUNT[0]]
                    for i, y_word in enumerate(y.split()):
                        new_i = round((int(i) + 1) / len_x * split_portion) - 1
                        if new_i in z_dict:
                            if y_word in final_count:
                                if new_i in final_count[y_word]:
                                    final_count[y_word][new_i].extend(z_dict[new_i])
                                else:
                                    final_count[y_word][new_i] = z_dict[new_i]
                            else:
                                final_count[y_word] = {}
                                final_count[y_word][new_i] = z_dict[new_i]
                for key in final_count:
                    stats[key] = {}
                    means = []
                    stds = []
                    for num in final_count[key]:
                        mean = np.mean(final_count[key][num])
                        std = np.std(final_count[key][num])
                        stats[key][num] = {}
                        stats[key][num]['mean'] = mean
                        stats[key][num]['std'] = std
                        means.append(mean)
                        stds.append(std)
                    stats[key]['means_mean'] = np.mean(means)
                    stats[key]['stds_mean'] = np.mean(stds)
                pickle.dump(stats, file_fas)
