import pickle
import math
import numpy as np

split_portion = 4
WORD_COUNT = (1.0360595565014956, 1)

with open('../iwslt/train.tok.en', 'rt') as file_en:
    with open('../iwslt/train.tok.de', 'rt') as file_de:
        with open('../iwslt/forward.subword.align', 'rt') as file_fa:
            with open('../iwslt/forward.subword.align.4.pickle', 'wb') as file_fas:
                final_count = {}
                stats = {}
                for x, y, z in zip(file_en, file_de, file_fa):
                    z_dict = {}
                    len_x = len(x.split())
                    len_y = len(y.split())
                    for z_element in z.split():
                        print("z_element", z_element)
                        a, b = z_element.rsplit('-')
                        a, b = float(a), int(b)
                        key = math.ceil((b + 0.5) / len_y * split_portion)  # round((b + 1) / len_x * split_portion) - 1
                        if key in z_dict:
                            z_dict[key].append(a - b * 1.0360595565014956)  # round((int(a) + 1) / len_x * split_portion) - 1
                        else:
                            z_dict[key] = [a - b * WORD_COUNT[0]]
                    for i, y_word in enumerate(y.split()):
                        new_i = round((int(i) + 1) / len_x * split_portion) - 1
                        if new_i in z_dict:
                            if y_word not in final_count:
                                final_count[y_word] = {}
                            if new_i in final_count[y_word]:
                                old_mean = final_count[y_word][new_i]['mean']
                                old_var = final_count[y_word][new_i]['var']
                                old_count = final_count[y_word][new_i]['count']
                                current_count = len(z_dict[new_i])
                                current_mean = np.mean(z_dict[new_i])
                                current_var = np.var(z_dict[new_i])
                                new_count = old_count + current_count
                                new_mean = (old_mean * old_count + sum(z_dict[new_i])) / new_count
                                new_var = (old_count * (old_var + (old_mean - new_mean) ** 2) + current_count * (current_var + (current_mean - new_mean) ** 2)) / new_count
                                final_count[y_word][new_i]['mean'] = new_mean
                                final_count[y_word][new_i]['var'] = new_var
                                final_count[y_word][new_i]['count'] = new_count
                            else:
                                final_count[y_word][new_i] = {}
                                final_count[y_word][new_i]['mean'] = np.mean(z_dict[new_i])
                                final_count[y_word][new_i]['var'] = np.var(z_dict[new_i])
                                final_count[y_word][new_i]['count'] = len(z_dict[new_i])
                for key in final_count:
                    stats[key] = {}
                    means = []
                    stds = []
                    for num in final_count[key]:
                        stats[key][num] = {}
                        stats[key][num]['mean'] = final_count[key][num]['mean']
                        stats[key][num]['std'] = math.sqrt(final_count[key][num]['var'])
                        means.append(stats[key][num]['mean'])
                        stds.append(stats[key][num]['std'])
                    stats[key]['means_mean'] = np.mean(means)
                    stats[key]['stds_mean'] = np.mean(stds)
                pickle.dump(stats, file_fas)
