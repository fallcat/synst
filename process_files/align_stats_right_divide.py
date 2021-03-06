import pickle
import math
import numpy as np

split_portion = 4
WORD_COUNT = (1.0360595565014956, 1)

SOS = '<SOS>'
EOS = '<EOS>'

with open('../iwslt/train.tok.bpe.32000.en', 'rt') as file_en:
    with open('../iwslt/train.tok.bpe.32000.de', 'rt') as file_de:
        with open('../iwslt/reverse.subword.align', 'rt') as file_fa:
            with open('../iwslt/reverse.subword.align.right2.len.4.pickle', 'wb') as file_fas:
                final_count = {}
                stats = {}
                for x, y, z in zip(file_de, file_en, file_fa):
                    z_dict = {}
                    x_list = x.split()
                    y_list = y.split()
                    len_x = len(x_list)
                    len_y = len(y_list)
                    for z_element in z.split():
                        try:
                            a, b = z_element.rsplit('-', 1)
                        except:
                            print("z_element", z_element)
                        a, b = float(a), int(b)
                        key = math.ceil((b + 0.5) / len_y * split_portion)  # round((b + 1) / len_x * split_portion) - 1
                        if key > split_portion:
                            print("key", key)
                            print("b", b)
                            print("len_y", len_y)
                        # print("b", b)
                        if b == 0:
                            # print("inside")
                            w = SOS
                            # print("w", w)
                        else:
                            w = y_list[b - 1]
                        if w not in z_dict:
                            z_dict[w] = {}
                        if key in z_dict[w]:
                            z_dict[w][key].append((a - b / WORD_COUNT[0]) / len_x)  # round((int(a) + 1) / len_x * split_portion) - 1
                        else:
                            z_dict[w][key] = [(a - b / WORD_COUNT[0]) / len_x]

                    # last one point to EOS
                    w = y_list[-1]
                    key = split_portion
                    a = len_x
                    b = len_y - 1
                    if w not in z_dict:
                        z_dict[w] = {}
                    if key in z_dict[w]:
                        z_dict[w][key].append(
                            (a - b / WORD_COUNT[0]) / len_x)  # round((int(a) + 1) / len_x * split_portion) - 1
                    else:
                        z_dict[w][key] = [(a - b / WORD_COUNT[0]) / len_x]

                    for w in z_dict:
                        for k in z_dict[w]:
                            if w not in final_count:
                                final_count[w] = {}
                            if k in final_count[w]:
                                old_mean = final_count[w][k]['mean']
                                old_var = final_count[w][k]['var']
                                old_count = final_count[w][k]['count']
                                current_count = len(z_dict[w][k])
                                current_mean = np.mean(z_dict[w][k])
                                current_var = np.var(z_dict[w][k])
                                new_count = old_count + current_count
                                new_mean = (old_mean * old_count + sum(z_dict[w][k])) / new_count
                                new_var = (old_count * (old_var + (old_mean - new_mean) ** 2) + current_count * (current_var + (current_mean - new_mean) ** 2)) / new_count
                                final_count[w][k]['mean'] = new_mean
                                final_count[w][k]['var'] = new_var
                                final_count[w][k]['count'] = new_count
                            else:
                                final_count[w][k] = {}
                                final_count[w][k]['mean'] = np.mean(z_dict[w][k])
                                final_count[w][k]['var'] = np.var(z_dict[w][k])
                                final_count[w][k]['count'] = len(z_dict[w][k])

                    # for i, y_word in enumerate(y_list):
                    #     new_i = round((int(i) + 1) / len_x * split_portion) - 1
                    #     if y_word in z_dict and new_i in z_dict[y_word]:
                    #         if y_word not in final_count:
                    #             final_count[y_word] = {}
                    #         if new_i in final_count[y_word]:
                    #             old_mean = final_count[y_word][new_i]['mean']
                    #             old_var = final_count[y_word][new_i]['var']
                    #             old_count = final_count[y_word][new_i]['count']
                    #             current_count = len(z_dict[y_word][new_i])
                    #             current_mean = np.mean(z_dict[y_word][new_i])
                    #             current_var = np.var(z_dict[y_word][new_i])
                    #             new_count = old_count + current_count
                    #             new_mean = (old_mean * old_count + sum(z_dict[y_word][new_i])) / new_count
                    #             new_var = (old_count * (old_var + (old_mean - new_mean) ** 2) + current_count * (current_var + (current_mean - new_mean) ** 2)) / new_count
                    #             final_count[y_word][new_i]['mean'] = new_mean
                    #             final_count[y_word][new_i]['var'] = new_var
                    #             final_count[y_word][new_i]['count'] = new_count
                    #         else:
                    #             final_count[y_word][new_i] = {}
                    #             final_count[y_word][new_i]['mean'] = np.mean(z_dict[y_word][new_i])
                    #             final_count[y_word][new_i]['var'] = np.var(z_dict[y_word][new_i])
                    #             final_count[y_word][new_i]['count'] = len(z_dict[y_word][new_i])

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
