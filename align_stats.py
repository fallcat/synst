split_portion = 50

with open('../iwslt/train.tok.en', 'rt') as file_en:
    with open('../iwslt/train.tok.de', 'rt') as file_de:
        with open('../iwslt/forward.align', 'rt') as file_fa:
            with open('../iwslt/forward.align.stats', 'wt') as file_fas:
                final_count = {}
                for x, y, z in zip(file_en, file_de, file_fa):
                    z_dict = {}
                    len_x = len(x.split())
                    for z_element in z.split():
                        a, b = z_element.split('-')
                        z_dict[int(b)] = round(int(a) + 1 / len_x * split_portion) - 1
                    for i, y_word in enumerate(y.split()):
                        if i in z_dict:
                            if y_word in final_count:
                                if i in final_count[y_word]:
                                    final_count[y_word][i].append(z_dict[i])
                                else:
                                    final_count[y_word][i] = [z_dict[i]]
                            else:
                                final_count[y_word] = {}
                                final_count[y_word][i] = [z_dict[i]]
                for key in final_count:
                    file_fas.write(key + ':' + str(final_count[key]) + '\n')
