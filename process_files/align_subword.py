with open('../iwslt/train.bpe.idx.mapping.en', 'rt') as source_file:
    with open('../iwslt/train.bpe.idx.mapping.de', 'rt') as target_file:
        with open('../iwslt/forward.align', 'rt') as align_file:
            with open('../iwslt/forward.subword.align', 'wt') as output_file:
                for source_line, target_line, align_line in zip(source_file, target_file, align_file):
                    source_list = [w.split(':')[1].split(',') for w in source_line.split()]
                    target_list = [w.split(':')[1].split(',') for w in target_line.split()]
                    align_list = [w.split('-') for w in align_line.split()]
                    print(align_list)
                    output_list = []
                    for item in align_list:
                        print(item)
                        source = item[0]
                        target = item[1]
                        new_source = source_list[source]
                        new_target = target_list[target]
                        if len(new_source) == 1 and len(new_target == 1):
                            output_list.append([new_source[0], new_target[0]])
                        else:
                            base = new_source[0] - 0.5
                            output_list.extend([[base + (new_source[-1] - new_source[0] + 1) *
                                                (new_target[i] - base) / (new_target[-1] + 0.5 - base), new_target[i]]
                                                for i in new_target])
                    output_file.write(' '.join(['-'.join(str(item)) for item in output_list]))
