with open('../iwslt/train.bpe.idx.mapping.de', 'rt') as source_file:
    with open('../iwslt/train.bpe.idx.mapping.en', 'rt') as target_file:
        with open('../iwslt/reverse.align', 'rt') as align_file:
            with open('../iwslt/reverse.subword.align', 'wt') as output_file:
                for source_line, target_line, align_line in zip(source_file, target_file, align_file):
                    source_list = [[int(x) for x in w.split(':')[1].split(',')] for w in source_line.split()]
                    target_list = [[int(x) for x in w.split(':')[1].split(',')] for w in target_line.split()]
                    align_list = [[int(x) for x in w.split('-')] for w in align_line.split()]
                    output_list = []
                    for item in align_list:
                        source = item[1]
                        target = item[0]
                        new_source = source_list[source]
                        new_target = target_list[target]
                        if len(new_source) == 1 and len(new_target) == 1:
                            output_list.append([new_source[0], new_target[0]])
                        else:
                            base_source = new_source[0] - 0.5
                            base_target = new_target[0] - 0.5
                            output_list.extend([[base_source + (new_source[-1] - new_source[0] + 1) *
                                                (w - base_target) / (new_target[-1] + 0.5 - base_target), w]
                                                for i, w in enumerate(new_target)])
                    output_file.write(' '.join(['-'.join([str(x) for x in item]) for item in output_list]) + '\n')
