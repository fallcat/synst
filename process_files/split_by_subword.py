with open('../iwslt/train.tok.bpe.32000.de', 'rt') as input_file:
    with open('../iwslt/train.bpe.idx.mapping.de', 'wt') as output_file:
        for line in input_file.readlines():
            line_list = line.split()
            mapping = []
            new_subwords = []
            for i, w in enumerate(line_list):
                new_subwords.append(i)
                if w[-2:] != '@@':
                    mapping.append(new_subwords)
                    new_subwords = []
            s = ' '.join([str(i)+':'+','.join([str(n) for n in subwords]) for i, subwords in enumerate(mapping)])
            output_file.write(s + '\n')