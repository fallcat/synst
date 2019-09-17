with open('../iwslt/train.tok.en', 'rt') as file1:
    with open('../iwslt/train.tok.de', 'rt') as file2:
        with open('../iwslt/train_tok_en_de.txt', 'wt') as file3:
            for x, y in zip(file1, file2):
                file3.write(x[:-1] + ' ||| ' + y[:-1] + '\n')