with open('/mnt/nfs/work1/miyyer/wyou/wmt/train.tok.clean.bpe.32000.en') as file1:
    with open('/mnt/nfs/work1/miyyer/wyou/wmt/train.tok.clean.bpe.32000.de') as file2:
        count = 0
        ratio = 0.0
        for x, y in zip(file1, file2):
            x_list = x.split()
            y_list = y.split()
            ratio += len(x_list) / float(len(y_list))
            count += 1
        print(ratio / count)
