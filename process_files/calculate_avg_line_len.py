import argparse

parser = argparse.ArgumentParser(description='calculate average line len')

parser.add_argument(
        '--file',
        type=str,
        nargs='+',
        default='/mnt/nfs/work1/miyyer/wyou/wmt/train.tok.clean.bpe.32000.en',
        help='Names of files to select lines from'
    )
args = parser.parse_args()

with open(args.file, 'rt') as file1:
    count = 0
    lang1 = 0
    for x in file1.readlines():
        x_list = x.strip().split()
        lang1 += len(x_list)
        count += 1
    print(lang1/float(count))
