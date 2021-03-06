import argparse
import os
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser(description='Select lines from translated files')
    parser.add_argument('--sum', dest='accumulate', action='store_const',
                        const=sum, default=max,
                        help='sum the integers (default: find the max)')

    parser.add_argument(
            '--translated-files',
            '-t',
            type=str,
            nargs='+',
            default='/mnt/nfs/work1/miyyer/wyou/synst/experiments/iwslt01/translated_100000.txt',
            help='Names of files to select lines from'
        )

    parser.add_argument(
            '--split-file',
            '-s',
            type=str,
            default='/mnt/nfs/work1/miyyer/wyou/synst/experiments/iwslt01/off_diagonal_pairs_100000_2_number_3.txt',
            help='The file that contains which lines are from which bin'
        )

    parser.add_argument(
        '--coder',
        '-c',
        type=str,
        choices=['encoder', 'decoder'],
        default='encoder',
        help='Store if it is from encoder or decoder'
    )

    args = parser.parse_args()

    with open(args.split_file, 'rt') as split_file:
        split_dict = {}
        split_reverse_dict = {}
        for line in split_file.readlines():
            line_list = line.split('\t')
            split_dict[int(float(line_list[0]))] = [int(x) for x in line_list[1].split()]
            for x in line_list[1].split():
                split_reverse_dict[int(x)] = int(float(line_list[0]))
        if type(args.translated_files) is not list:
            translated_files = [args.translated_files]
        else:
            translated_files = args.translated_files
        for file in translated_files:
            split_names = str(os.path.basename(file)).split('.')
            basename = split_names[0]
            dirname = os.path.dirname(file)
            new_files = defaultdict(str)
            with open(file, 'rt') as input_file:
                for i, line in enumerate(input_file.readlines()):
                    new_files[split_reverse_dict[i]] += line
            for k in new_files.keys():
                new_file_path = os.path.join(dirname, basename + '_bin' + str(k) + '_' + args.coder + '.' + split_names[1])
                with open(new_file_path, 'wt') as output_file:
                    output_file.write(new_files[k])




if __name__ == "__main__":
    main()