import pprint
from collections import defaultdict

file_path = "/mnt/nfs/work1/miyyer/wyou/data/small_enro/train.tok.bpe.32000.en"

lengths = defaultdict(int)
with open(file_path) as file:
    for line in file.readlines():
        line_list = line.split()
        lengths[len(line_list)] += 1

pp = pprint.PrettyPrinter(indent=4)
pp.pprint(lengths)