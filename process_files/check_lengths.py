import pprint
from collections import defaultdict

file_path = "/mnt/nfs/work1/miyyer/wyou/data/small_enro/train.tok.bpe.32000.en"

lengths = defaultdict(int)
count = 0
with open(file_path) as file:
    for line in file.readlines():
        line_list = line.split()
        lengths[len(line_list)] += 1
        count += 1

pp = pprint.PrettyPrinter(indent=4)
pp.pprint(lengths)

bins = 6
bins_list = []
curr_bin = 0
max_sentences = count/bins
for key in sorted(lengths.keys()):
    curr_bin += lengths[key]
    if curr_bin > max_sentences:
        bins_list.append(key)
        curr_bin = 0

print(bins_list)
# [15, 22, 30, 42]
