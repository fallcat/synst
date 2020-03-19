import random

# total = 612422
# sample_size = 4000
# sampled = random.sample(range(total), sample_size)
# sampled2 = random.sample(sampled, 2000)
#
# print(sampled)
# print(len(sampled))
#
# with open("sampled_train_idx4000.txt", "wt") as output_file:
#     for n in sorted(sampled):
#         if n not in sampled2:
#             output_file.write(str(n) + '\n')
#
# with open("sampled_train_idx2000.txt", "wt") as output_file:
#     for n in sorted(sampled2):
#         output_file.write(str(n) + '\n')

# ------------------------------------------------------------
idx_filename = "/mnt/nfs/work1/miyyer/wyou/synst/process_files/sampled_train_idx4000.txt"
with open(idx_filename, "rt") as idx_file:
    with open('/mnt/nfs/work1/miyyer/wyou/data/small_enro/train.tok.bpe.32000.ro', 'rt') as ro_file:
        with open('/mnt/nfs/work1/miyyer/wyou/data/small_enro/train.tok.bpe.32000.en', 'rt') as en_file:
            with open('/mnt/nfs/work1/miyyer/wyou/data/small_enro/train4000.tok.bpe.32000.ro', 'wt') as output_ro_file:
                with open('/mnt/nfs/work1/miyyer/wyou/data/small_enro/train4000.tok.bpe.32000.en', 'wt') as output_en_file:
                    ro_lines = list(ro_file.readlines())
                    en_lines = list(en_file.readlines())
                    for idx in idx_file.readlines():
                        output_ro_file.write(ro_lines[int(idx)])
                        output_en_file.write(en_lines[int(idx)])
