import os
import pdb

template_path = "rl-scripts/small_enro_rl04/oracle_translate0000.sh"
experiment_folder = "rl-scripts/small_enro_rl04/"
experiment_folder2 = "rl-scripts/small_enro_rl05/"
#
# with open("not_found_ids2.txt", "rt") as file:
#     ids = [int(i.strip()) for i in file.readlines()]
#     print(ids)
#     count = 0
#     with open(template_path, "rt") as template_file:
#         template_text = str(template_file.read())
#
#     while count < int(len(ids)/35) + 1:
#         finished = False
#         if 35*(count + 1)-1 > len(ids) - 1:
#             end = len(ids) - 1
#         else:
#             end = 35*(count + 1)-1
#         with open("rl-scripts/new2/random-oracle-example-" + str(ids[35*count]) + "-" + str(ids[end])+".sh", "wt") as output_file:
#             new_write = template_text
#
#             new_write = new_write.replace("--job-name=layermask-eval-0-0", "--job-name=layermask-eval-"
#                               + str(ids[count*35]) + "-" + str(ids[end]))
#             for i in range(end - 35*count + 1):
#                 # try:
#                 new_write = new_write.replace("--example-id " + str(i) + " \\", "--example-id " + str(ids[count*35 + i]) + " \\")
#                 # except:
#                 #     finished = True
#                 #     break
#             # print(new_write)
#             output_file.write(new_write)
#         if finished:
#             break
#         count += 1
#


for filename in os.listdir(experiment_folder):
    if filename.startswith("oracle_translate"):
        with open(os.path.join(experiment_folder, filename), 'rt') as input_file:
            with open(os.path.join(experiment_folder, "test_" + filename), 'wt') as output_file:
                text = str(input_file.read())
                text = text.replace('--job-name=', '--job-name=04t')
                # text = text.replace('small_enro_rl04', 'small_enro_rl05')
                text = text.replace('2080ti-short', '1080ti-long')
                text = text.replace('--split valid', '--split test')
                text = text.replace('--fix-combination', '--output-filename test_oracle --fix-combination')
                output_file.write(text)