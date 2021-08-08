import os
import sys

table_fname = sys.argv[1]
delete_op = sys.argv[2]
save_fname = sys.argv[3]

# table_fname = os.path.join(base_dir, "non-isom.txt")
# delete_op = "nor_conv_1x1"
with open(table_fname) as rf:
    lines = rf.readlines()

delete_names = []
for line in lines:
    if delete_op not in line:
        delete_names.append(line)

print("Num: {}; After delete: {}".format(len(lines), len(delete_names)))
with open(save_fname, "w") as w_f:
    w_f.write("".join(delete_names))
