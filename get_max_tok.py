"""

	input: script for training the model
	output: max tok

	process: read in the script, execute the script with different -b, get return value
			 (set all max-steps to 100)

"""
import re
import os
import sys
import math

SCRIPT_FILE = sys.argv[1]

def roundup(val):
	return 5 * math.ceil(val / 5)

# read file
with open(SCRIPT_FILE, 'r') as f:
	data = f.readlines()

# process script
data = ' '.join([l for l in data if not l.startswith("#")])
data = data.replace("env $(cat ~/.comet.ml | xargs) ", "").replace("--track", "")
data = data.replace("--checkpoint-directory $EXPERIMENT_PATH", "--max-steps 100")
data = data.replace("--accumulate 2", "--accumulate 1")

# get batch size
pat = re.compile("-b (\d+)")
bsize = pat.findall(data)[0]

# try oom
# 1) first try bsize * n
low = int(bsize)
old = low
for i in range(2, 20):
	new_b = int(old) * i
	print("trying %i" % new_b)
	data = data.replace("-b %s" % bsize, "-b %i" % new_b)
	bsize = new_b
	ret = os.system(data)
	os.system("rm -rf /tmp/synst")
	if ret != 0:
		up = new_b # upper bound
		break
	else:
		low = new_b

# 2) binary search
while True:
	mid = low + roundup((up - low) / 2 )
	print("trying %i" % mid)
	data = data.replace("-b %i" % new_b, "-b %i" % mid)
	ret = os.system(data)
	os.system("rm -rf /tmp/synst")
	if ret != 0: # oom
		up = mid
		new_b = up
	else:
		low = mid
		new_b = low
	if abs(low - up) <= 10:
		break

with open("max-tok.res", "a+") as f:
	f.write("%s %i\n" % (SCRIPT_FILE, low))




