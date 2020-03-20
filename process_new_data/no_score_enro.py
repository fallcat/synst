import os


files = os.listdir()
files = [f for f in files if f.startswith('c.')]
for fname in files:
	with open(fname, "r") as f:
		data = f.read()

	data = data.replace("--enc-dec-attn-score 1", "--enc-dec-attn-score 0")

	with open(fname, "w") as f:
		f.write(data)