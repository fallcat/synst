import os

script_path = './scripts/iwslt_grid_search/'
files = sorted(os.listdir(script_path))
write_line = ''
for fname in files:
	with open(os.path.join(script_path, fname), 'r') as f:
		lines = f.readlines()
		params = lines[17]
		write_line += fname.split('_')[-1][:-3] + '\t' + params
with open('params_by_expid.txt', 'w') as f:
	f.write(write_line)

