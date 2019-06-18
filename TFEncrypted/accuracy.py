from __future__ import division
import os

path = './logs'
files = []

def line_to_list(line):
	return [int(s) for s in line.split() if s.isdigit()]

def get_accuracy(expected, result):
	total = 0
	for idx, value in enumerate(expected):
		if value == result[idx]:
			total = total + 1
	return total / len(expected)
	

for r, _, f in os.walk(path):
	for file in f:
		if '.txt' in file:
			files.append(os.path.join(r, file))

for f in files:
	accuracy = 0.0
	count = 0
	has_results = False
	expect_array = []
	result_array = []
	with open(f) as fp:
		for line in fp:
			if 'Expect' in line:
				expected_array = line_to_list(line)
				has_results = False
			if 'Result' in line:
				result_array = line_to_list(line)
				has_results = True
				count = count + 1
				accuracy = accuracy + get_accuracy(expected_array, result_array)
		accuracy = accuracy / count
	with open(f, 'a') as fp:
		fp.write("Accuracy: {}\n".format(accuracy))
				
