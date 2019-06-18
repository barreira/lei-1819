import os
import pandas as pd
import numpy as np
import io

root_dir = 'logs/8gb/mnist_full'

def get_csv_dataframe(subdir, f):
	filename = os.path.join(subdir, f)
	pathname = os.path.splitext(filename)[0]

	with open(filename) as fp:
		text = fp.readlines()[6:]
		text = io.StringIO(''.join(text))
		data = pd.read_csv(text)
		data = data.drop(['time', 'buff', 'cach', 'free', 'sys', 'idl', 'wai', 'hiq', 'siq'], axis=1)
	return data

def get_test_name(subdir):
	subdir = subdir.split('/')[-1]
	test = subdir.split('-')[-2:]
	test_string = test[0] + "-" + test[1]
	return test_string

def get_time_acc(subdir, f):
	filename = os.path.join(subdir, f)
	pathname = os.path.splitext(filename)[0]

	with open(filename) as fp:
		lines = fp.readlines()[-3:]
		acc_string = lines[0].split(' ')[-2]
		acc_parts = acc_string.split('/')
		acc = float(acc_parts[0]) / float(acc_parts[1])
		time = float(lines[2].split(' ')[-1])
	return time, acc

def get_df_with_std(df_sub, time, acc):
	df_std = df_sub.std()
	df_sub = df_sub.mean().to_frame().T
	df_sub['test'] = [get_test_name(subdir)]
	df_sub = df_sub[['test', 'used', 'usr', 'recv', 'send', 'read', 'writ']]
	df_sub['acc'] = [acc]
	df_sub['time'] = [time]
	
	used = '{:.2f} +/- {:.2f}'.format(df_sub['used'][0], df_std[0])	
	usr = '{:.2f} +/- {:.2f}'.format(df_sub['usr'][0], df_std[1])
	recv = '{:.2f} +/- {:.2f}'.format(df_sub['recv'][0], df_std[2])
	send = '{:.2f} +/- {:.2f}'.format(df_sub['send'][0], df_std[3])
	read = '{:.2f} +/- {:.2f}'.format(df_sub['read'][0], df_std[4])
	writ = '{:.2f} +/- {:.2f}'.format(df_sub['writ'][0], df_std[5])

	df_sub = df_sub.astype(str)
	df_sub.at[0, 'used'] = used
	df_sub.at[0, 'usr'] = usr
	df_sub.at[0, 'recv'] = recv
	df_sub.at[0, 'send'] = send
	df_sub.at[0, 'read'] = read
	df_sub.at[0, 'writ'] = writ

	return df_sub

data = pd.DataFrame(columns=['test', 'used', 'usr', 'recv', 'send', 'read', 'writ', 'acc', 'time'])

for subdir, _, _ in os.walk(root_dir):
	if subdir != root_dir:
		for _, _, files in os.walk(subdir):
			df_sub = pd.DataFrame(columns=['used', 'usr', 'recv', 'send', 'read', 'writ'])
			i = 0
			time = 0.0
			acc = 0.0
			time_array = []
			acc_array = []
			for f in files:
				if f.endswith('.csv'):
					df_sub = df_sub.append(get_csv_dataframe(subdir, f), ignore_index=True)
				else:
					i = i + 1
					a_time, a_acc = get_time_acc(subdir, f)
					time = time + a_time
					acc = acc + a_acc
					time_array.append(a_time)
					acc_array.append(a_acc)
			acc = acc / i
			time = time / i
			time_array = np.asarray(time_array)
			acc_array = np.asarray(acc_array)
			time_std = np.std(time_array)
			acc_std = np.std(acc_array)
			time_str = '{:.2f} +/- {:.2f}'.format(time, time_std)
			acc_str = '{:.2f} +/- {:.2f}'.format(acc, acc_std)

			df_sub = get_df_with_std(df_sub, time_str, acc_str)
			data = data.append(df_sub, ignore_index=True)
									
out_csv = root_dir + "/results.csv"
data.to_csv(out_csv, index = None, header=True)
