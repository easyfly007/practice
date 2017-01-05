import numpy as np 
import random

def loaddataset(filename):
	datamat = []
	labelmat = []
	f = open(filename)
	for line in f.readlines():
		items = line.strip().split('\t')
		datamat.append([float(items[0]), float(items[1])] )
		labelmat.append(float(items[2]))
	return np.mat(datamat), np.mat(labelmat)

def selectjrand(i, m):
	j=i
	while (j==i):
		j = int(random.uniform(0, m))
	return j

def clipalpha(aj, high, low):
	if aj >high:
		aj = high
	if aj < low:
		aj = low
	return aj
