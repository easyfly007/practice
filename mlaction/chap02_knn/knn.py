import numpy as np
import operator
import os
import matplotlib
import matplotlib.pyplot as plt

def createdataset():
	group = np.array([[1.0, 1.1],[1.0, 1.0], [0.0, 0.1]])
	labels = ['A', 'A', 'B', 'B']
	return group, labels
	
def classify0(inX, dataset, labels, k):
	datasetsize = dataset.shape[0] 
	# get the shape size of the dataset
	diffmat = np.tile(inX, (datasetsize, 1) ) - dataset
	# print(inX)
	# print(datasetsize)
	# print(diffmat)
	# print(diffmat.shape)
	sqdiffmat = diffmat**2
	# print(sqdiffmat)
	sqdistances = sqdiffmat.sum(axis=1)
	# print(sqdistances)
	distances = sqdistances**0.5
	# print(distances)
	sorteddistindices = distances.argsort()
	# print(sorteddistindices)

	classcount = {}
	for i in range(k):
		votelabel = labels[sorteddistindices[i]]
		# print(votelabel)
		classcount[votelabel] = classcount.get(votelabel, 0) +1
	# print(type(classcount))
	# print(classcount)
	# os.system("pause") 

	sortedclasscount = sorted(classcount.items(), key = operator.itemgetter(1), reverse = True)
	# print(type(sortedclasscount))
	# print(sortedclasscount)
	return sortedclasscount[0][0]

def file2matrix(filename):
	fr = open(filename)
	arrayoflines = fr.readlines()
	numberoflines = len(arrayoflines)
	returnmat = np.zeros((numberoflines, 3))
	classlabelvector=[]
	index = 0
	for line in arrayoflines:
		line = line.strip()
		listfromline = line.split('\t')
		returnmat[index, :] = listfromline[0:3]
		classlabelvector.append(listfromline[-1])
		index +=1
	return returnmat, classlabelvector

def autonorm(dataset):
	minvals = dataset.min(0)
	maxvals = dataset.max(0)
	ranges  = maxvals - minvals
	normdataset = np.zeros(dataset.shape)
	m = dataset.shape[0]
	normdataset = dataset - np.tile(minvals, (m,1))
	normdataset = normdataset/np.tile(ranges, (m, 1))
	# print(normdataset)
	return normdataset, ranges, minvals
	# normalize to 0-1

def datingclasstest():
	horate = 0.1
	datingdatamat, datinglabels = file2matrix('datingTestSet.txt')
	normmat, ranges, minvals=autonorm(datingdatamat)
	m = normmat.shape[0]
	numtestvecs = int(m*horate)
	errorcount = 0.0
	# print(numtestvecs)
	# os.system('pause')
	for i in range(numtestvecs):
		# print(i)
		classifierresult = classify0(normmat[i, :], normmat[numtestvecs:m, :], datinglabels[numtestvecs:m], 3)
		print('the classfier came back iwth: %s, the real answer is: %s' %(classifierresult, datinglabels[i]))
		if classifierresult != datinglabels[i]:
			errorcount += 1.0
	print('the total error rate is: %f' %(errorcount/np.float(numtestvecs)))

	pass

if __name__ == '__main__':
	datingclasstest()

	# datingdatamat, datinglabels = file2matrix('datingTestSet.txt')
	# print(datingdatamat.shape)
	# normmat, ranges, minvals=autonorm(datingdatamat)
	# print(datingdatamat)
	# # os.system('pause')
	# fig = plt.figure()
	# ax = fig.add_subplot(111)
	# ax.scatter(datingdatamat[:,1], datingdatamat[:,2])
	# plt.show()

