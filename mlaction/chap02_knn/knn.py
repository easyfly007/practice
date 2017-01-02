import numpy as np
import operator
import os


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
	print(type(classcount))
	print(classcount)
	# os.system("pause") 

	sortedclasscount = sorted(classcount.items(), key = operator.itemgetter(1), reverse = True)
	print(type(sortedclasscount))
	print(sortedclasscount)
	return sortedclasscount[0][0]

def file2matrix(filename):
	fr = open(filename)
	arrayoflines = fr.readlines()
	numberoflines = len(arrayoflines)
	returnmat = numpy.zeros(numberoflines, 3)
	classlabelvector=[]
	index = 0
	for line in arrayoflines:
		line = line.strip()
		listfromline = lines.split('\t')
		returnmat[index, :] = listfromline[0:3]
		classlabelvector.append(int(listfromline[-1]))
		index +=1
	return returnmat, classlabelvector





if __name__ == '__main__':
	group, labels = createdataset()
	sortedclasscount = classify0([0,0], group, labels, 3)
	print(sortedclasscount)

