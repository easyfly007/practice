import numpy as np
import os
import matplotlib.pyplot as plt
import random

def loaddataset():
	datamat = []
	labelmat = []
	f = open('testset.txt')
	for line in f.readlines():
		linearr = line.strip().split()
		dataitem = [1.0, float(linearr[0]), float(linearr[1])]
		datamat.append(dataitem)
		labelmat.append(int(linearr[2]))
	return datamat, labelmat

def sigmoid(x):
	return 1.0/(1+np.exp(-x))

def gradascent(datalist, labellist):
	datamat  = np.mat(datalist)
	labelmat = np.mat(labellist).transpose()
	m,n = np.shape(datamat)
	# print(np.shape(datamat))
	# print(datamat)
	# print(labelmat)
	# m is the sample count
	# n is the dimention for one sample, including w and b
	alpha = 0.001
	# learning rate
	epoch = 500
	weights = np.ones((n,1))
	for k in range(epoch):
		h = sigmoid(datamat*weights)
		# print(h.shape)
		# print(labelmat.shape)
		error = labelmat - h
		# print(error)
		weights = weights + alpha*datamat.transpose()*error
	return weights

def stochasticgradientascent0(datamat, classlabels):
	datamat = np.mat(datamat)

	m,n = datamat.shape
	alpha = 0.001
	weights = np.ones((n,1))
	epoch = 500
	for k in range(epoch):
		for i in range(m):
			# print(datamat[i].shape)
			# os.system('pause')
			h = sigmoid(sum(datamat[i]*weights))
			error = classlabels[i] -h
			weights += (alpha*error*datamat[i]).transpose()
	return weights

def stochasticgradientascent1(datamat, classlabels):
	datamat = np.mat(datamat)
	m,n = datamat.shape
	weights = np.ones((n,1))
	dataindex = [x for x in range(m)]
	random.shuffle(dataindex)
	
	numIter = 100
	for j in range(numIter):
		for i in range(m):
			alpha = 4.0/(1.0+j+i)+0.01
			randindex = dataindex[i]
			h = sigmoid(sum(datamat[randindex]*weights))
			error = classlabels[randindex] -h
			weights += (alpha*error*datamat[randindex]).transpose()
	return weights



def plotbestfit(weights):

	datamat, labelmat = loaddataset()
	dataarr = np.array(datamat)
	n = np.shape(dataarr)[0]
	xcord1, ycord1, xcord0, ycord0 = [], [], [], []
	for i in range(n):
		if int(labelmat[i]) ==1:
			xcord1.append(dataarr[i, 1])
			ycord1.append(dataarr[i, 2])
		else:
			xcord0.append(dataarr[i, 1])
			ycord0.append(dataarr[i, 2])
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.scatter(xcord1, ycord1, s=30, c='red', marker = 's')
	ax.scatter(xcord0, ycord0, s=30, c='green')
	x = np.arange(-3.0, 3.0, 0.1)
	y = (-weights[0] -weights[1]*x)/weights[2]
	# y is a matrix
	y = y.transpose()
	ax.plot(x, y)
	plt.xlabel('X1')
	plt.ylabel('X2')
	plt.show()


def classifyvector(x, w):
	prob = sigmoid(sum(x*w))
	if prob > 0.5:
		return 1.0
	else:
		return 0.0

def colictest():
	ftrain = open('horseColicTraining.txt')
	ftest  = open('horseColicTest.txt')
	trainingset = []
	traninglabel = []
	for line in ftrain.readlines():
		currline = line.strip().split('\t')
		linearr = []
		for attr in range(21):
			linearr.append(float(currline[attr]))
		trainingset.append(linearr)
		traninglabel.append(float(currline[21]))
	trainingweights = stochasticgradientascent1(np.array(trainingset), traninglabel)
	errorcount = 0
	numtestvec = 0
	testset = []
	for line in ftest.readlines():
		numtestvec += 1
		currline = line.strip().split('\t')
		linearr = []
		for sttr in range(21):
			linearr.append(float(currline[attr]))
		# testset.append(linearr)
		if int(classifyvector(np.mat(linearr), np.mat(trainingweights))) != int(currline[21]):
			errorcount += 1
	errorrate = float(errorcount) / numtestvec
	print('the error rate for test is: %f' %errorrate)
	return errorrate

def test():
	colictest()
	# datalist, labellist = loaddataset()
	# weights = stochasticgradientascent1(np.array(datalist), labellist)
	# plotbestfit(weights)


if __name__ == '__main__':
	test()


