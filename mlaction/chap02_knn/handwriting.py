import numpy as np 
import os
from knn import classify0

def img2vec(filename):
	returnvect = np.zeros((1,1024))
	fr = open(filename)
	for i in range(32):
		linestr = fr.readline()
		for j in range(32):
			returnvect[0, 32*i + j] = int(linestr[j])
	return returnvect


def handwritingclassifier():
	hwlabels = []
	trainingfilelist = os.listdir('trainingDigits')
	m = len(trainingfilelist)
	trainingmat = np.zeros((m, 1024))
	for i in range(m):
		filenamestr = trainingfilelist[i]
		filestr = filenamestr.split('.')[0]
		classnumstr = int(filestr.split('_')[0])
		# print(classnumstr)
		hwlabels.append(classnumstr)
		trainingmat[i,:] = img2vec('trainingDigits/'+filenamestr)

	testfilelist = os.listdir('testDigits')
	errorcount = 0.0
	mtest = len(testfilelist)
	for i in range(mtest):
		filenamestr = testfilelist[i]
		filestr = filenamestr.split('.')[0]
		classnumstr = int(filestr.split('_')[0])
		vectorundertest = img2vec('testDigits/'+filenamestr)
		classifierresult = classify0(vectorundertest, trainingmat, hwlabels,3)
		print('the classifier came back with %d, the real answer is %d' %(classifierresult, classnumstr))
		if classifierresult != classnumstr:
			errorcount +=1
	print('\n the total number of error count is %d' %errorcount)
	print('\n the total error rate is %f' %(errorcount/float(mtest)))


if __name__ == '__main__':
	handwritingclassifier()