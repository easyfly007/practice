import numpy as np
import math

def loaddataset():
	posttingglist = [
		['my', 'dog', 'has','flea','problem', 'help', 'please'],
		['maybe', 'not', 'take', 'him', 'to', 'spark','stupid'],
		['my','dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
		['stop', 'posting', 'stupid', 'worthless', 'grabage'],
		['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to','stop','him'],
		['quit','buying','worthless', 'dog', 'food', 'stupid'],]
	classvec = [0,1,0,1,0,1]
	return posttingglist, classvec

def createvocablist(dataset):
	vocabset = set([])
	for document in dataset:
		vocabset = vocabset | set(document)
	return list(vocabset)

def setofwords2vec(vocablist, inputset):
	returnvec = [0]*len(vocablist)
	for word in inputset:
		if word in vocablist:
			returnvec[vocablist.index(word)] = 1
		else:
			print('the word %s is not in your vocablist' %word)
	return returnvec


def trainNB0(trainingmatrix, traningcategory):
	numtraindocs = len(trainingmatrix)

	#  vocab count
	numwords = len(trainingmatrix[0])
	pabusive = sum(traningcategory) / float(numtraindocs)
	# p0num = np.zeros(numwords)
	p0num = np.ones(numwords)
	# p1num = np.zeros(numwords)
	p1num = np.ones(numwords)
	# p0denom = 0.0
	p0denom = 2.0
	# p1denom = 0.0
	p1denom = 2.0
	for i in range(numtraindocs):
		if traningcategory[i] == 1:
			p1num += trainingmatrix[i]
			p1denom += sum(trainingmatrix[i])
		else:
			p0num += trainingmatrix[i]
			p0denom += sum(trainingmatrix[i])
	p1vect = p1num/p1denom
	p1vect = np.log(p1vect)
	p0vect = p0num/p0denom
	p0vect = np.log(p0vect)
	return p0vect, p1vect, pabusive






def test():
	listofposts, listclasses = loaddataset()
	vocablist = createvocablist(listofposts)
	print(vocablist)
	wordvec = setofwords2vec(vocablist, listofposts[0])
	print(wordvec)

