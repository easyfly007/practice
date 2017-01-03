import numpy as np
import functools

def loaddataset():
	posttingglist = [
		['my', 'dog', 'has','flea','problem', 'help', 'please'],
		['maybe', 'not', 'take', 'him', 'to', 'spark','stupid'],
		['my','dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
		['stop', 'posting', 'stupid', 'worthless', 'garbage'],
		['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to','stop','him'],
		['quit','buying','worthless', 'dog', 'food', 'stupid'],]
	classvec = [0,1,0,1,0,1]
	return posttingglist, classvec

def createvocablist(dataset):
	vocabset = set([])
	for document in dataset:
		vocabset = vocabset | set(document)
	# return list(vocabset)
	return list(functools.reduce(lambda x,y:set(x)|set(y), dataset))


def setofwords2vec(vocablist, inputset):
	returnvec = [0]*len(vocablist)
	for word in inputset:
		if word in vocablist:
			returnvec[vocablist.index(word)] = 1
		else:
			print('the word %s is not in your vocablist' %word)
	return returnvec

def bagofwords2vec(vocablist, inputset):
	returnvec = [0]*len(vocablist)
	for word in inputset:
		if word in vocablist:
			returnvec[vocablist.index(word)] +=1
		else:
			print('the word %s is not in your vocablist' %word)
	return returnvec

def trainNB0(trainingmatrix, traningcategory):
	numtraindocs = len(trainingmatrix)

	#  vocab count
	numwords = len(trainingmatrix[0])
	pabusive = sum(traningcategory) / float(numtraindocs)
	# pabusive is based on the abused doc count / total doc count
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


def classifyNB(vec2classify, p0vec, p1vec, pclass1):
	p1 = sum(vec2classify*p1vec) + np.log(pclass1)
	p0 = sum(vec2classify*p0vec) + np.log(1-pclass1)
	# print(p1)
	# print(p0)
	if p1 > p0:
		return 1
	else:
		return 0





def testNB():
	listofposts, listclasses = loaddataset()
	myvocablist = createvocablist(listofposts)
	trainmatrix = []
	for postindoc in listofposts:
		trainmatrix.append(setofwords2vec(myvocablist, postindoc))
	p0v, p1v, pab = trainNB0(np.array(trainmatrix), np.array(listclasses))
	testentry = ['love', 'my', 'dalmation']
	thisdoc = np.array(setofwords2vec(myvocablist,testentry))
	print(testentry, ' classified as: ', classifyNB(thisdoc, p0v, p1v, pab))
	testentry = ['stupid', 'garbage']
	thisdoc = np.array(setofwords2vec(myvocablist, testentry))
	print(testentry, ' classified as: ', classifyNB(thisdoc, p0v, p1v, pab))

if __name__ == '__main__':
	testNB()

