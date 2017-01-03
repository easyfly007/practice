import numpy as np
import functools
import os
import random
import operator
import feedparser

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


def textparse(bigstring):
	import re
	listoftokens = re.split(r'\W*', bigstring)
	return [tok.lower() for tok in listoftokens if len(tok)>2]

def spamtest():
	doclist = []
	classlist = []
	fulltext = []
	for i in range(1,26):
		wordlist = textparse(open('spam/%d.txt' %i).read())
		# os.system('pause')
		doclist.append(wordlist)
		fulltext.extend(wordlist)
		classlist.append(1)
		filename = 'ham/'+str(i)+'.txt'
		wordlist = textparse(open(filename).read())
		doclist.append(wordlist)
		fulltext.extend(wordlist)
		classlist.append(0)
	# os.system('pause')
	vocablist = createvocablist(doclist)

	trainingset = [x for x in range(50)]
	random.shuffle(trainingset)
	testset, trainingset = trainingset[0:10], trainingset[11:]

	# for i in range(10):
	# 	randindex = int(np.random.uniform(0, len(trainingset)))
	# 	print(randindex)
	# 	testset.append(trainingset[randindex])
	# 	del(trainingset[randindex])

	trainingclasses = []
	trainmat = []
	for docindex in trainingset:
		trainmat.append(setofwords2vec(vocablist, doclist[docindex]))
		trainingclasses.append(classlist[docindex])
	p0v, p1v, pspam = trainNB0(np.array(trainmat), np.array(trainingclasses))
	errorcount = 0

	for docindex in testset:
		wordvector = setofwords2vec(vocablist, doclist[docindex])
		if classifyNB(wordvector, p0v, p1v, pspam) != classlist[docindex]:
			errorcount += 1
	print('the total error rate is: ', np.float(errorcount)/len(testset))

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



def calcmostfreq(vocablist, fulltext):
	freqdict = {}
	for token in vocablist:
		freqdict[token] = fulltext.count(token)
	sortedfreq = sorted(freqdict.items(), key = operator.itemgetter(1), reverse = True)
	return sortedfreq[:30]

def localwords(feed1, feed0):
	import feedparser
	doclist = []
	classlist = []
	fulltext = []
	minlen = min(len(feed1['entries'][i]['summary']), len(feed0['entries'][i]['summary']))
	for i in range(minlen):
		wordlist = textparse(feed1['entries'][i]['summary'])
		doclist.append(wordlist)
		fulltext.extend(wordlist)
		classlist.append(1)

		wordlist = textparse(feed0['entries'][i]['summary'])
		doclist.append(wordlist)
		fulltext.extend(wordlist)
		classlist.append(0)
	vocablist = createvocablist(doclist)
	top30words = calcmostfreq(vocablist, fulltext)
	for i in range(20):
		randindex = int(random.uniform(0, len(trainingset)))
		testset.append(trainingset[randindex])
		del(trainingset[randindex])

	trainmat = []
	trainclasses = []
	for docindex in trainingset:
		trainmat.append(bagofwords2vec(vocablist, doclist[docindex]))
		trainclasses.append(classlist[docindex])
	p0v, p1v, pspam =trainNB0(np.array(trainmat), np.array(trainclasses))
	errorcount = 0
	for docindex in testset:
		wordvector = bagofwords2vec(vocablist, doclist[docindex])
		if classifyNB(np.array(wordvector), p0v, p1v, pspam) != classlist[docindex]:
			errorcount += 1
	print('the error rate is: ', float(errorcount)/len(testset))
	return vocablist, p0v, p1v


def spamtest2():
	ny = feedparser.parse('http://newyork.craigslist.org/stp/index.rss')
	sf = feedparser.parse('http://sfbay.craigslist.org/stp/index.rss')
	vocablist, psf, pny = localwords(ny, sf)
	vocablist, psf, pny = localwords(ny, sf)
	
	



if __name__ == '__main__':
	spamtest2()

