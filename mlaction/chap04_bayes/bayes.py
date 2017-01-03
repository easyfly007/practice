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


if __name__ == '__main__':
	listofposts, listclasses = loaddataset()
	vocablist = createvocablist(listofposts)
	print(vocablist)
	wordvec = setofwords2vec(vocablist, listofposts[0])
	print(wordvec)