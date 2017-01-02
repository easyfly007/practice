from math import log
def calcshannonenttropy(dataset):
	numentries = len(dataset)
	labelcounts ={}
	for featvec in dataset:
		currentlabel = featvec[-1]
		labelcounts[currentlabel] = labelcounts.get(currentlabel, 0)
		labelcounts[currentlabel] += 1
	shannonentropy = 0
	for key in labelcounts:
		prob = float(labelcounts[key]) / numentries
		shannonentropy = shannonentropy -prob*log(prob, 2)
	return shannonentropy

def createdataset():
	dataset = [[1,1,'yes'],
		[1, 1, 'yes'],
		[1, 0, 'yes'],
		[0, 1, 'no'],
		[0, 1, 'no'],]
	labels = ['no surfacing', 'flippers']
	return dataset, labels

def splitdataset(dataset, axis, value):
	retdataset = []
	for featvec in dataset:
		if dataset[axis] == value:
			reducedfeatvec = featvec[:axis]
			reducedfeatvec.extend(featvec[axis+1:])
			retdataset.append(reducedfeatvec)
	return retdataset

def choosebestfeaturetosplit(dataset):
	numfeatures = len(dataset[0])-1
	# -1 because the last is the classfy result
	baseentropy = calcshannonenttropy(dataset)
	bestinfogain = 0.0
	bestfeature = -1
	for i in range(numfeatures):
		featlist = [example[i] for example in dataset]
		uniquevalue=set(featlist)
		newentropy = 0
		for value in uniquevalue:
			subdataset = splitdataset(dataset, i, value)
			prob = len(subdataset)/float(len(dataset))
			newentropy += prob*calcshannonenttropy(subdataset)
		infogain = baseentropy- newentropy
		if bestinfogain < infogain:
			bestinfogain = infogain
			bestfeature = i
	return bestfeature

def majoritycountclass(dataset):
	classcount = {}
	for vote in dataset:
		classcount[vote] = classcount.get(vote, 0)
		classcount[vote] +=1
	sortedclasscount = sorted(classcount.items(), key=operator.itemgetter(1), reverse=True)
	return classcount[0][0]

def createtree(dataset, labels):
	classlist = [example[-1], for example in dataset]
	if classlit.count(classlist[0]) == len(classlist):
		return classlist[0]
	if len(dataset[0]) ==1:
		# not any more features
		return majoritycountclass(classlist)
	bestfeature = choosebestfeaturetosplit(dataset)
	bestfeaturelabel = labels[bestfeature]
	mytree = {bestfeaturelabel:{}}
	del(labels[bestfeature])
	featurevalues = [example[bestfeature] for example in dataset]
	uniquevals = set(featurevalues)
	for value in uniquevals:
		sublabels = labels[:]
		mytree[bestfeaturelabel][value] = createtree(splitdataset(dataset, bestfeature, value), sublabels)
	return mytree
	





if __name__ == '__main__':
	dataset, label  = createdataset()
	shannonentropy = calcshannonenttropy(dataset)
	print(shannonentropy)

		