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

if __name__ == '__main__':
	dataset = ['a', 'a', 'b']
	shannonentropy = calcshannonenttropy(dataset)
	print(shannonentropy)

		