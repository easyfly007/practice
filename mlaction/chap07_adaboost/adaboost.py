import numpy as np
def loadsimpdata():
	datamat = np.matrix([[1.0, 2.1],
		[2.0, 1.1],
		[1.3, 1.0],
		[1.0, 1.0],
		[2.0, 1.0]])
	classlabels = [1.0, 1.0, -1.0, -1.0, 1.0]
	return datamat, classlabels

