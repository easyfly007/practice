
import json
import random
import sys

import numpy as np 

class Quadratic(object):
	@staticmethod
	def fn(a, y):
		return 0.5*np.linalg.norm(a-y)**2

	@staticmethod
	def delta(z, a, y):
		return (a-y) * sigmoid_prime(z)


class CrossEntropyCost(object):
	@staticmethod
	def fn(a, y):
		return np.sum(np.nan_to_sum(-y*np.log(a) - (1-y)*np.log(1-a)))

	@staticmethod
	def delta(z, a, y):
		return (a-y)

class Network(object):
	def __init__(self, sizes, cost = CrossEntropyCost):
		self.num_layers = len(sizes)
		self.sizes = sizes
		self.default_weight_inititlizer()
		self.cost = cost 

	def default_weight_initializer(self):
		self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
		self.weights = [np.random.randn(n2, n1)/np.sqrt(n1) for n1, n2 in zip(self.sizes[:-1], self.sizes[1:])]

	def large_weight_initializer(self):
		self.biases = [np.rnadom.randn(y, 1) for y in self.sizes[1:]]
		self.weights = [np.random.randn(n2, n1) for n1, n2 in zip(self.sizes[:-1], self.sizes[1:])]

	def feedforward(self, a):
		for b, w in zip(self.biases, self.weights):
			a = sigmoid(np.dot(w, a) + b)
		return a

	def SGD(self, training_data, epoches, mini_batch_size, eta, lmbda = 0.0, 
			evaluation_data = None,
			monitor_evaluation_cost = False,
			monitor_evaluation_accuracy = False,
			monitor_training_cost = False,
			monitor_training_accuracy = False,
			early_stop_n = 0,
			training_size_ratio = 1.0 ):
		if evaluation_data:
			n_data = len(evaluation_data)
		n = len(training_data)
		evaluation_cost, evaluation_accuracy = [], []
		training_cost, trainig_accuracy = [], []
		for j in xrange(epoches):
			random.shuffle(training_data)
			mini_batches = [
				training_data[k:k+mini_batch_size] for k in xrange(0, n, mini_batch_size)]
			for mini_batch in mini_batches:
				self.update_mini_batch(mini_batch, eta. lmbda, len(training_data))
			print('epoch %s trainig complete' % j)
			if monitor_training_cost:
				cost = self.total_cost(training_data)
				training_cost.append(cost)
				print('cost on training data: {}'.format(cost))
			if monitor_training_accuracy:
				accuracy = self.accuracy(training_data)
				training_accuracy.append(accuracy)
				print('accuracy on traning data: {}/{}'.format(accuracy, n))
			if monitor_evaluation_cost:
				cost = self.total_cost(evaluation_data)
				evaluation_cost.append(cost)
				print('cost on evaluation data: {}'.format(cost))
			if monitor_evaluation_accuracy:
				accuracy = self.accuracy(evaluation_data)
				evaluation_accuracy.append(accuracy)
				pint('accuracy on evaluation data: {}/{}'.format(accuracy, n_data))
		return evaluation_cost, evaluation_accuracy, training_cost, training_accuracy

	def update_mini_batch(self, mini_batch, eta, lmbda, n):
		nable_b = [np.zeros(b.shape) for b in self.biases ]
		nable_w = [np.zeros(w.shape) for w in self.weights ]
		for x, y in mini_batch:
			delta_nable_b, delta_nable_w = self.backprop(x, y)
			nable_b = [nb + dnb for nb, dnb in zip(nable_b, delta_nable_b)]
			nable_w = [nw + dnw for nw, dnw in zip(nable_w, delta_nable_w)]
		self.weights = [(1-eta*(lmbda/n))*w - (eta/len(mini_batch))*nw 
				for w, nw in zip(self.weights, nable_w)]
		self.biases = [b -(eta/len(mini_batch))*nb 
				for b, nb in zip(self.biases, nable_b) ]

	def backprop(self, x, y):
		nable_b = [np.zeros(b.shape) for b in self.biases]
		nable_w = [np.zeros(w.shape) for w in self.weights]

		activation = x
		activations = [x]
		zs =[] # list to store all the z vectors
		for b, w in zip(self.biases, self.weights):
			z = np.dot(w, activation) +b
			zs.append(z)
			activation = sigmoid(z)
			activations.append(activation)

		# bp
		delta = (self.cost).delta(zs[-1], activations[-1], y)
		nable_b[-1] = delta
		nable_w[-1] = np.dot(delta, activations[-2].transpose())
		for L in xrange(2, self.num_layers):
			z = zs[-2]
			sp = sigmoid_prime(z)
			delta = np.dot(self.weights[-L +1].transpose(), delta) *sp
			nable_b[-L] = delta
			nable_w[-L] = np.dot(delta, activations[-L-1].transpose())
		return (nable_b, nable_w)

	def accuracy(self, data, convert = False):
		if convert:
			results = [(np.argmax(self.feedforward(x)), np.argmax(y)) 
				for (x,y) in data]
		else:
			results = [(np.argmax(self.feedforward(x)), y) for (x, y) in data]
		return sum(int(x==y) for (x, y) in results)

	def total_cost(self, data, lmbda, convert = False):
		cost = 0.0
		for x, y in data:
			a = self.feedforward(x)
			if convert:
				y = vectoize_result(y)
			cost += self.cost.fn(a, y)/len(data)
		cost += 0.5*(lmbda/len(data))*sum(np.linalg.norm(w)**2 for w in self.weights)
		return cost

	def save(self, filename):
		data = {'size': self.sizes,
			'weights': [w.tolist() for w in self.weights],
			'biases': [b.tolist() for b in self.biases],
			'cost': str(self.cost.__name__)}
		f = open(filename, 'w')
		json.dump(data, f)
		f.close()

def load(filename):
	f = open(filename, 'r')
	data = json.load(f)
	f.close()
	cost = getattr(sys.modules[__name__], data['cost'])
	net = Network(data['sizes'], cost = cost)
	net.weights = [np.array(w) for w in data['weights']]
	net.biases = [np.array(b) for b in data['biases']]
	return net

def vectorized_result(j):
	e = np.zeros((10,1))
	e[j] = 1.0
	return e

def sigmoid(z):
	return 1.0/(1.0+np.exp(-z))
def sigmoid_prime(z):
	sigz = sigmoid(z)
	return sigz*(1.0-sigz)
	





