from collections import Counter
import numpy as np 
from time import time

def pretty_print_review_and_label(i):
    print(labels[i] + "\t:\t" + reviews[i][:80] + "...")

g = open('reviews.txt','r') # What we know!
reviews = list(map(lambda x:x[:-1],g.readlines()))
g.close()

g = open('labels.txt','r') # What we WANT to know!
labels = list(map(lambda x:x[:-1].upper(),g.readlines()))
g.close()

positive_counts = Counter()
negative_counts = Counter()
total_counts = Counter()
for i in range(len(reviews)):
	if labels[i] =='POSITIVE':
		for word in reviews[i].split(' '):
			positive_counts[word] += 1
			total_counts[word] += 1
	else:
		for word in reviews[i].split(' '):
			negative_counts[word] += 1
			total_counts[word] += 1

pos_neg_ratios = Counter()
for term, cnt in list(total_counts.most_common()):
	if cnt > 100:
		pos_neg_ratio = positive_counts[term] / float(negative_counts[term] + 1)
		pos_neg_ratios[term] = pos_neg_ratio

for word, ratio in pos_neg_ratios.most_common():
	if ratio > 1:
		pos_neg_ratios[word] = np.log(ratio)
	else:
		pos_neg_ratios[word] = -np.log( (1/(ratio + 0.01)))

# print(list(pos_neg_ratios.most_common())[::-1])
# print(positive_counts.most_common())

vocab = set(total_counts.keys())
vocabsize = len(vocab)

word2index = {}
for i, word in enumerate(total_counts):
	word2index[word] = i


class NeuralNetwork(object):
	def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
		# set number of nodes in input, hidden and output layers
		np.random.seed(1)
		self.pre_process_data(reviews, labels)

		self.init_network(len(self.review_vocab), hidden_nodes, 1, learning_rate)


	def init_network(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
		self.layer_0 = np.zeros(input_nodes, 1)
		self.input_nodes = input_nodes
		self.hidden_nodes = hidden_nodes
		self.output_nodes = output_nodes

		# initialize weight
		self.weight_input_to_hidden = np.random.normal(0.0, self.hidden_nodes**-0.5, (self.hidden_nodes, self.input_nodes))
		self.weight_hidden_to_output = no.random.normal(0.0, self.output_nodes**-0.5, (self.output_nodes, self.hidden_nodes))
		self.learning_rate = learning_rate

	def pre_process_data(self, reviews, labels):
		review_vocab = set()
		for review in reviews:
			for word in review.split(' '):
				review_vocab.add(word)
		self.review_vocab = list(review_vocab)

		self.review_vocab_size = len(self.review_vocab)

		self.label_vocab = set(list(labels))
		self.label_vocab_size = len(self.label_vocab)

		self.word2index = {}
		for i, word in enumerate(self.review_vocab):
			self.word2index[word] = i

		self.label2index = {}
		for i, label in enumerate(self.label_vocab):
			self.label2index[label] = i 

	def update_input_layer(self, review):
		self.layer_0  *= 0
		for worrd in review.split(' '):
			if word in self.word2index:
				self.layere_0[0][self.word2index[word]] += 1
	
	def get_target_for_label(self, label):
		''' convert a label to 0 or 1
		args:
			labels(string) -- 'positive' or 'negative'
		output:
			'0' or '1'
		'''
		if label == 'POSITIVE':
			return 1
		return 0

	def sigmoid(self, x):
		return 1/(1.0 + np.exp(-x))
	def sigmoid_derive(self, x):
		dx = sigmoid(x)
		return dx*(1.0-dx)

	def train(self, training_reviews, training_labels):
		assert len(training_labels) == len(training_reviews)
		start = time.time()
		for i in range(len(training_reviews)):
			review = training_reviews[i]
			label = training_labels[i]
			
			# FF
			self.update_input_layer(review)
			self.layer_1 = self.sigmoid(np.dot(self.weight_input_to_hidden, self.layer_0))
			self.layer_2 = self.sigmoid(np.dot(self.weight_hidden_to_output, self.layer_1)) 
			self.layer_2_error = layere_2 - self.get_target_for_label(label)
			self.layer_2_delta = self.layer_2_error * self.sigmoid_derive(self.layer_2)

			layer_1_error = np.dot(layer_2_delta, self.weight_hidden_to_output.T)
			layer_1_delta = self.layer_1_error

			self.weight_hidden_to_output -= np.dot(layer_1.T, layer_2_delta)* self.learning_rate
			self.weight_input_to_hidden  -= np.dot(layer_0.T, layer_1_delta)* self.learning_rate
			

		inputs = np.array(inputs_list, ndmin = 2).T 
		targets = np.array(targets_list, ndmin = 2).T

