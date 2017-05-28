import time
from collections import namedtuple

import numpy as np 
import tensorflow as tf 

with open('anna.txt', 'r') as f:
	text = f.read()

vocab = set(text)
vocab_to_int = {c:i for i, c in enumerate(vocab)}
int_to_vocab = dict(enumerate(vocab_to_int))
chars = np.array([vocab_to_int[c] for c in text], dtype = np.int32)


def split_data(chars, batch_size, num_steps, split_frac = 0.9):
	slice_size = batch_size*num_steps
	n_batches = int(len(chars)/slice_size)

	x = chars[:n_batches*slice_size]
	y = chars[1:n_batches*slice_size +1]

	x = np.split(x, batch_size)
	y = np.split(y, batch_size)

	x = np.stack(x)
	y = np.stack(y)

	split_idx = int(n_batches*split_frac)
	train_x, train_y = x[:, :split_idx*num_steps], y[:, :split_idx*num_steps]
	val_x, val_y = x[:, split_idx*num_steps:], y[:, split_idx*num_steps:]
	return train_x, train_y, val_x, val_y

def get_batch(arrs, num_steps):
	batch_size, slice_size = arrs[0].shape

	n_batches = int(slice_size/num_steps)
	for b in range(n_batches):
		yield [x[:, b*num_steps:(b+1)*num_steps] for x in arrs]

def build_rnn(num_classes, 
	batch_size = 50, 
	num_steps = 50, 
	lstm_size = 128, 
	num_layers = 2, 
	learning_rate = 0.001, 
	grad_clip = 5, 
	sampling = False):
	if sampling == True:
		batch_size, num_steps = 1, 1
	tf.reset_default_graph()
	inputs = tf.placeholder(tf.int32, [batch_size, num_steps], name = 'inputs')
	targets = tf.placeholder(tf.int32, [batch_size, num_steps], name = 'targets')
	keep_prob = tf.placeholder(tf.float32, name = 'keep_prob')
	print(inputs.shape)

	x_one_hot = tf.one_hot(inputs, num_classes)
	y_one_hot = tf.one_hot(targets, num_classes)

	lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
	drop = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob = keep_prob)
	cell = tf.contrib.rnn.MultiRNNCell([drop]*num_layers)
	initial_state = cell.zero_state(batch_size, tf.float32)

	rnn_inputs = [tf.squeeze(i, squeeze_dims = [1]) for i in tf.split(value = x_one_hot, num_or_size_splits = num_steps, axis = 1)]
	outputs, state = tf.contrib.rnn.static_rnn(cell, rnn_inputs, initial_state = initial_state)
	final_state = state

	seq_output = tf.concat(outputs, axis = 1)

	with tf.variable_scope('softmax'):
		softmax_w = tf.Variable(tf.truncated_normal((lstm_size, num_classes), stddev = 0.1))
		softmax_b = tf.Variable(tf.zeros(num_classes))

	logits = tf.matmul(output, softmax_w) + softmax_b
	preds = tf.nn.softmax(logits, name = 'predictions')

	y_reshaped = tf.reshape(y_one_hot, [-1, num_classes])
	loss = tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = y_reshaped)
	cost = tf.reduce_mean(loss)

	tvars = tf.trainable_variables()
	grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), grad_clip)
	train_op = tf.train.AdamOptimizer(learning_rate)
	optimizer = train_op.apply_gradients(zip(grads, tvars))

	export_nodes = ['inputs', 'targets', 'initial_state', 'final_State', 'keep_prob',
		'cost', 'preds', 'optimizer']
	Graph = namedtuple('Graph', export_nodes)
	local_dict = locals()
	graph = Graph(*[local_dict[each] for each in export_nodes])

	return graph 	

batch_size = 100
num_steps = 100
lstm_size = 512
num_layers = 2
learning_rate = 0.001


epochs = 20
save_every_n = 200
train_x, train_y, val_x, val_y = split_data(chars, batch_size, num_steps)

model = build_rnn(len(vocab),
	batch_size = batch_size,
	num_steps = num_steps,
	learning_rate = learning_rate,
	lstm_size = lstm_size,
	num_layers = num_layers)
saver = tf.train.Saver(max_to_keep = 100)

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	n_batches = int(train_x.shape[1]/num_steps)
	iterations = n_batches *epochs
	for e in range(epochs):
		new_state = sess.run(model.initial_state)
		loss = 0
		for b, (x, y) in enumerate(get_batch([train_x, train_y], num_steps), 1):
			iteration = e*n_batches +b
			start = time.time()
			feed = {
				model.inputs: x,
				model.targets: y,
				model.keep_prob: 0.1,
				model.initial_state: new_state}
			batch_loss, new_state, _ = sess.run([model.cost, model.final_state, model.optimizer],
				feed_dict = feed)
			loss += batch_loss
			end = time.time()
			print('epoch {}/{}'.format(e+1, epochs),
				'iteration {}/{}'.format(iteration, iterations),
				'training loss: {:.4f}'.format(loss/b),
				'{:.4f} sec/batch'.format((end-start)))

			if (iteration %save_every_n == 0) or (iteration == iterations):
				val_loss = []
				new_state = sess.run(model.initial_state)
				for x, y in get_batch([val_x, val_y], num_steps):
					feed = {
						model.inputs: x,
						model.targets: y,
						model.keep_prob: 1.0,
						model.initial_state: new_state}
					batch_loss, new_state = sess.run([model.cost, model.final_state], 
						feed_dict = feed)
					val_loss.append(batch_loss)
				print('validation loss:', np.mean(val_loss), 'saving checkpoint!')
				saver.save(sess, 'checkpoints/anna/i{}_l{}_{:.3f}.ckpt'.format(iteration, lstm_size))

def pick_top_n(preds, vocab_size, top_n = 5):
	p = np.squeeze(preds)
	p[np.argsort(p)[:-top_n]] = 0
	p = p / np.sum(p)
	c = np.random.choice(vocab_size, 1, p = p)[0]
	return c

def sample(checkpoint, n_samples, lstm_size, vocab_size, prime = 'The '):
	prime = 'Far'
	samples = [c for c in prime]
	model = build_rnn(vocab_size, lstm_size = lstm_size, sampling = True)
	saver = tf.train.Saver()
	with tf.Session() as sess:
		saver.restore(sess, checkpoint)
		new_state = sess.run(model.initial_state)
		for c in prime:
			x = np.zeros((1,1))
			x[0, 0] = vocab_to_int[c]
			feed = {
				model.inputs: x,
				model.keep_prob: 1.0,
				model.initial_state: new_state}
			preds, new_state = sess.run([model.preds, model.final_state], feed_dict = feed)
		c = pick_top_n(preds, len(vocab))
		samples.append(int_to_vocab[c])

		for i in range(n_samples):
			x[0,0] = c
			feed = {model.inputs: x, 
				model.keep_prob: 1.0,
				model.initial_state: new_state}
			preds, new_state = sess.run([model.preds, model.final_state], feed_dict = feed)

			c = pick_top_n(preds, len(vocab))
			samples.append(int_to_vocab[c])
	return ''.join(samples)

checkpopint = 'checkpoints/anna/i3560_l512_1.122.ckpt'
samp = sample(checkpoint, 2000, lstm_size, len(vocab), prime = 'Far')
print(samp)

checkpopint = 'checkpoints/anna/i200_l512_2.432.ckpt'
samp = sample(checkpoint, 1000, lstm_size, len(vocab), prime = 'Far')
print(samp)

checkpopint = 'checkpoints/anna/i600_l512_1.750.ckpt'
samp = sample(checkpoint, 1000, lstm_size, len(vocab), prime = 'Far')
print(samp)

checkpopint = 'checkpoints/anna/i1000_l512_1.750.ckpt'
samp = sample(checkpoint, 1000, lstm_size, len(vocab), prime = 'Far')
print(samp)