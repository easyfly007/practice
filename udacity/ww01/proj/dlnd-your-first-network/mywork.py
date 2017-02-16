import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data_path = 'Bike-Sharing-Dataset/hour.csv'
rides = pd.read_csv(data_path)
# print(rides)
rides[:24*10].plot(x='dteday', y='cnt')

dummy_fields = ['season', 'weathersit', 'mnth', 'hr', 'weekday']
for each in dummy_fields:
	dummies = pd.get_dummies(rides[each], prefix = each, drop_first = False)
	rides = pd.concat([rides, dummies], axis = 1)
fields_to_drop = ['instant', 'dteday','season', 'weathersit', 'weekday', 'atemp', 'mnth', 'workingday', 'hr']
data = rides.drop(fields_to_drop, axis = 1)
# print(data.head())

quant_features = ['casual', 'registered', 'cnt', 'temp', 'hum', 'windspeed']
scaled_features = {}
for each in quant_features:
	mean, std = data[each].mean(), data[each].std()
	scaled_features[each] = [mean, std]
	data.loc[:, each] = (data[each] - mean) / std

test_data = data[-21*24:]
data = data[:-21*24]
target_fields = ['cnt', 'casual', 'registered']
features, targets = data.drop(target_fields, axis =1), data[target_fields]
test_features, test_targets = test_data.drop(target_fields, axis = 1), test_data[target_fields]

train_features, train_targets = features[:-60*24], targets[:-60*24]
val_features, val_targets = features[-60*24:], targets[-60*24:]

class NeuralNetwork(object):
	def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
		self.input_nodes = input_nodes
		self.hidden_nodes = hidden_nodes
		self.output_nodes = output_nodes

		self.weights_input_to_hidden = np.random.normal(0.0, self.hidden_nodes**-0.5, (self.hidden_nodes, self.input_nodes))

		self.weights_hidden_to_output =	np.random.normal(0.0, self.output_nodes**-0.5, (self.output_nodes, self.hidden_nodes))
		self.lr = learning_rate

		self.activation_function = lambda x: 1.0/(1.0+np.exp(-x))


	def train(self, inputs_list, targets_list):
		inputs = np.array(inputs_list, ndmin = 2).T
		targets = np.array(targets_list,ndmin = 2).T

		hidden_inputs = np.dot(self.weights_input_to_hidden, inputs)
		hidden_outputs = self.activation_function(hidden_inputs)

		final_inputs = np.dot(self.weights_hidden_to_output, hidden_outputs)
		final_outputs = final_inputs

		output_error = final_outputs - targets

		hidden_error = np.dot( self.weights_hidden_to_output.T,  output_error) * hidden_outputs *(1- hidden_outputs )
		self.weights_hidden_to_output += - np.dot( hidden_outputs, output_error).T *self.lr  /inputs.shape[1]
		self.weights_input_to_hidden += - np.dot( hidden_error, inputs.T) *self.lr # / inputs.shape[1]

	def run(self, inputs_list):
		inputs = np.array(inputs_list, ndmin = 2).T

		hidden_inputs = np.dot(self.weights_input_to_hidden, inputs)
		hidden_outputs = self.activation_function(hidden_inputs)

		final_inputs = np.dot(self.weights_hidden_to_output, hidden_outputs)
		final_outputs = final_inputs

		return final_outputs


def MSE(y, Y):
	return np.mean((y-Y)**2)


import sys
epochs = 100
learning_rate = 0.1
hidden_nodes = 2
output_nodes = 1

N_i = train_features.shape[1]

network = NeuralNetwork(N_i, hidden_nodes, output_nodes, learning_rate)
losses = {'train':[], 'validation':[]}

for e in range(epochs):
	batch = np.random.choice(train_features.index, size = 128)
	for record, target in zip(train_features.ix[batch].values, train_targets.ix[batch]['cnt']):
		network.train(record, target)

	train_loss = MSE(network.run(train_features), train_targets['cnt'].values)
	val_loss = MSE(network.run(val_features), val_targets['cnt'].values)
	sys.stdout.write("\rProgress: " + str(100*e/float(epochs))[:4] \
		+ "% ... Training loss: " + str(train_loss)[:5] \
		+ " ... Validation loss: " + str(val_loss)[:5])
	losses['train'].append(train_loss)
	losses['validation'].append(val_loss)


plt.plot(losses['train'], label='Training loss')
plt.plot(losses['validation'], label='Validation loss')
plt.legend()
plt.ylim(ymax=0.5)

fig, ax = plt.subplots(figsize=(8,4))

mean, std = scaled_features['cnt']
predictions = network.run(test_features)*std + mean
ax.plot(predictions[0], label='Prediction')
ax.plot((test_targets['cnt']*std + mean).values, label='Data')
ax.set_xlim(right=len(predictions))
ax.legend()

dates = pd.to_datetime(rides.ix[test_data.index]['dteday'])
dates = dates.apply(lambda d: d.strftime('%b %d'))
ax.set_xticks(np.arange(len(dates))[12::24])
_ = ax.set_xticklabels(dates[12::24], rotation=45)




import unittest

inputs = [0.5, -0.2, 0.1]
targets = [0.4]
test_w_i_h = np.array([[0.1, 0.4, -0.3], 
                       [-0.2, 0.5, 0.2]])
test_w_h_o = np.array([[0.3, -0.1]])

class TestMethods(unittest.TestCase):
    
    ##########
    # Unit tests for data loading
    ##########
    
    def test_data_path(self):
        # Test that file path to dataset has been unaltered
        self.assertTrue(data_path.lower() == 'bike-sharing-dataset/hour.csv')
        
    def test_data_loaded(self):
        # Test that data frame loaded
        self.assertTrue(isinstance(rides, pd.DataFrame))
    
    ##########
    # Unit tests for network functionality
    ##########

    def test_activation(self):
        network = NeuralNetwork(3, 2, 1, 0.5)
        # Test that the activation function is a sigmoid
        self.assertTrue(np.all(network.activation_function(0.5) == 1/(1+np.exp(-0.5))))

    def test_train(self):
        # Test that weights are updated correctly on training
        network = NeuralNetwork(3, 2, 1, 0.5)
        network.weights_input_to_hidden = test_w_i_h.copy()
        network.weights_hidden_to_output = test_w_h_o.copy()
        
        network.train(inputs, targets)
        self.assertTrue(np.allclose(network.weights_hidden_to_output, 
                                    np.array([[ 0.37275328, -0.03172939]])))
        self.assertTrue(np.allclose(network.weights_input_to_hidden,
                                    np.array([[ 0.10562014,  0.39775194, -0.29887597],
                                              [-0.20185996,  0.50074398,  0.19962801]])))

    def test_run(self):
        # Test correctness of run method
        network = NeuralNetwork(3, 2, 1, 0.5)
        network.weights_input_to_hidden = test_w_i_h.copy()
        network.weights_hidden_to_output = test_w_h_o.copy()

        self.assertTrue(np.allclose(network.run(inputs), 0.09998924))

suite = unittest.TestLoader().loadTestsFromModule(TestMethods())
unittest.TextTestRunner().run(suite)
