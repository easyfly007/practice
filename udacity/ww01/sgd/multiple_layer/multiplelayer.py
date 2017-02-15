import numpy as np

def sigmoid(x):
    """
    Calculate sigmoid
    """
    return 1/(1+np.exp(-x))

# Network size
N_input = 4
N_hidden = 3
N_output = 2

np.random.seed(42)
# Make some fake data
X = np.random.randn(4)
weights_input_to_hidden = np.random.normal(0, scale=0.1, size=(N_input, N_hidden))
weights_hidden_to_output = np.random.normal(0, scale=0.1, size=(N_hidden, N_output))
# weights_input_to_hidden = np.mat(weights_input_to_hidden)
# weights_hidden_to_output = np.mat(weights_hidden_to_output)

X=np.mat(X).T
weights_input_to_hidden = weights_input_to_hidden.T
# TODO: Make a forward pass through the network
# hidden_layer_in = np.dot(X, weights_input_to_hidden)
hidden_layer_in = weights_input_to_hidden*X

hidden_layer_out = sigmoid(hidden_layer_in)

print('Hidden-layer Output:')
print(hidden_layer_out)

# print(hidden_layer_out.shape)
output_layer_in = weights_hidden_to_output.T*hidden_layer_out
output_layer_out = sigmoid(output_layer_in)

print('Output-layer Output:')
print(output_layer_out)