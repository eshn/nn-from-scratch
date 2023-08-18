import numpy as np

class Dense:
    '''
    Base class for dense (fully connected) layer.
    Inputs:
        - n_inputs: Number of inputs to the layer (ie number of neurons from the previous layer)
        - n_neurons: Number of neurons in the layer
        - weight_init: Initialization method for neuron weights, to be passed to Neuron class. Default: Gaussian - standard normal
        - activation: Activation function [pass to Neuron class]
    '''
    def __init__(self, n_inputs=1, n_neurons=5, weight_init='Gaussian', activation=None):
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons
        self.weight_init = weight_init
        self.activation = activation

        self.weights, self.bias = self.initialize_weights(size=(self.n_inputs, self.n_neurons), type=self.weight_init)

    def initialize_weights(self, size, type='Gaussian'):
        if self.weight_init == 'Gaussian':
            weights = np.random.normal(0.0, 1.0, size=size)
            bias = np.random.normal(0.0, 1.0)
        elif self.weight_init == 'Uniform':
            weights = np.ones(size)
            bias = np.array(1)
        
        return weights, bias
        
    def eval_activation(self, x):
        if self.activation is None:
            return x
        elif self.activation == 'relu':
            return np.maximum(x, 0)
        elif self.activation == 'tanh':
            return np.tanh(x)
        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        else:
            return -1
    
    def grad_activation(self, x): # gradients of activation functions
        if self.activation is None:
            return np.ones(x.shape)
        elif self.activation == 'relu':
            return np.heaviside(x, 0) # setting gradient to zero at non-differentiable point
        elif self.activation == 'tanh':
            return np.ones(x.shape) - (np.tanh(x) * np.tanh(x)) # derivative is 1 - tanh^2(x)
        elif self.activation == 'sigmoid':
            return x * (np.ones(x.shape) - x) # derivative is x(1-x)
        else:
            return 0


    def forward(self, x):
        self.x = self.eval_activation(np.dot(self.weights.T, x) + self.bias)
        return self.x
        

if __name__ == '__main__':
    # build a 3 layer network: 
    input_size = 3

    x = np.array([3,1,5])

    input0 = Dense(n_inputs=3, n_neurons=2, weight_init='Gaussian', activation='relu')
    hidden1 = Dense(n_inputs=2, n_neurons=4, weight_init='Gaussian', activation='relu')
    hidden2 = Dense(n_inputs=4, n_neurons=6, weight_init='Gaussian', activation='relu')
    hidden3 = Dense(n_inputs=6, n_neurons=6, weight_init='Gaussian', activation='relu')
    hidden4 = Dense(n_inputs=6, n_neurons=4, weight_init='Gaussian', activation='relu')
    out0 = Dense(n_inputs=4, n_neurons=1, weight_init='Gaussian', activation='sigmoid')


    y = input0.forward(x)
    print(y)
    y = hidden1.forward(y)
    print(y)
    y = hidden2.forward(y)
    print(y)
    y = hidden3.forward(y)
    print(y)
    y = hidden4.forward(y)
    print(y)
    y = out0.forward(y)
    print(y)


