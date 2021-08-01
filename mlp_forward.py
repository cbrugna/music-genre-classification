import numpy as np
from random import random


# save activations and derivatives
# implement backpropagation
# implement gradient descent
# implement a train method that uses both
# train our network with some dummy dataset

class MLP:
    """A class of a MLP for an ANN"""

    def __init__(self, num_inputs=3, hidden_layers=[3, 5], num_outputs=2):
        """ Constructor for the MLP class

        :param num_inputs: int
            number of inputs
        :param hidden_layers: list
            number of hidden layers stored in list, length is number of hidden layers
        :param num_outputs: int
            number of outputs
        """
        self.num_inputs = num_inputs
        self.hidden_layers = hidden_layers
        self.num_outputs = num_outputs

        # layer is a list where each input is number of nuerons in a layer
        layers = [num_inputs] + hidden_layers + [num_outputs]

        # initiate random weights
        weights = []
        for i in range(len(layers) - 1):
            # creates random arrays that can have a matrix with rand(rows, columns)
            # rows is #nuerons in i-th layer, column is #nuerons in i-th+1 layer
            # ex. 2->3 would be (2,3) b/c we need |w11 w12 w13|
            # (w23 is weight from nueron 2 -> 3   |w21 w22 w23|
            w = np.random.rand(layers[i], layers[i + 1])
            weights.append(w)
        self.weights = weights

        activations = []
        for i in range(len(layers)):
            a = np.zeros(layers[i])
            activations.append(a)
        # instance variable
        self.activations = activations

        derivatives = []
        for i in range(len(layers) - 1):
            # same logic as weights above
            d = np.zeros((layers[i], layers[i + 1]))
            derivatives.append(d)
        # instance variable
        self.derivatives = derivatives

    def forward_propagate(self, inputs):
        """Forward propogation step of ANN

        Loops through weights of weight array and calculates weight.
        Then performs activation function (sigmoid) on each nueron.

        :param inputs: array
            array of inputs (net sum)
        :return activations: array
            activation layer
        """
        activations = inputs
        self.activations[0] = inputs

        for i, w in enumerate(self.weights):
            # calculate net inputs
            net_inputs = np.dot(activations, w)

            # calculate the activations
            activations = self._sigmoid(net_inputs)

            # save activations for backprop
            self.activations[i + 1] = activations

        return activations

    def back_propagate(self, error, verbose=False):

        for i in reversed(range(len(self.derivatives))):
            # get activations for the previous layer
            activations = self.activations[i + 1]

            # apply the sigmoid derivative function and reshape
            delta = error * self._sigmoid_derivative(activations)
            delta_reshaped = delta.reshape(delta.shape[0], -1).T

            # get activations for the current layer and reshape
            current_activations = self.activations[i]
            current_activations_reshaped = current_activations.reshape(current_activations.shape[0], -1)

            # save derivative after applying matrix mult.
            self.derivatives[i] = np.dot(current_activations_reshaped, delta_reshaped)

            # back prop the next error
            error = np.dot(delta, self.weights[i].T)

            if verbose:
                print("Derivatives for W{}: {}".format(i, self.derivatives[i]))

        return error

    def gradient_descent(self, learning_rate):
        for i in range(len(self.weights)):
            weights = self.weights[i]
            derivatives = self.derivatives[i]
            weights += derivatives * learning_rate

    def train(self, inputs, targets, epochs, learning_rate):
        """ Train the model and run forward and backward pro

        :param inputs: ndarray
            X
        :param targets: ndarray
            Y
        :param epochs: int
            Number of epochs we want to train network with
        :param learning_rate: int
            Step to apply gradient descent
        """
        for i in range(epochs):
            sum_error = 0

            # iterate through all the training data
            for j, input in enumerate(inputs):
                target = targets[j]

                # forward propagation
                output = self.forward_propagate(input)

                # calculate the error
                error = target - output

                # back propagation
                self.back_propagate(error)

                # apply gradient descent
                self.gradient_descent(learning_rate)

                sum_error += self._mse(target, output)

            # report error for each error - are we improving?
            print("Error: {} at epoch {}".format((sum_error / len(inputs)), i))

    def _mse(self, target, output):
        return np.average((target - output)**2)

    def _sigmoid_derivative(self, x):
        return x * (1.0 - x)

    def _sigmoid(self, x):
        """

        :param x: int
            input for sigmoid function
        :return: result
            result of performing sigmoid on input
        """
        return 1.0 / (1 + np.exp(-x))


if __name__ == "__main__":
    # create a dataset to train a network for the sum operation
    items = np.array([[random() / 2 for _ in range(2)] for _ in range(1000)])
    targets = np.array([[i[0] + i[1]] for i in items])

    # create a Multilayer Perceptron with one hidden layer
    mlp = MLP(2, [5], 1)

    # train network
    mlp.train(items, targets, 50, 0.1)

    # create dummy data
    input = np.array([0.3, 0.1])
    target = np.array([0.4])

    # get a prediction
    output = mlp.forward_propagate(input)

    print()
    print("Our network believes that {} + {} is equal to {}".format(input[0], input[1], output[0]))
    print("Nice!")






