import numpy as np
import random


class Network(object):
    def __init__(self, layers):
        """random initialisation of biases and weights
        :param list layers: umber of neurons in each layer as a vector
        """
        self.layers = layers
        self.num_layers = len(layers)
        self.biases = [np.random.randn(y, 1) for y in layers[1:]]             # first layer is input layer hence omitted
        self.weights = [np.random.randn(y, x) for x, y in zip(layers[:-1], layers[1:])]
        # zip([1,2,3], [4,5,6]) = [(1,4), (2,5), (3,6)]

    @staticmethod
    def __sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))         # overloaded when z is np.array

    @staticmethod
    def __sigmoid_prime(z):
        """Derivative of the sigmoid function."""
        return Network.__sigmoid(z) * (1 - Network.__sigmoid(z))

    def forwardfeed(self, a):
        """ return output of network given input a """
        for b, w in zip(self.biases, self.weights):
            a = self.__sigmoid(np.dot(w, a) + b)
        # updates like recurrence during mutation of input a
        return a

    def forwardfeedrecur(self, a, depth=0):
        """
        :param a:
        :param depth: self.num_layers-1
        :return:
        """
        # function default argument are evaluated at declaration
        if depth == self.num_layers-1:
            return a
        else:
            return self.forwardfeedrecur( self.__sigmoid(np.dot(self.weights[depth], a) + self.biases[depth]), depth+1 )

    def sgd(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        """ implements Stochastic Gradient Descent """
        n = len(training_data)

        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k + mini_batch_size] for k in
                            range(0, n, mini_batch_size)]              # partition the randomly permuted training_data
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            print('Epoch {0} : {1} / {2}'.format(j, self.precision(test_data), len(test_data))
                  if test_data
                  else 'Epoch {0} complete'.format(j))

    def sgd_mat(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        n = len(training_data)

        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k + mini_batch_size] for k in
                            range(0, n, mini_batch_size)]  # partition the randomly permuted training_data
             ###self.weights * mini_batches + self.biases

            # for mini_batch in mini_batches:
            #     self.update_mini_batch(mini_batch, eta)
            # print('Epoch {0} : {1} / {2}'.format(j, self.precision(test_data), len(test_data))
            #       if test_data
            #       else 'Epoch {0} complete'.format(j))

    def update_mini_batch(self, mini_batch, eta):
        """
        updates weights and biases per sampled mini_batch for convergence
        :param mini_batch:
        :param eta:
        :return:
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.biases = [ b - eta/len(mini_batch) * nb for b, nb in zip(self.biases, nabla_b)]
        self.weights = [ w - eta/len(mini_batch) * nw for w, nw in zip(self.weights, nabla_w)]

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x]  # list to store all the activations, layer by layer
        zs = []  # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = Network.__sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * Network.__sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = Network.__sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        return (nabla_b, nabla_w)

    def precision(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.forwardfeed(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations - y)
