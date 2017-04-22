from numpy import exp, array, random, dot, delete, nditer, argmax
from pandas import read_csv
from sklearn.cross_validation import KFold
from argparse import ArgumentParser


# The Sigmoid function, which describes an S shaped curve.
# We pass the weighted sum of the inputs through this function to
# normalise them between 0 and 1.
def sigmoid(x):
    return 1 / (1 + exp(-x))


# The derivative of the Sigmoid function.
# This is the gradient of the Sigmoid curve.
# It indicates how confident we are about the existing weight.
def sigmoid_derivative(x):
    return x * (1 - x)


class NeuronLayer():
    def __init__(self, number_of_neurons, number_of_inputs_per_neuron):
        self.synaptic_weights = 2 * random.random((number_of_inputs_per_neuron, number_of_neurons)) - 1

    def __str__(self):
        return "| layer: #inputs=%s #neurons=%s |"%self.synaptic_weights.shape

    def __unicode__(self):
        return u"%s"%self.__str__()

    def __repr__(self):
        return self.__str__()


class NeuralNetwork():
    def __init__(self, layers):
        self.layers = layers

    # We train the neural network through a process of trial and error.
    # Adjusting the synaptic weights each time.
    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        for iteration in xrange(number_of_training_iterations):
            # Pass the training set through our neural network
            outputs = self.think(training_set_inputs)

            # Calculate the error for the last layer (The difference between
            # the desired output and the predicted output).
            curr_error = training_set_outputs - outputs[-1]
            curr_delta = curr_error * sigmoid_derivative(outputs[-1])
            deltas = [curr_delta]

            # Calculate the error for the previous layers (By looking at the
            # weights in each layer, we can determine by how much that layer
            # contributed to the error in the previous).
            for i in reversed(xrange(len(self.layers)-1)):
                layer_error = deltas[-1].dot(self.layers[i+1].synaptic_weights.T)
                layer_delta = layer_error * sigmoid_derivative(outputs[i])
                deltas.append(layer_delta)

            # As we created the list in reverse order, we need to reverse it
            # back to use it properly
            deltas.reverse()

            curr_inputs = training_set_inputs
            for i in xrange(len(self.layers)):
                self.layers[i].synaptic_weights += curr_inputs.T.dot(deltas[i])
                curr_inputs = outputs[i]

    # The neural network thinks.
    def think(self, inputs):
        outputs = []
        curr_inputs = inputs
        for layer in self.layers:
            output = sigmoid(dot(curr_inputs, layer.synaptic_weights))
            outputs.append(output)
            curr_inputs = output
        return outputs

    # The neural network prints its weights
    def print_weights(self):
        print "Synaptic Weights for each layer:"
        for layer in self.layers:
            print layer.synaptic_weights


def number_to_layers(num):
    switcher = {
        0: [NeuronLayer(10, 11)],
        1: [NeuronLayer(20, 11),NeuronLayer(10, 20)],
        2: [NeuronLayer(20, 11),NeuronLayer(30, 20),NeuronLayer(10, 30)]
    }
    return switcher[num]


def number_to_array(num):
    num = float(num)
    switcher = {
        1.0: array([1,0,0,0,0,0,0,0,0,0]),
        2.0: array([0,1,0,0,0,0,0,0,0,0]),
        3.0: array([0,0,1,0,0,0,0,0,0,0]),
        4.0: array([0,0,0,1,0,0,0,0,0,0]),
        5.0: array([0,0,0,0,1,0,0,0,0,0]),
        6.0: array([0,0,0,0,0,1,0,0,0,0]),
        7.0: array([0,0,0,0,0,0,1,0,0,0]),
        8.0: array([0,0,0,0,0,0,0,1,0,0]),
        9.0: array([0,0,0,0,0,0,0,0,1,0]),
        10.0: array([0,0,0,0,0,0,0,0,0,1])
    }
    return switcher[num]


def array_to_number(x):
    return float(argmax(x)+1)


def switch_output_type(initial_outputs, func):
    resulting_outputs = []
    for o in initial_outputs:
        resulting_outputs.append(func(o))
    return array(resulting_outputs)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--hidden-layers", type=int, choices=xrange(2),  default=0)
    parser.add_argument("--wine-type", choices=["red","white"], default="red")
    parser.add_argument("--learning-iterations", type=int, default=60000)

    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args()

    wine_qual_filename = "winequality-%s.csv"%args.wine_type
    wine_qual = read_csv(wine_qual_filename,delimiter=";")

    kf = KFold(len(wine_qual), shuffle=True, n_folds=10)

    layers = number_to_layers(args.hidden_layers)
    print "Using layers:"
    print str(layers)

    for train, test in kf:


        # Combine the layers to create a neural network
        neural_network = NeuralNetwork(layers)

        print "Stage 1) Random starting synaptic weights: "
        # neural_network.print_weights()

        training_set = wine_qual.get_values()[train]
        training_set_inputs = delete(training_set,11,1)
        training_set_outputs = delete(training_set,xrange(11),1)
        training_set_outputs = switch_output_type(training_set_outputs,number_to_array)
        neural_network.train(training_set_inputs, training_set_outputs, args.learning_iterations)

        print "Stage 2) New synaptic weights after training: "
        # neural_network.print_weights()

        # Test the neural network with a new situation.
        print "Stage 3) Considering a new situation"
        testing_set = wine_qual.get_values()[test]
        testing_set_inputs = delete(testing_set,11,1)
        testing_set_outputs = delete(testing_set,xrange(11),1)
        testing_set_outputs = [float(o) for o in nditer(testing_set_outputs)]
        vectored_outputs = neural_network.think(testing_set_inputs)
        outputs = switch_output_type(vectored_outputs[-1],array_to_number)
        diff = outputs - testing_set_outputs
        diff_count = 0
        for d in nditer(diff):
            if not float(d) == 0.0:
                diff_count += 1
        print "Testing set size = " + str(len(testing_set))
        print "diff count = " + str(diff_count)
