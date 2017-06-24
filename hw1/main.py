from numpy import exp, array, random, dot, delete, nditer, argmax
from pandas import read_csv
from sklearn.cross_validation import KFold
from argparse import ArgumentParser
import logging

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
    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations, rate):
        for iteration in range(number_of_training_iterations):
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
            for i in reversed(range(len(self.layers) - 1)):
                layer_error = deltas[-1].dot(self.layers[i + 1].synaptic_weights.T)
                layer_delta = layer_error * sigmoid_derivative(outputs[i])
                deltas.append(layer_delta)

            # As we created the list in reverse order, we need to reverse it
            # back to use it properly
            deltas.reverse()

            curr_inputs = training_set_inputs
            for i in range(len(self.layers)):
                self.layers[i].synaptic_weights += rate * curr_inputs.T.dot(deltas[i])
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
        string = """Synaptic Weights for each layer:"""
        for layer in self.layers:
            string += str(layer.synaptic_weights) + "\n"
        return string


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


def fold_train(input, layers, iters, rate):
    folds = 10.0
    kf = KFold(len(input), n_folds=folds, shuffle=True)
    folds_error_sum = 0.0

    for train, test in kf:
        # Combine the layers to create a neural network
        neural_network = NeuralNetwork(layers)

        logging.debug("Stage 1) Random starting synaptic weights: ")
        logging.debug(neural_network.print_weights())

        training_set = input.get_values()[train]
        training_set_inputs = delete(training_set,11,1)
        training_set_outputs = delete(training_set,xrange(11),1)
        training_set_outputs = switch_output_type(training_set_outputs,number_to_array)
        neural_network.train(training_set_inputs, training_set_outputs, iters, rate)

        logging.debug("Stage 2) New synaptic weights after training: ")
        logging.debug(neural_network.print_weights())

        # Test the neural network with a new situation.
        logging.debug("Stage 3) Evaluating a new situation")
        testing_set = input.get_values()[test]
        testing_set_inputs = delete(testing_set,11,1)
        testing_set_outputs = delete(testing_set,xrange(11),1)
        testing_set_outputs = [float(o) for o in nditer(testing_set_outputs)]
        vectored_outputs = neural_network.think(testing_set_inputs)
        outputs = switch_output_type(vectored_outputs[-1],array_to_number)

        diff = outputs - testing_set_outputs
        diff_count = 0.0
        for d in nditer(diff):
            if not float(d) == 0.0:
                diff_count += 1
        logging.debug("Testing set size = " + str(len(testing_set)))
        logging.debug("diff count = " + str(diff_count))
        error = diff_count / len(test)
        folds_error_sum += error

    averaged_error = folds_error_sum / folds
    logging.info("averaged error: " + str(averaged_error))
    return averaged_error


def grid_search(input, labels, iters, rates):
    min_error = 1
    best_rate = 0
    best_iters = 0
    for i in iters:
        min_rate_error = 1
        best_rate_for_iter = 0
        for rate in rates:
            logging.info("iterations = " + str(i) + " rate = " + str(rate))
            rate_error = fold_train(input, labels, i, rate)
            if rate_error < min_rate_error:
                min_rate_error = rate_error
                best_rate_for_iter = rate
        logging.info("best rate for " + str(i) + " iterations is " + str(best_rate_for_iter) + " with error " + str(
            min_rate_error))
        if min_rate_error < min_error:
            min_error = min_rate_error
            best_iters = i
            best_rate = best_rate_for_iter
    logging.info("best error is " + str(min_error) + " for " + str(best_iters) + " iterations and " + str(best_rate) + " rate")


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--hidden-layers", type=int, choices=xrange(2),  default=0)
    parser.add_argument("--wine-type", choices=["red","white"], default="red")
    # parser.add_argument("--learning-iterations", type=int, default=60000)
    parser.add_argument("-d","--debug",action="store_true")

    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        logging.getLogger().setLevel(logging.INFO)

    wine_qual_filename = "hw1/winequality-%s.csv"%args.wine_type
    logging.info("using file=%s"%wine_qual_filename)

    input = read_csv(wine_qual_filename, delimiter=";")

    iters = [600, 6000, 60000]
    rates = [0.4, 1, 1.5, 2, 2.5, 2.7, 3]

    layers = number_to_layers(args.hidden_layers)
    logging.info("Using layers:")
    logging.info(layers)

    grid_search(input, layers, iters, rates)
