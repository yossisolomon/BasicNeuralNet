from numpy import exp, array, random, dot, array_equal
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


def read_pgm(filename):
    pgm = []
    with open(filename) as f:
        line = f.readline()
        split_line = line.split()
        width = int(split_line[0])
        height = int(split_line[1])
        for _ in xrange(width):
            arr = []
            for _ in xrange(height):
                line = f.readline()
                arr.append(int(line))
            pgm.append(array(arr))
    pgm2d = array(pgm)
    assert pgm2d.shape == (width,height)
    return pgm2d


def write_pgm(array, filename):
    lines = ["P5\n","240 240\n"]
    for i in array.flatten():
        lines.append(str(int(i))+"\n")
    with open(filename,'w') as f:
        f.writelines(lines)


def split_array_to_list(input_array, split_size=30, split_sqrt = 8):
    output = []
    for x in xrange(split_sqrt):
        for y in xrange(split_sqrt):
            output.append(input_array[x*split_size:(x+1)*split_size,y*split_size:(y+1)*split_size].flatten())
    assert len(output) == split_sqrt * split_sqrt
    return array(output)


def unsplit_array_from_list(array_list, split_size=30, split_sqrt = 8):
    # for l in array_list:
    arr = []
    for x in xrange(split_sqrt):
        for y in xrange(split_sqrt):
            for z in xrange(split_size):
                arr.extend(array_list[x*split_sqrt+y][z*split_size:z*(split_size+1)])
    return array(arr)


def number_to_layers(num):
    return [NeuronLayer(num, 900),NeuronLayer(900, num)]


def auto_train(input, layers, iters, rate):
    # Combine the layers to create a neural network
    neural_network = NeuralNetwork(layers)

    logging.debug("Stage 1) Random starting synaptic weights: ")
    logging.debug(neural_network.print_weights())
    weights = neural_network.layers[0].synaptic_weights

    neural_network.train(input, input, iters, rate)

    assert array_equal(weights ,neural_network.layers[0].synaptic_weights)

    logging.debug("Stage 2) New synaptic weights after training: ")
    logging.debug(neural_network.print_weights())

    # Test the neural network with a new situation.
    logging.debug("Stage 3) Evaluating a new situation")
    output = array([neural_network.think(i)[-1] for i in input])

    # Stretch to fit actual range
    output *= 255

    assert input.shape == output.shape
    diff = output - input
    error_sum = diff.sum()
    logging.info("total error: " + str(error_sum))

    error_avg = error_sum / (input.shape[0] * input.shape[1])

    logging.info("averaged error: " + str(error_avg))

    return neural_network, error_avg


def grid_search(input, labels, iters, rates):
    min_error = 255
    best_rate = 0
    best_iters = 0
    for i in iters:
        min_rate_error = 255
        best_rate_for_iter = 0
        for rate in rates:
            logging.info("iterations = " + str(i) + " rate = " + str(rate))
            rate_error = auto_train(input, labels, i, rate)[1]
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
    parser.add_argument("--hidden-layer-size", type=int, choices=xrange(900,0),  default=300)
    parser.add_argument("-d","--debug",action="store_true")
    parser.add_argument("--lenna",default="lenna.pgm")

    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        logging.getLogger().setLevel(logging.INFO)

    lenna_filename = args.lenna
    logging.info("using file=%s"%lenna_filename)

    input_array = read_pgm(lenna_filename)

    input_split = split_array_to_list(input_array)

    iters = [600, 6000, 60000]
    rates = [0.4, 1, 1.5, 2, 2.5, 2.7, 3]

    layers = number_to_layers(args.hidden_layer_size)
    logging.info("Using layers:")
    logging.info(layers)

    # grid_search(input_split, layers, iters, rates)
    net, error = auto_train(input_split, layers, iters[0], rates[0])
    new_image_pieces = net.think(input_split)[-1] * 255
    new_image_arr = unsplit_array_from_list(new_image_pieces)
    write_pgm(new_image_arr,"new"+lenna_filename)
