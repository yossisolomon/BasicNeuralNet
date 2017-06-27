from argparse import ArgumentParser
from math import sqrt
import logging
from collections import namedtuple
from matplotlib import pyplot
import numpy as np
import re

image_props = namedtuple('image_props', ['type', 'width' ,'height', 'depth'])
# The Sigmoid function, which describes an S shaped curve.
# We pass the weighted sum of the inputs through this function to
# normalise them between 0 and 1.
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# The derivative of the Sigmoid function.
# This is the gradient of the Sigmoid curve.
# It indicates how confident we are about the existing weight.
def sigmoid_derivative(x):
    return x * (1 - x)


class NeuronLayer():
    def __init__(self, number_of_neurons, number_of_inputs_per_neuron):
        self.synaptic_weights = 2 * np.random.random((number_of_inputs_per_neuron, number_of_neurons)) - 1

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
            output = sigmoid(np.dot(curr_inputs, layer.synaptic_weights))
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
        type = line
        while not line[0] in ['0','1','2','3','4','5','6','7','8','9']:
            line = f.readline()
        split_line = line.split()
        width = int(split_line[0])
        height = int(split_line[1])
        depth = f.readline()
        for _ in xrange(width):
            arr = []
            for _ in xrange(height):
                line = f.readline()
                arr.append(int(line))
            pgm.append(np.array(arr))
    pgm2d = np.array(pgm)
    assert pgm2d.shape == (width,height)
    return pgm2d, image_props(type,width, height, depth)


def write_pgm(arr, filename):
    lines = ["P2\n","240 240\n","255\n"]
    for i in arr.flatten():
        lines.append(str(int(i))+"\n")
    with open(filename,'w') as f:
        f.writelines(lines)


def split_array_to_list(input_array, split_size=30, split_sqrt = 8):
    output = np.vsplit(input_array[0:split_size*split_sqrt,0:split_size*split_sqrt],split_sqrt)
    output = [np.hsplit(block,split_sqrt) for block in output]
    return output


def unsplit_array_from_list(array_list):
    blocks = [np.hstack(block) for block in array_list]
    return np.vstack(blocks)



def number_to_layers(num):
    return [NeuronLayer(num, 900),NeuronLayer(900, num)]


def auto_train(input, layers, iters, rate):
    # Combine the layers to create a neural network
    neural_network = NeuralNetwork(layers)

    logging.debug("Stage 1) Random starting synaptic weights: ")
    logging.debug(neural_network.print_weights())
    weights = neural_network.layers[0].synaptic_weights.copy()

    # make blocks
    blocks = []
    for block_list in input:
        for block in block_list:
            block = block.flatten()
            blocks.append(block)
    blocks = np.array(blocks)
    neural_network.train(blocks, blocks, iters, rate)

    assert not np.array_equal(weights ,neural_network.layers[0].synaptic_weights)

    logging.debug("Stage 2) New synaptic weights after training: ")
    logging.debug(neural_network.print_weights())

    # Test the neural network with a new situation.
    logging.debug("Stage 3) Evaluating a new situation")

    output = neural_network.think(blocks)[-1]
    diff = output - blocks
    rms_errors = [sqrt(sum(n*n for n in diff_part)/len(diff_part)) for diff_part in diff]

    logging.info("RMS errors: " + str(rms_errors))

    rms_error_avg = sum(rms_errors) / len(rms_errors)
    logging.info("Averaged RMS errors: " + str(rms_error_avg))

    return neural_network, rms_error_avg


def grid_search(input, labels, iters, rates):
    min_error = 1
    best_rate = 0
    best_iters = 0
    for i in iters:
        min_rate_error = 1
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
    parser.add_argument("--hidden-layer-size", type=int, choices=xrange(900,0),  default=600)
    parser.add_argument("-d","--debug",action="store_true")
    parser.add_argument("--lenna",default="lenna.pgm")
    parser.add_argument("-b","--block-test",action="store_true")

    return parser.parse_args()


def read_pgm(filename, byteorder='>'):
    """Return image data from a raw PGM file as numpy array.

    Format specification: http://netpbm.sourceforge.net/doc/pgm.html

    """
    with open(filename, 'rb') as f:
        buffer = f.read()
    try:
        header, width, height, maxval = re.search(
            b"(^P5\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n]\s)*)", buffer).groups()
    except AttributeError:
        raise ValueError("Not a raw PGM file: '%s'" % filename)
    return np.frombuffer(buffer,
                            dtype='u1' if int(maxval) < 256 else byteorder+'u2',
                            count=int(width)*int(height),
                            offset=len(header)
                            ).reshape((int(height), int(width)))


if __name__ == "__main__":
    image = read_pgm("baboon.pgm", byteorder='<')
    pyplot.imshow(image, pyplot.cm.gray)
    pyplot.show()
    image = image / 256
    layers = number_to_layers(10)

    neural_network = NeuralNetwork(layers)
    blocks = []
    for i in range(256-30):
        for j in range(256 - 30):
            blocks.append(np.reshape(image[i: i + 30, j: j + 30], (900)))
    blocks = np.array(blocks)
    neural_network.train(blocks, blocks, 100, 0.000004)
    result = neural_network.think(blocks)[-1]
    norms = np.linalg.norm(result - blocks, axis=1)
    res = norms.mean()





    # args = parse_args()
    #
    # if args.debug:
    #     logging.getLogger().setLevel(logging.DEBUG)
    # else:
    #     logging.getLogger().setLevel(logging.INFO)
    #
    # lenna_filename = args.lenna
    # logging.info("using file=%s"%lenna_filename)
    #
    # input_array = read_pgm(lenna_filename)[0]
    #
    # #normalize input (scale to [0,1])
    # input_array = input_array / 255.0
    #
    # input_split = split_array_to_list(input_array)
    #
    # #just to see re-assembly of blocks works
    # if args.block_test:
    #     new_image_arr = unsplit_array_from_list(input_split)
    #     new_image_arr *= 255
    #     write_pgm(new_image_arr,"new"+lenna_filename)
    # else:
    #     iters = [600, 6000, 60000]
    #     rates = [0.4, 1, 1.5, 2, 2.5, 2.7, 3]
    #
    #     layers = number_to_layers(args.hidden_layer_size)
    #     logging.info("Using layers:")
    #     logging.info(layers)
    #
    #
    #     # grid_search(input_split, layers, iters, rates)
    #     net, error = auto_train(input_split, layers, iters[1], rates[3])
    #
    #     #predict new lenna
    #     new_image_pieces = []
    #     for block_list in input_split:
    #         new_block_list = []
    #         for block in block_list:
    #             new_block = net.think(block.flatten())[-1]
    #
    #             #rescale image to 0-255
    #             new_block *= 255
    #
    #             new_block.reshape(30,30)
    #
    #             new_block_list.append(new_block)
    #         new_image_pieces.append(new_block_list)
    #
    #     new_image_arr = unsplit_array_from_list(new_image_pieces)
    #     write_pgm(new_image_arr,"new"+lenna_filename)
