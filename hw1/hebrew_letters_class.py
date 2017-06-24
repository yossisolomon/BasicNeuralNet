import numpy as np
from numpy import exp, array, random, dot, delete, nditer, argmax
from pandas import read_csv
from sklearn.cross_validation import KFold
import os


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
    # synaptic weights is a num_of_inputs X num_of_neurons matrix
    def __init__(self, number_of_neurons, number_of_inputs_per_neuron):
        self.synaptic_weights = 2 * random.random((number_of_inputs_per_neuron, number_of_neurons)) - 1


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
        print("Synaptic Weights for each layer:")
        for layer in self.layers:
            print(layer.synaptic_weights)


def number_to_array(num):
    num = float(num)
    switcher = {
        1.0: array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        2.0: array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0]),
        3.0: array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0]),
        4.0: array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0]),
        5.0: array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0]),
        6.0: array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0]),
        7.0: array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0]),
        8.0: array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0])
    }
    return switcher[num]


def array_to_number(x):
    return float(argmax(x) + 1)


def switch_output_type(initial_outputs, func):
    resulting_outputs = []
    for o in initial_outputs:
        resulting_outputs.append(func(o))
    return array(resulting_outputs)


def fold_train(input, labels, iters, rate):
    folds = 4
    kf = KFold(len(input), n_folds=folds, shuffle=True)
    folds_error_sum = 0

    for train, test in kf:
        layer1 = NeuronLayer(150, 256)
        layer2 = NeuronLayer(70, 150)
        layer3 = NeuronLayer(10, 70)
        # layer4 = NeuronLayer(10, 15)
        # layer5 = NeuronLayer(1, 3)

        # Combine the layers to create a neural network
        neural_network = NeuralNetwork([layer1, layer2, layer3])  # ,layer2,layer3,layer4,layer5])

        # print("Stage 1) Random starting synaptic weights: ")
        # neural_network.print_weights()

        # The training set is a matrix with a row for every input
        training_set_inputs = input[train]
        training_set_outputs = labels[train]
        training_set_outputs = switch_output_type(training_set_outputs, number_to_array)
        neural_network.train(training_set_inputs, training_set_outputs, iters, rate)

        # print("Stage 2) New synaptic weights after training: ")
        # neural_network.print_weights()

        # Test the neural network with a new situation.
        #  print("Stage 3) Considering a new situation")
        testing_set_inputs = input[test]
        testing_set_outputs = labels[test]
        testing_set_outputs = [float(o) for o in nditer(testing_set_outputs)]
        vectored_outputs = neural_network.think(testing_set_inputs)
        outputs = switch_output_type(vectored_outputs[-1], array_to_number)
        diff = outputs - testing_set_outputs
        # print outputs
        # print testing_set_outputs
        diff_count = 0.0
        for d in nditer(diff):
            # print d
            if not float(d) == 0.0:
                diff_count += 1
                # print("Testing set size = " + str(len(testing_set)))
                # print("diff count = " + str(diff_count))
        error = diff_count / len(test)
        folds_error_sum += error
        # print("error = " + str(error))

    averaged_error = folds_error_sum / folds
    print("averaged error: " + str(averaged_error))
    return averaged_error


def grid_search(input, labels, iters, rates):
    min_error = 1
    best_rate = 0
    best_iters = 0
    for i in iters:
        min_rate_error = 1
        best_rate_for_iter = 0
        for rate in rates:
            print("iterations = " + str(i) + " rate = " + str(rate))
            rate_error = fold_train(input, labels, i, rate)
            if rate_error < min_rate_error:
                min_rate_error = rate_error
                best_rate_for_iter = rate
        print("best rate for " + str(i) + " iterations is " + str(best_rate_for_iter) + " with error " + str(
            min_rate_error))
        if min_rate_error < min_error:
            min_error = min_rate_error
            best_iters = i
            best_rate = best_rate_for_iter
    print("best error is " + str(min_error) + " for " + str(best_iters) + " iterations and " + str(best_rate) + " rate")


def get_letter_vector(root, file):
    letter_file = open(os.path.join(root, file))
    vec = []
    for line in letter_file.readlines():
        l = []
        for c in line.strip():
            if c == '0':
                l.append(0)
            elif c == '1':
                l.append(1)
        if not l == []:
            vec.append(l)
    return np.array(vec)


def letter_to_number(letter):
    if  'aleph' in letter:
            return 1
    elif 'alef' in letter:
        return 1
    elif 'bet' in letter:
        return 2
    elif 'gimmel' in letter:
        return 3
    elif 'gimel' in letter:
        return 3
    elif 'dalet' in letter:
        return 4
    elif 'he' in letter:
        return 5
    elif 'vav' in letter:
        return 6
    elif 'kaf' in letter:
        return 7
    elif 'lamed' in letter:
        return 8


def get_noisy_vector(vec):
    get_random_indices = lambda: [random.randint(0, 255) for x in range(25)]
    a = vec.copy()
    for i in get_random_indices():
        if a[i] == 1:
            a[i] = 0
        if a[i] == 0:
            a[i] = 1
    return a


if __name__ == "__main__":
    iters = [100, 200]  # range(100, 500, 100)
    rates = [0.4, 1, 1.5, 2, 2.5, 2.7, 3]

    vectors = []
    label = []
    for root, dirs, files in os.walk('hebrew_letters'):
        for file in files:
            letter = file.split('.')[0][:-1].lower()
            letter_vector = np.reshape(get_letter_vector(root, file), (16, 16))

            label.append(letter_to_number(letter))
            vectors.append(np.reshape(letter_vector, 256))

            #translate image by 1 in every direction
            label.append(letter_to_number(letter))
            vectors.append(np.reshape(np.roll(letter_vector, 1, axis=0), 256))
            label.append(letter_to_number(letter))
            vectors.append(np.reshape(np.roll(letter_vector, 1, axis=1), 256))
            label.append(letter_to_number(letter))
            vectors.append(np.reshape(np.roll(letter_vector, 15, axis=0), 256))
            label.append(letter_to_number(letter))
            vectors.append(np.reshape(np.roll(letter_vector, 15, axis=1), 256))

            #randomly flip 25 bits
            letter_vector = np.reshape(letter_vector, 256)
            for i in range(4):
                vectors.append(get_noisy_vector(letter_vector))
                label.append(letter_to_number(letter))

    vecs = np.vstack(vectors)
    grid_search(vecs, np.array(label), iters, rates)


# exit(0)
