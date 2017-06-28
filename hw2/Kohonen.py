import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import rv_continuous


class Kohonen:
    def __init__(self, shape, dim, max_value):
        num_clusters = np.prod(shape)
        self.dim = dim
        self.max = max_value
        self.weights = np.random.uniform(size=(num_clusters, dim))
        self.indices = np.zeros(shape)
        self.shape = shape
        self._cluster_to_location = []
        self._location_to_cluster = np.reshape(range(num_clusters), shape)
        for i in range(num_clusters):
            self._cluster_to_location.append(np.argwhere(self._location_to_cluster == i)[0])

    def train(self, data, iterations):
        self._iterations = 10*iterations
        self._iteration = 0
        for i in range(10):
            for x in data:
                if self._iteration % 10000 == 0:
                    print(self._iteration)
                    self.print()
                self._train_single(x)
        self.print()
        # np.apply_along_axis(self._train_single, axis = 1, arr=data)

    def _train_single(self, p):
        self._iteration += 1
        bmu = np.argmin(np.linalg.norm(self.weights - p, axis=1))
        neighborhood = self._get_neighborhood(bmu, 5)
        for v in neighborhood:
            self.weights[v] = self.weights[v] + self._space_damp(bmu, v) * self._time_damp() * (p - self.weights[v])

    def _get_neighborhood(self, i, r):
        neighborhood = []
        indices = self._cluster_to_location[i]
        neighborhood.append(i)
        for i in range(len(indices)):
            y = np.zeros(indices.shape)
            for j in range(-r, r):
                y[i] = j
                if ((indices + y) >= 0).all() and ((indices + y) < self.shape[i]).all():
                    neighborhood.append(*self._location_to_cluster[[[i] for i in indices + y]])
        return neighborhood

    def _space_damp(self, x, v):
        sigma_0 = 4
        lam = self._iterations / np.log(sigma_0)
        sigma = sigma_0*np.exp(-1* self._iteration/lam)
        x_ind = self._cluster_to_location[x]
        v_ind = self._cluster_to_location[v]
        dist = np.linalg.norm(x_ind - v_ind)
        return np.exp(-1 * dist / (2 * sigma * sigma))


    def _time_damp(self):
        return 0.1 * np.exp(-1 * self._iteration / self._iterations)

    def print(self):
        x = [i[0] for i in self.weights]
        y = [i[1] for i in self.weights]
        for j in self.shape:
            for i in range(j):
                plt.plot(x[i * j : (i + 1) * j], y[i * j : (i + 1) * j], 'ro-')
            for i in range(j):
                plt.plot(x[i : j*j - (j - i - 1) : j], y[i : j*j - (j - i - 1) : j], 'ro-')

        plt.axis([-self.max, self.max, -self.max, self.max])
        if self.max == 1:
            plt.axis([0, self.max, 0, self.max])
        plt.show()





if __name__ == '__main__':
    NUM_SAMPLES = 3000

    print("Square")

    # UNIFORM SAMPLING
    print("uniform sampling")
    data = np.random.uniform(size=(NUM_SAMPLES, 2))
    plt.plot(data[:, 0], data[:, 1], 'ro')
    plt.axis([0, 1, 0, 1])
    plt.show()

    print("line topology")
    kohonen = Kohonen((30, 1), 2, 1)
    kohonen.train(data, NUM_SAMPLES)

    print("grid topology")
    kohonen = Kohonen((12, 12), 2, 1)
    kohonen.train(data, NUM_SAMPLES)

    #PROPORTIAONLAL TO X
    print("Proportional to x sampling")
    data = np.empty(shape=(NUM_SAMPLES, 2))
    i = 0
    while i < NUM_SAMPLES:
        p = np.random.uniform(size=(1,2))
        if np.random.uniform() <= p[0, 0]:
            data[i, :] = p
            i += 1

    plt.plot(data[:, 0], data[:, 1], 'ro')
    plt.axis([0, 1, 0, 1])
    plt.show()

    print("line topology")
    kohonen = Kohonen((30, 1), 2, 1)
    kohonen.train(data, NUM_SAMPLES)
    print("grid topology")
    kohonen = Kohonen((12, 12), 2, 1)
    kohonen.train(data, NUM_SAMPLES)

   # PROPORTIONAL TO NORM
    print("Proportional to norm sampling")
    data = np.empty(shape=(NUM_SAMPLES, 2))
    i = 0
    while i < NUM_SAMPLES:
        p = np.random.uniform(size=(1, 2))
        if np.random.uniform() <= np.linalg.norm(p):
            data[i, :] = p
            i += 1

    plt.plot(data[:, 0], data[:, 1], 'ro')
    plt.axis([0, 1, 0, 1])
    plt.show()

    print("line topology")
    kohonen = Kohonen((30, 1), 2, 1)
    kohonen.train(data, NUM_SAMPLES)

    print("grid topology")
    kohonen = Kohonen((12, 12), 2, 1)
    kohonen.train(data, NUM_SAMPLES)

    print("Donut")
    # Donut
    # Uniform sampling
    print("uniform sampling")
    data = np.empty(shape=(NUM_SAMPLES, 2))
    i = 0
    while i < NUM_SAMPLES:
        p = np.random.uniform(low=-2, high=2, size=(1, 2))
        #check if p is within the donut
        if 1 <= np.linalg.norm(p) and np.linalg.norm(p) <= 2:
            data[i, :] = p
            i += 1

    plt.plot(data[:, 0], data[:, 1], 'ro')
    plt.axis([-3, 3, -3, 3])
    plt.show()

    print("line topology")
    kohonen = Kohonen((30, 1), 2, 3)
    kohonen.train(data, NUM_SAMPLES)

    print("grid topology")
    kohonen = Kohonen((12, 12), 2, 3)
    kohonen.train(data, NUM_SAMPLES)


    data = np.empty(shape=(NUM_SAMPLES, 2))
    i = 0
    while i < NUM_SAMPLES:
        p = np.random.uniform(low=-2, high=2, size=(1, 2))
        # check if p is within the donut
        if 1 <= np.linalg.norm(p) and np.linalg.norm(p) <= 2:
            if np.random.uniform(low = 0.5, high=1) <= 1/(np.linalg.norm(p)):
                data[i, :] = p
                i += 1

    plt.plot(data[:, 0], data[:, 1], 'ro')
    plt.axis([-3, 3, -3, 3])
    plt.show()

    print("line topology")
    kohonen = Kohonen((30, 1), 2, 3)
    kohonen.train(data, NUM_SAMPLES)

    print("grid topology")
    kohonen = Kohonen((12, 12), 2, 3)
    kohonen.train(data, NUM_SAMPLES)
