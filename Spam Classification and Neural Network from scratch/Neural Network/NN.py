import numpy as np
import random
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
from scipy.special import expit
import sys
import math


class Layer(object):
    def __init__(self, num_units, activation, num_units_in_prev_layer):
        self.num_units = num_units
        self.activation = activation
        self.outputs = np.zeros(num_units)
        self.thetas = np.random.randn(num_units, num_units_in_prev_layer) * 0.001
        self.inputs = None
        self.gradients = None

    def __repr__(self):
        return "(Num_Units: %d, Activation_function: %s, Thetas shape: %s)" % (self.num_units, self.activation, self.thetas.shape)


class Neural_Network(object):
    def __init__(self, num_inputs, num_hidden_units_list, activation):
        if len(num_hidden_units_list) == 0:
            self.hidden_layer_sizes = num_hidden_units_list
            self.layers = [Layer(46, "sigmoid", num_inputs)]
            self.num_layers = len(num_hidden_units_list) + 1
        else:
            self.hidden_layer_sizes = num_hidden_units_list
            self.num_layers = len(num_hidden_units_list) + 1
            self.layers = [Layer(num_hidden_units_list[0], activation, num_inputs)]

            for i in range(1, len(num_hidden_units_list)):
                layer = Layer(num_hidden_units_list[i], activation, num_hidden_units_list[i - 1])
                self.layers.append(layer)

            layer = Layer(46, "sigmoid", num_hidden_units_list[-1])
            self.layers.append(layer)

    def forward_pass(self, inp):
        inp = np.matrix(inp)
        self.layers[0].inputs = inp
        self.layers[0].netj = inp @ (np.matrix(self.layers[0].thetas).T)
        self.layers[0].outputs = self.nonlinearity(self.layers[0].netj, self.layers[0].activation)
        # print(self.layers[0].activation)
        for i in range(1, self.num_layers):
            layer = self.layers[i]
            layer.inputs = self.layers[i - 1].outputs
#             print(layer.inputs)
            layer.netj = layer.inputs @ (np.matrix(self.layers[i].thetas).T)
            layer.outputs = self.nonlinearity(layer.netj, layer.activation)

    def backward_pass(self, gold):
        # for output layer
        out_layer = self.layers[-1]
        gold = np.matrix(gold)
        out_layer.grad_wrt_netj = -np.multiply((gold - out_layer.outputs), self.gnl(out_layer.netj, out_layer.activation))
        out_layer.gradients = (out_layer.grad_wrt_netj.T) @ out_layer.inputs

        # for other layers
        for i in range(self.num_layers - 2, -1, -1):
            layer = self.layers[i]
            next_layer = self.layers[i + 1]
            layer.grad_wrt_netj = np.multiply((next_layer.grad_wrt_netj @ next_layer.thetas), self.gnl(layer.netj, layer.activation))
            layer.gradients = (layer.grad_wrt_netj.T) @ layer.inputs

    def nonlinearity(self, output, activation):
        if activation == "sigmoid":
            return self.sigmoid(output)
        if activation == "relu":
            return self.relu(output)

    def gnl(self, netj, activation):
        if activation == "sigmoid":
            oj = self.sigmoid(netj)
            return np.multiply(oj, (1 - oj))
        if activation == "relu":
            temp = np.matrix(netj)
            temp[temp < 0] = 0
            temp[temp >= 0] = 1
            return temp

    def update_thetas(self, eeta, momentum=5):
        for layer in self.layers:
            # print("Updating", np.max(layer.gradients))
            layer.thetas = layer.thetas - eeta * (layer.gradients)

    def error(self, gold):
        out_layer = self.layers[-1]
        err = gold - out_layer.outputs
        err = np.sum(np.square(err)) / len(gold)
        return err

    def train(self, data, labels, eeta=0.01, batch_size=100, max_iter=1000, threshold=1e-4, decay=False):
        # pdb.set_trace()
        zip_data = list(zip(data.tolist(), labels.tolist()))
        random.shuffle(zip_data)
        old_error = None
        epochs = 1
        lr = eeta
        factor = 1
        while(epochs <= max_iter):
            error = 0
            for i in range(0, len(zip_data), batch_size):
                batch = zip_data[i: i + batch_size]
                x, y = zip(*batch)
                self.forward_pass(np.array(x))
                error += self.error(np.array(y))
                self.backward_pass(np.array(y))
                if decay:
                    self.update_thetas(eeta / math.sqrt(factor))
                else:
                    self.update_thetas(lr)

            error /= (len(zip_data) / batch_size)
            if epochs == 1:
                print("\rEpoch: %d, Error: %f" % (epochs, error), end=" ")
            else:
                print("\rEpoch: %d, Error: %f old_error: %f" % (epochs, error, old_error), end=" ")

                if error > old_error:
                    factor += 1

            old_error = error
            epochs += 1
            # random.shuffle(zip_data)

        print("\n")

    def predict(self, inp):
        self.forward_pass(inp)
        out = self.layers[-1].outputs
        return np.array(out.argmax(axis=1)).flatten()
    """ Activation Functions """

    def relu(self, output):
        """[rectified linear unit]
        f(x)=max(x,0)
        """
        output[output < 0] = 0
        return output

    def sigmoid(self, output):
        """
        f(x) = 1 / (1 + exp(x))
        """
        # TODO check for correctness
        return expit(output)

    def __repr__(self):
        representation = ""
        for i, layer in enumerate(self.layers):
            temp = "Layer %d - %s\n" % (i, layer)
            representation += temp
        return representation

    def print_outputs(self):
        for layer in self.layers:
            print(layer.outputs)

    def print_graidents(self):
        for layer in self.layers:
            print(layer.gradients)


def read_data(file):
    x = pd.read_csv(file, header=None)
    x = x.values
    y = x[:, 0]
    x = np.delete(x, 0, axis=1)
    return x, y


part = sys.argv[1]
train = sys.argv[2]
test = sys.argv[3]
out = sys.argv[4]
if part == 'a':
    batch_size = int(sys.argv[5])
    lr = float(sys.argv[6])
    activation = sys.argv[7]
    hidden_layers = list(map(int, sys.argv[8:]))
else:
    batch_size = 100
    lr = 0.01
    activation = 'sigmoid'
    hidden_layers = [100]

train_x, train_y = read_data(train)
train_x = train_x / 255
# train_x = scale(train_x)
lb = LabelBinarizer()
lb.fit([i for i in range(46)])
train_y = lb.transform(train_y)
test_x, test_y = read_data(test)

model = Neural_Network(1024, hidden_layers, activation)
model.train(train_x, train_y, eeta=lr, batch_size=batch_size, max_iter=100, threshold=1e-10, decay=True)
pred = model.predict(test_x)

with open(out, "w") as f:
    for each in pred:
        f.write(str(each) + "\n")