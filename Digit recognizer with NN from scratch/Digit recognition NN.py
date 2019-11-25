#!/usr/bin/env python
# coding: utf-8

# In[66]:


import numpy as np
import random
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from scipy.special import expit
import sys
import math
import matplotlib.pyplot as plt


# In[67]:


class Layer(object):
    
    def __init__(self, num_units, activation, num_units_in_prev_layer):
        self.num_units = num_units
        self.activation = activation
        self.outputs = np.zeros(num_units)
        self.thetas = np.random.randn(num_units, num_units_in_prev_layer) * 0.001 #to initialize small weights
        self.inputs = None
        self.gradients = None
        
        
    #function to return characteritics of layer
    def __repr__(self):
        return "(Num_Units: %d, Activation_function: %s, Thetas shape: %s)" % (self.num_units, self.activation, self.thetas.shape)
    


# In[68]:


class Neural_Network(object):
    
    def __init__(self, num_inputs, num_hidden_units_list, activation):
        if len(num_hidden_units_list) == 0:
            self.hidden_layer_sizes = num_hidden_units_list
            self.layers = [Layer(784, "sigmoid", num_inputs)]#default 784 pixels in digit dataset 28X28
            self.num_layers = len(num_hidden_units_list) + 1
        else:
            self.hidden_layer_sizes = num_hidden_units_list
            self.num_layers = len(num_hidden_units_list) + 1
            self.layers = [Layer(num_hidden_units_list[0], activation, num_inputs)]

            for i in range(1, len(num_hidden_units_list)):
                layer = Layer(num_hidden_units_list[i], activation, num_hidden_units_list[i - 1])
                self.layers.append(layer)

            layer = Layer(10, "sigmoid", num_hidden_units_list[-1]) #output layer:  10 outputs for 10 digits in the dataset
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
            #print(layer.inputs) print(layer.netj)
            layer.netj = layer.inputs @ (np.matrix(self.layers[i].thetas).T)
            layer.outputs = self.nonlinearity(layer.netj, layer.activation)
            #print(layer.outputs)

    def backward_pass(self, gold):
        
        #for output layer
        out_layer = self.layers[-1]
        gold = np.matrix(gold)
        out_layer.grad_wrt_netj = -np.multiply((gold - out_layer.outputs), self.gnl(out_layer.netj, out_layer.activation))
        out_layer.gradients = (out_layer.grad_wrt_netj.T) @ out_layer.inputs

        #for other layers
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

    def train(self, data, labels, eeta=0.01, batch_size=100, max_iter=1000, threshold=1e-4):
        # pdb.set_trace()
        zip_data = list(zip(data.tolist(), labels.tolist()))
        random.shuffle(zip_data)
        plot_epoch = []
        plot_error = []
        old_error = None
        epochs = 1
        lr = eeta
        while(epochs <= max_iter):
            error = 0
            for i in range(0, len(zip_data), batch_size):
                batch = zip_data[i: i + batch_size]
                x, y = zip(*batch)
                self.forward_pass(np.array(x))
                error += self.error(np.array(y))
                self.backward_pass(np.array(y))
                self.update_thetas(lr)

            error /= (len(zip_data) / batch_size)
            print("Epoch:"+ str(epochs)+ " Error: "+str(error))
            
            plot_epoch.append(epochs)
            plot_error.append(error)

            old_error = error
            epochs += 1
            # random.shuffle(zip_data)
        plt.plot(plot_epoch,plot_error)
        plt.title("training error")
        print("\n")

    def predict(self, inp):
        self.forward_pass(inp)
        out = self.layers[-1].outputs
        return np.array(out.argmax(axis=1)).flatten()
    

    def relu(self, output):
        output[output < 0] = 0
        return output

    def sigmoid(self, output):
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

    def print_gradients(self):
        for layer in self.layers:
            print(layer.gradients)


# In[69]:


def read_data(file):
    x = pd.read_csv(file)
    x = x.values
    y = x[:, 0]
    x = np.delete(x, 0, axis=1)
    return x, y


# In[71]:


def accuracy(y1,y2):
    cor=0
    wro=0
    for i in range(len(y1)):
        if(y1[i]==y2[i]):
            cor+=1
        else:
            wro+=1
    m = (cor)/(cor+wro)
    print(str(m))


# In[70]:


train = 'train.csv'
test = 'test.csv'
#out = sys.argv[4]
batch_size = 1000
lr = 0.01
activation = "sigmoid" # sigmoid or relu
hidden_layers = [32,16] #no of units in each layer


train_x, train_y = read_data(train)
train_x = train_x / 255
test_y = train_y[train_y.shape[0]-10000:]
# train_x = scale(train_x)
lb = LabelBinarizer()
lb.fit([i for i in range(10)]) #since 10 outputs are possible
train_y = lb.transform(train_y)

#original test dataset
#test_x, test_y = read_data(test)

#taking test split from train for validation
test_x = train_x[train_y.shape[0]-10000:]
#test_y = train_y[train_y.shape[0]-10000:]
train_x = train_x[:-10000]
train_y = train_y[:-10000]


# In[76]:


#(self, num_inputs, num_hidden_units_list, activation)
model = Neural_Network(784, hidden_layers, activation)
#earlystopping by max_iter and threshold
model.train(train_x, train_y, eeta=lr, batch_size=batch_size, max_iter=250, threshold=1e-4) 
pred = model.predict(test_x)
accuracy(pred,test_y)


# In[62]:


100 - 94.37
150 - 95.25
200 - 95.49
300 - 95.88


# In[28]:


x,y = read_data('train.csv')


# In[26]:


train_x[]


# In[32]:


pred


# In[30]:


y


# In[45]:


train_y.shape[0]


# In[ ]:




