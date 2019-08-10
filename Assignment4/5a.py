import numpy as np
import pandas as pd
import random
import time as t
from scipy import stats

import os
import sys

start_initial = t.time()
x_train_path = sys.argv[1]
train_path = os.path.abspath(x_train_path)
path_train = os.path.dirname(train_path)
os.chdir(path_train)

x_test_path = sys.argv[2]
test_path = os.path.abspath(x_test_path)
path_test = os.path.dirname(test_path)
os.chdir(path_test)

x_train = np.load(x_train_path)
x_test = np.load(x_test_path)

y_train = x_train[:,0].reshape((x_train.shape[0],1))
x_train = x_train[:,1:]

label = np.full((x_train.shape[0], 1), -1)

col_mean = x_train.mean(axis=0)
x_train = x_train[:,(col_mean != 0)]
x_test = x_test[:,(col_mean != 0)]

col_mean = x_train.mean(axis=0)
x_train = x_train - col_mean
col_std = x_train.std(axis=0)
for i in range(x_train.shape[1]):
    if(col_std[i] != 0):
        x_train[:,i] = x_train[:,i]/(col_std[i])

col_mean = x_test.mean(axis=0)
x_test = x_test - col_mean
col_std = x_test.std(axis=0)
for i in range(x_test.shape[1]):
    if(col_std[i] != 0):
        x_test[:,i] = x_test[:,i]/(col_std[i])


print(x_train.shape)


k = 200

start = t.time()

cluster = []

for i in range(0,x_train.shape[0],round(x_train.shape[0]/k)):
    cluster.append(np.mean(x_train[i:i+round(x_train.shape[0]/k),:], axis = 0))
    
print("Time take to make random initialized cluster - " + str(t.time()-start))


cluster = np.array(cluster)


print(cluster.shape)
print(label.shape)

start = t.time()
steps = 0
old_cluster = cluster*100

while(np.max(np.linalg.norm(old_cluster-cluster)) > k/15):
    d_dict = {}
    for i in range(k):
        d_dict[str(i)] = []
    
    old_cluster = np.copy(cluster)
    
    for i in range(x_train.shape[0]):
        distances = []
        for j in range(k):
            distances.append(np.linalg.norm(x_train[i,:]-cluster[j]))
        a = np.argmin(distances)
        label[i] = a
        d_dict[str(a)].append(x_train[i])
    for i in range(k):
        cluster[i] = np.mean(d_dict[str(i)], axis = 0)
    
#    print("Max shift in any centroid is " + str(np.max(np.linalg.norm(old_cluster-cluster))))
    steps += 1

class_label = np.full((k,1), -1)

for i in range(k):
    class_label[i] = stats.mode(y_train[label==i])[0]



predictions = np.full((x_test.shape[0],1), 0)

for i in range(x_test.shape[0]):
    distances = []
    for j in range(k):
        distances.append(np.linalg.norm(x_test[i,:]-cluster[j]))
    predictions[i] = class_label[np.argmin(distances)]




print("Total time taken to run - " + str(t.time() - start_initial))

x_output = sys.argv[3]
actual_path_out = os.path.abspath(x_output)
path_out = os.path.dirname(actual_path_out)
os.chdir(path_out)

np.savetxt(x_output, predictions)

