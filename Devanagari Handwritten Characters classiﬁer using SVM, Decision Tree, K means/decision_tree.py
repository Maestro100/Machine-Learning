import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import sys
import time


#argument handling
part = sys.argv[1]
train = pd.read_csv(sys.argv[2],header=0).values
valid = pd.read_csv(sys.argv[3],header=0).values
test = pd.read_csv(sys.argv[4],header=0).values
output_csv = sys.argv[5]
plot = sys.argv[6]


#label encoding of data
le = LabelEncoder()
train[:,2] = le.fit_transform(train[:,2])
train[:,4]=le.fit_transform(train[:,4])
train[:,6]=le.fit_transform(train[:,6])
train[:,7]=le.fit_transform(train[:,7])
train[:,8]=le.fit_transform(train[:,8])
train[:,9]=le.fit_transform(train[:,9])
train[:,10]=le.fit_transform(train[:,10])
train[:,-1]=le.fit_transform(train[:,-1])

valid[:,2] = le.fit_transform(valid[:,2])
valid[:,4]=le.fit_transform(valid[:,4])
valid[:,6]=le.fit_transform(valid[:,6])
valid[:,7]=le.fit_transform(valid[:,7])
valid[:,8]=le.fit_transform(valid[:,8])
valid[:,9]=le.fit_transform(valid[:,9])
valid[:,10]=le.fit_transform(valid[:,10])
valid[:,-1]=le.fit_transform(valid[:,-1])

test[:,2] = le.fit_transform(test[:,2])
test[:,4]=le.fit_transform(test[:,4])
test[:,6]=le.fit_transform(test[:,6])
test[:,7]=le.fit_transform(test[:,7])
test[:,8]=le.fit_transform(test[:,8])
test[:,9]=le.fit_transform(test[:,9])
test[:,10]=le.fit_transform(test[:,10])
test[:,-1]=le.fit_transform(test[:,-1])


#preparation of train, test and validation data
X_train = train[:,1:]
y_train = train[:,0].astype('int')
X_test = test[:,1:]
X_valid = valid[:,1:]
y_valid = valid[:,0].astype('int')


#helper functions
def prediction(X_test, clf_object):
    y_pred = clf_object.predict(X_test)
    return y_pred
def cal_accuracy(y_test, y_pred):
    print ("Accuracy : ", accuracy_score(y_test,y_pred)*100)
def train_using_entropy_a(X_train, y_train):
    clf_entropy = DecisionTreeClassifier(criterion="entropy",splitter='best',random_state=0,max_depth=None)
    clf_entropy.fit(X_train, y_train)
    return clf_entropy
def train_using_entropy_b(X_train, y_train):
    clf_entropy = DecisionTreeClassifier(criterion="entropy",splitter='best',random_state=0,max_depth=13,min_samples_leaf=34)
    clf_entropy.fit(X_train, y_train)
    return clf_entropy
def func_train_valid(x):
    clf_entropy = DecisionTreeClassifier(criterion="entropy",splitter='best',random_state=0,max_depth=None,max_leaf_nodes=x)
    clf_entropy.fit(X_train, y_train)
    return accuracy_score(y_train,prediction(X_train, clf_entropy))*100, accuracy_score(y_valid,prediction(X_valid, clf_entropy))*100
def h(b):
    l = []
    k = []
    for i in range(len(b)):
        p,q = func_train_valid(b[i])
        l.append(p)
        k.append(q)
    return l,k

print('starting to predict')
#part a and b
if(part=='a'):
    clf_entropy = train_using_entropy_a(X_train,y_train)
    y_pred_entropy = prediction(X_test, clf_entropy)
    np.savetxt(output_csv,y_pred_entropy)
    rr = np.arange(2, 1000, 1)
    y, z = h(rr)
    plt.plot(rr, y, linestyle='-', color='g')
    plt.plot(rr, z, linestyle='-', color='b')
    plt.title('part a')
    plt.xlabel('no. of nodes in decision tree')
    plt.ylabel('accuracy in %')
    plt.legend(['train set', 'validation set'])
    plt.savefig(plot)
else:
    clf_entropy = train_using_entropy_b(X_train, y_train)
    y_pred_entropy = prediction(X_test, clf_entropy)
    np.savetxt(output_csv, y_pred_entropy)
    rr = np.arange(200, 1000, 1)
    y, z = h(rr)
    plt.plot(rr, y, linestyle='-', color='g')
    plt.plot(rr, z, linestyle='-', color='b')
    plt.title('part b')
    plt.xlabel('no. of nodes in decision tree')
    plt.ylabel('accuracy in %')
    plt.legend(['train set', 'validation set'])
    plt.savefig(plot)