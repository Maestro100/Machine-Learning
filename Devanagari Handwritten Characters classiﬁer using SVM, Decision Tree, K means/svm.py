import pandas as pd
import numpy as np
import cvxopt as cv
import scipy as scip
from scipy.spatial.distance import pdist, squareform
import sys
from sklearn.preprocessing import normalize
from numpy import array
from numpy import mean
from numpy import cov
from numpy.linalg import eig

#argument handling
part = sys.argv[1]
train_dataset = pd.read_csv(sys.argv[2],header=None).values
test_dataset = pd.read_csv(sys.argv[3],header=None).values
output = sys.argv[4]
c = (float)(sys.argv[5])
if (part!='a'):
    gamma = (float)(sys.argv[6])
else:
    gamma=1
threshold = 1e-4

#making train and test data
x = train_dataset[:,1:]/255
y= train_dataset[:,0]
y[y==0]=-1
test = test_dataset[:,1:]/255


#helper functions
def svm_fit(x,y,kernel,c,gamma):
    n_samples,n_features = x.shape
    if(kernel=="linear"):
        k = np.zeros((n_samples, n_samples))
        k = np.matmul(x,np.transpose(x))
    else:
        pairwise_sq_dists = squareform(pdist(x, 'sqeuclidean'))
        k = scip.exp(-gamma*pairwise_sq_dists)
    p=cv.matrix(np.outer(y,y) * k)
    q=cv.matrix(np.ones(n_samples)*-1)
    a=cv.matrix(y.reshape(1, -1))
    a=cv.matrix(a, (1, n_samples), 'd')
    b=cv.matrix(0.0)
    g=cv.matrix(np.vstack((np.diag(np.ones(n_samples) * -1), np.identity(n_samples))))
    h=cv.matrix(np.hstack((np.zeros(n_samples), np.ones(n_samples) * c)))
    cv.solvers.options['show_progress'] = False
    solution = cv.solvers.qp(p, q, g, h, a, b)
    alphas = np.ravel(solution['x'])
    return alphas
def prediction(x,y,test,kernel):
    alphas = svm_fit(x,y,kernel,c,gamma)
    f = np.multiply(alphas,y)
    i_d = np.where(alphas>threshold)[0][0]
    if(kernel=="linear"):
        bias = y[i_d] - np.matmul(np.matmul(x[i_d],np.transpose(x)),np.transpose(f))
        t = np.matmul(np.matmul(test,np.transpose(x)),f)
        sg = np.sign(t+bias)
        sg[sg<0]=0
        return sg
    else:
        asd = scip.exp(-gamma*np.linalg.norm(x-x[i_d],axis=1))
        bias = y[i_d] - np.matmul(asd,np.transpose(f))
        result = list()
        for i in range(len(test)):
            asv = scip.exp(-gamma*np.linalg.norm(x-test[i],axis=1))
            t = np.matmul(asv,np.transpose(f))
            sg = np.sign(t+bias)
            if(sg<0):
                result.append(0)
            else:
                result.append(1)
        return result
        
def checker(ans):
    ttt=test_dataset[:,0]
    ttt[ttt<0]=0
    cor=0
    wro=0
    for i in range(len(ans)):
        if(ans[i]==ttt[i]):
            cor+=1
        else:
            wro+=1
    return cor/(cor+wro)

def pca(A,test):
    M = mean(A.T, axis=1)
    C = A - M
    V = cov(C.T)
    values, vectors = eig(V)
    idx = values.argsort()[::-1]
    values = values[idx]
    vectors = vectors[:,idx]
    values = values[0:50]
    vectors = vectors[:,0:50]
    P_train = vectors.T.dot(A.T)
    m = mean(test.T,axis=1)
    c = test - m
    P_test = vectors.T.dot(test.T)
    return P_train.T,P_test.T

#part a,b,c
if(part=='a'):
    kernel = "linear"
    predict = prediction(x,y,test,kernel)
    np.savetxt(output,predict)
elif(part=='b'):
    kernel = "rbf"
    predict = prediction(x,y,test,kernel)
    np.savetxt(output,predict)
else:
    kernel = "rbf"
    pp,tt = pca(x,test)
    predict = prediction(pp,y,tt,kernel)
    np.savetxt(output,predict)
'''
test_y = test_dataset[:,0]
cor=0
wro=0
for i in range(len(predict)):
    if(test_y[i]==predict[i]):
        cor+=1
    else:
        wro+=1
print(cor)
print(wro)
print(cor/(cor+wro))
'''

# from sklearn.decomposition import RandomizedPCA
# from sklearn.decomposition import PCA
# import matplotlib.pyplot as plt

# pca = PCA(100)
# pca.fit(x_train[0:100].data)

# fig, axes = plt.subplots(2, 4, figsize=(8, 8),
#                          subplot_kw={'xticks':[], 'yticks':[]},
#                          gridspec_kw=dict(hspace=0.1, wspace=0.1))

# for i, ax in enumerate(axes.flat):
#     ax.imshow(pca.components_[i].reshape(32, 32), cmap='gray')

#-----------------------------------------------------------------------

# from pylab import *

# fig, axes = plt.subplots(2, 4, figsize=(8, 8),
#                          subplot_kw={'xticks':[], 'yticks':[]},
#                          gridspec_kw=dict(hspace=0.1, wspace=0.1))
# for i, ax in enumerate(axes.flat):
#     ax.imshow(x_train[i,:].reshape((32,32)), cmap='gray')