import pandas as pd
import numpy as np
import sys

x_train = pd.read_csv(sys.argv[1], header = None).values
#print("train_read")
x_test = pd.read_csv(sys.argv[2], header = None).values
#print("test_read")

y_train = list(x_train[:,0])
y_test = list(x_test[:,0])
x_train[:,0] = 1
x_test[:,0]=1

#print(x_train[0,1])

def square_loss_func(w,x_test,y_test):
    y_tr_out = (np.matmul(x_test,np.transpose(w)))
    for i in range(len(y_tr_out)):
        y_tr_out[i]=np.round(y_tr_out[i],decimals=0)
    return np.linalg.norm(y_test-y_tr_out)**2/np.linalg.norm(y_test)**2


def ridge_reg_closedform_Nfolds(x_train,y_train,k,N):
    l = len(x_train)
    l=l-l%10
    l=l//10
    error=0
    #print(len(x_train))
    for i in range(0,N):
        ab = np.arange(i*l,(i+1)*l)
        x_train_fold = np.delete(x_train,ab,0)
        x_test_fold = x_train[i*l:(i+1)*l]
        y_train_fold = np.delete(y_train,ab,0)
        y_test_fold = y_train[i*l:(i+1)*l]

        a1 = np.matmul(np.transpose(x_train_fold),x_train_fold)
        a2 = np.linalg.inv(a1+k*np.matrix(np.identity(len(a1))))
        a3 = np.matmul(np.transpose(x_train_fold),y_train_fold)
        w = np.matmul(a2,a3)
        #print(len(x_train_fold))
        error+= square_loss_func(w,x_test_fold,y_test_fold)
    error=error/N
    return error

#print(ridge_reg_closedform_Nfolds(x_train,y_train,5,10))

def predict(x_train, y_train, x_test, y_test, l):
    iden = np.identity(91)
    iden[0,0] = 0
    inv1 = np.matmul(np.transpose(x_train),x_train) + l*iden
    inv2 = np.linalg.inv(inv1)
    inv3 = np.matmul(np.transpose(x_train),y_train)
    coeff = np.matmul(inv2,inv3)
    y_out = np.matmul(x_test,coeff)
    return y_out


ans = predict(x_train, y_train, x_test, y_test, 100000)

np.savetxt(sys.argv[3], ans)

