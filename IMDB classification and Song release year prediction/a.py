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

def lin_reg_closedform(x_train,y_train):
    w = np.matmul(np.linalg.inv(np.matmul(np.transpose(x_train),x_train)),np.matmul(np.transpose(x_train),y_train))
    return w

def square_loss_func(w,x_test,y_test):
    y_tr_out = np.matmul(x_test,np.transpose(w))
    for i in range(len(y_tr_out)):
        y_tr_out[i]=np.round(y_tr_out[i],decimals=0)
    return np.linalg.norm(y_test-y_tr_out)/(2*len(y_test))

#print(square_loss_func(lin_reg_closedform(x_train,y_train),x_test,y_test))
def predict(x_test,w):
    output = np.matmul(np.transpose(w),x_test)
    for i in range(len(output)):
        output[i]=np.round(output[i],decimals=0)
    return output

ans = predict(x_test,lin_reg_closedform(x_train,y_train))
ans = np.savetxt(sys.argv[3],ans)