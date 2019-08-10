import pandas as pd
import numpy as np
import sys
import re
import collections
import csv
import math
import time
import scipy
import sklearn
#start = time.time()
#part = sys.argv[1]
train = sys.argv[1]
test = sys.argv[2]

words = {}
col_num=0
with open(train, newline='') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
         csv_words = row[1].split(" ")
         for i in csv_words:
              if i not in words:
                  words[i]=col_num
                  col_num+=1
m = len(words)
voc = np.zeros((5,m))
#print(col_num)
with open(train, newline='') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        cl = int(float(row[0]))-1
        csv_words = row[1].split(" ")
        for i in csv_words:
            voc[cl][words[i]]+=1
total = voc.sum(axis=1)
total_sum = total.sum(axis=0)

def f(x, y):
    res=math.log10(total[y]/total_sum)
    for i in x:
        if i in words:
            prob = (voc[y,words[i]]+1)/(total[y]+m)
            res+=math.log10(prob)
    return res

output=list()

with open(test,newline='') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        test_words = row[1].split(" ")
        arr= list()
        arr.append(f(test_words, 0))
        arr.append(f(test_words, 1))
        arr.append(f(test_words, 2))
        arr.append(f(test_words, 3))
        arr.append(f(test_words, 4))
        output.append(int(arr.index(max(arr)))+1)
        arr.clear()

np.savetxt(sys.argv[3],output)

#print(time.time()-start)