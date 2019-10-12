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
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
ps = PorterStemmer()
nltk.download('stopwords')
nltk.download('punkt')
start = time.time()
#part = sys.argv[1]
train = sys.argv[1]
test = sys.argv[2]

words = {}
col_num=0
stop_words = set(stopwords.words('english'))

#print("starts")

with open(train, newline='') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
         example_sent = row[1]
         word_tokens = word_tokenize(example_sent)
         for i in word_tokens:
              if i not in stop_words:
                  if ps.stem(i) not in words:
                      words[ps.stem(i)]=col_num
                      col_num+=1
m = len(words)
voc = np.zeros((5,m))
#print(col_num)
#print(words)
#print("one traversal done")
with open(train, newline='') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        cl = int(float(row[0]))-1
        example_sent = row[1]
        word_tokens = word_tokenize(example_sent)
        for i in word_tokens:
            if i not in stop_words:
                voc[cl][words[ps.stem(i)]]+=1
total = voc.sum(axis=1)
total_sum = total.sum(axis=0)
#print("2 traversals done")
def f(x, y):
    word_tokens = word_tokenize(x)
    res=math.log10(total[y]/total_sum)
    for i in word_tokens:
        if ps.stem(i) in words:
            prob = (voc[y,words[ps.stem(i)]]+1)/(total[y]+m)
            res+=math.log10(prob)
    return res

output=list()
xx=0
with open(test,newline='') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        #print(xx)
        xx+=1
        test_words = row[1]
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
