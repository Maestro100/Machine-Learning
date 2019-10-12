import pandas as pd
import numpy as np
import sys
import scipy
import keras
from keras.utils import to_categorical
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU


#-------------------------data processing-------------------------------------
x_train = pd.read_csv(sys.argv[1], header = None).values
x_test = pd.read_csv(sys.argv[2], header = None).values
y_train = np.asarray(list(x_train[:,0]))
x_train = x_train[:,1:]
x_test = x_test[:,1:]
train_data = list()
test_data = list()
for i in range(len(x_train)):
    train_data.append(x_train[i].reshape(32,32))
for i in range(len(x_test)):
    test_data.append(x_test[i].reshape(32, 32))
train_data = np.asarray(train_data)
test_data = np.asarray(test_data)
classes = np.unique(y_train)
nClasses = len(y_train)
train_data = train_data.reshape(-1,32,32,1)
test_data = test_data.reshape(-1,32,32,1)
train_data = train_data.astype('float32')
test_data = test_data.astype('float32')
train_data = train_data / 255.
test_data = test_data / 255.
y_train_one_hot = to_categorical(y_train)

#----------------------modelling the data-------------------------------------------
batchsize = 100
epochs = 21
num_classes = 46

#--------------------------architecture-------------------------------
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),input_shape=(32,32,1),padding='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D((2, 2),padding='same'))
model.add(Conv2D(64, (3, 3),padding='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
model.add(Conv2D(128, (3, 3),padding='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
model.add(Flatten())
model.add(Dense(256, activation='tanh'))
model.add(Dense(128,activation='tanh'))
#model.add(LeakyReLU(alpha=0.1))
model.add(Dense(num_classes, activation='softmax'))

#----------------------------compiling the model---------------------------------
model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])
#model.summary()


#----------------------------------training the model------------------------------
model_train = model.fit(train_data,y_train_one_hot,batch_size=batchsize,epochs=epochs)

predictions = model.predict_classes(test_data,batch_size=10)
np.savetxt(sys.argv[3],predictions)