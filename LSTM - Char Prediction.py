
# coding: utf-8

# In[1]:

import numpy
import sys
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
# Don't know what is this used for
from keras.callbacks import ModelCheckpoint
# Don't know what is this used for
from keras.utils import np_utils


# In[2]:

# load ascii data and convert it to lower case
filename = "data/wonderland.txt"
with open(filename) as wonderland:
    raw_text = wonderland.read()
raw_text = raw_text.lower()


# In[3]:

# mapping of unique chars to integers
chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))


# In[4]:

n_chars = len(raw_text)
n_vocab = len(chars)
print("Total Characters: ", n_chars)
print("Total Vocab", n_vocab)


# In[5]:

seq_length = 100
print(n_chars - seq_length)
dataX = []
dataY = []
for i in range(0, n_chars - seq_length, 1):
    seq_in = raw_text[i:i + seq_length]
    seq_out = raw_text[i + seq_length]
    dataX.append([char_to_int[char] for char in seq_in])
    dataY.append(char_to_int[seq_out])
n_patterns = len(dataX)
print("Total Patterns", n_patterns)


# In[6]:

X = numpy.reshape(dataX, (n_patterns, seq_length, 1))


# In[7]:

X = X / float(n_vocab)


# In[8]:

y = np_utils.to_categorical(dataY)
y


# In[9]:

y.shape


# In[ ]:

# define LSTM Model
model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')
# define the checkpoint
filepath="weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor="loss", verbose=1, save_best_only=True, mode="min")
callbacks_list = [checkpoint]
# fit the model
model.fit(X, y, epochs=50, batch_size=64, callbacks=callbacks_list)


# In[ ]:

# Reverse mapping of integer to characters
int_to_char = dict((i, c) for i, c in enumerate(chars))


# In[ ]:

# pick a random seed
start = numpy.random.randint(0, len(dataX)-1)
pattern = dataX[start]
print("Seed:")
print("\"", ''.join([int_to_char[value] for value in pattern]), "\"")
# generate characters
for i in range(1000):
    x = numpy.reshape(pattern, (1, len(pattern), 1))
    x = x / float(n_vocab)
    prediction = model.predict(x, verbose=0)
    index = numpy.argmax(prediction)
    result = int_to_char[index]
    seq_in = [int_to_char[value] for value in pattern]
    sys.stdout.write(result)
    pattern.append(index)
    pattern = pattern[1:len(pattern)]
print("\nDone.")


# In[ ]:

# from keras.models import load_model
# models = load_model("weights-improvement-19-1.9663.hdf5")


# In[ ]:

# model.load_weights("weights-improvement-19-1.9663.hdf5")


# In[ ]:



