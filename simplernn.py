#import tensorflow onehotencoding
from tensorflow.keras.preprocessing.text import one_hot
#import libraries related to  embediing  sequences and sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import SimpleRNN
import numpy as np
from tensorflow.keras.datasets import imdb

max_features = 10000
#load imdb dataset
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

#print shape of data
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

#inspect sample review and its label
print(x_train[0])
print(y_train[0])

#map word to index
word_index = imdb.get_word_index()
print(word_index)

max_len=500
#add padding to train and test data
x_train=pad_sequences(x_train,maxlen=max_len,padding='post')
x_test=pad_sequences(x_test,maxlen=max_len,padding='post')

print(x_train[0])
print(x_test[0])

#build model
model=Sequential()
model.add(Embedding(max_features,128,input_length=max_len))
model.add(SimpleRNN(128,activation='relu'))
model.add(Dense(1,activation='sigmoid'))
model.summary()

