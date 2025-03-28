#import tensorflow onehotencoding
from tensorflow.keras.preprocessing.text import one_hot


###sentences
sent=["the glass of milk",
    "the glass of water",
"the glass of juice",
"the glass of wine",
"the glass of beer",
"i am good boy",
"i am good girl",
"i am bad boy",
"i am bad girl"]

#define vocabaluray size
voc_size=10000

#one hotrepresentaion
onehot_repr=[one_hot(s,voc_size) for s in sent]
#print(onehot_repr)

#import libraries related to  embediing  sequences and sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding
import numpy as np

#pad sequences
sent_length=8
embedded_docs=pad_sequences(onehot_repr, padding='post',maxlen=sent_length)
print(embedded_docs)







