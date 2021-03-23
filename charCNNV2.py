#%%

import tensorflow as tf
import pandas as pd
import numpy as np
from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Dense, Conv1D, MaxPool1D, Activation, Embedding, Flatten, BatchNormalization
from tensorflow.keras.models import Model
import os
from functools import reduce
import datetime

#%%


df = pd.read_csv("https://raw.githubusercontent.com/manuelr417/charecterlevel/master/sample_training2.csv")
texts = df.iloc[:,0].to_list()

tk =  Tokenizer(num_words=None, char_level=True, oov_token='UNK')
tk.fit_on_texts(texts)
print(tk.word_index)
print("word index len: ", len(tk.word_index))

sequences = tk.texts_to_sequences(texts)
#print(texts[0])
#print(sequences[0])

lens = [len(x) for i, x in enumerate(sequences)]
#print(lens)
print("max: ", max(lens))
sum_ser = reduce(lambda x, y: x + y, lens)
print("sum ", sum_ser)
avg_len = (sum_ser * 1.0)/(len(lens))
print("avg_len: ", avg_len)

data = pad_sequences(sequences, maxlen=1400, padding='post')


#%%

np_data = np.array(data)
print("Shape X ", np_data.shape)
xlogs = df.iloc[:, 1].to_list()

y_data = np.array(xlogs)

print("Shape Y ", y_data.shape)

#%%

# Neural net
input_size = 1400
dimension = 50
vocabulary_size = len(tk.word_index)

input_layer = Input(shape=(input_size,), name="input_layer")
embedding_layer = Embedding(vocabulary_size + 1, dimension, input_length=input_size, name="embedding")(input_layer)
num_filters = 64
filter_size = 7

X = Conv1D(num_filters, filter_size, padding='same', name="conv1")(embedding_layer)
X = BatchNormalization()(X)
X = Activation(activation='relu')(X)
#X = MaxPool1D(pool_size=2, name="maxpool1")(conv_1)

num_filters = 64
filter_size = 7


X = Conv1D(num_filters, filter_size, padding='same', name="conv2")(X)
X = BatchNormalization()(X)
X = Activation(activation='relu')(X)
#max_pool2 = MaxPool1D(pool_size=2, name="maxpool2")(conv_2)

num_filters = 64
filter_size = 7


X = Conv1D(num_filters, filter_size, padding='same', name="conv3")(X)
X = BatchNormalization()(X)
X = Activation(activation='relu')(X)
#max_pool3 = MaxPool1D(pool_size=2, name="maxpool3")(conv_3)


X = Flatten()(X)
X = Dense(1024*2, activation='relu', name="dense1")(X)
X = Dense(1024, activation='relu', name="dense2")(X)
X = Dense(512, activation='relu', name="dense3")(X)
X = Dense(512/2, activation='relu', name="dense4")(X)

output = Dense(1, name="dense5")(X)

#%%

model = Model(inputs=input_layer, outputs=output)
model.compile(optimizer='adam', loss='mae', metrics=['mse', 'mae']) # Adam, categorical_crossentropy
model.summary()

#%%

model.fit(np_data, y_data, epochs=10, batch_size= 64, validation_split=0.3)

#%%



#%%


