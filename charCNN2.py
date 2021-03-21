import tensorflow as tf
import pandas as pd
import numpy as np
from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Dense, Conv1D, MaxPool1D, Activation, Embedding, Flatten, GlobalMaxPool1D, Dropout
from tensorflow.keras.models import Model
import os
from functools import reduce
import datetime


os.environ['KMP_DUPLICATE_LIB_OK']='True'

df = pd.read_csv("sample_training2.csv")
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



print()
print(data[0])

np_data = np.array(data)
print("Shape X ", np_data.shape)
xlogs = df.iloc[:, 1].to_list()
print(xlogs)

y_data = np.array(xlogs)

print("Shape Y ", y_data.shape)

# Neural net
input_size = 1400
dimension = 50
vocabulary_size = len(tk.word_index)

input_layer = Input(shape=(input_size,), name="input_layer")
embedding_layer = Embedding(vocabulary_size + 1, dimension, input_length=input_size, name="embedding")(input_layer)
num_filters = 64
filter_size = 7

conv_1 = Conv1D(num_filters, filter_size, activation='relu', name="conv1")(embedding_layer)

num_filters = 128
filter_size = 7


conv_2 = Conv1D(num_filters, filter_size, activation='relu', name="conv2")(conv_1)
drop_out = Dropout(0.50, name="dropout")(conv_2)
max_pool = MaxPool1D(name="maxpool2")(drop_out)


X = Flatten()(max_pool)

# X = Dense(256, activation='relu', name="dense1")(X)
# X = Dense(128, activation='relu', name="dense2")(X)

X = Dense(64, activation='relu', name="dense1")(X)

X = Dense(32, activation='relu', name="dense2")(X)

output = Dense(1, activation='linear', name="dense3")(X)

model = Model(inputs=input_layer, outputs=output)
model.compile(optimizer='adam', loss='mae', metrics=['mse', 'mae']) # Adam, categorical_crossentropy
model.summary()

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

model.fit(np_data, y_data, epochs=30, batch_size= 64, validation_split=0.3, callbacks=[tensorboard_callback])

#model.fit(np_data, y_data, epochs=10, batch_size= 64, validation_split=0.3)
