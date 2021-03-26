import tensorflow as tf
import pandas as pd
import numpy as np
from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Dense, Conv1D, MaxPool1D, Activation, Embedding, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras import layers

import os
from functools import reduce
import datetime
from transformer import TransformerBlock, TokenAndPositionEmbedding

os.environ['KMP_DUPLICATE_LIB_OK']='True'

df = pd.read_csv("sample_training2_old.csv")
texts = df.iloc[:,0].to_list()

tk =  Tokenizer(num_words=None, char_level=True, oov_token='UNK')
tk.fit_on_texts(texts)
print(tk.word_index)
print("word index len: ", len(tk.word_index))

sequences = tk.texts_to_sequences(texts)
print(texts[0])
print(sequences[0])

lens = [len(x) for i, x in enumerate(sequences)]
print(lens)
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
embed_dim = 50
num_heads = 2
ff_dim = 32

input_size = 1400
dimension = 50
vocabulary_size = len(tk.word_index)

inputs = layers.Input(shape=(input_size,))
embedding_layer = TokenAndPositionEmbedding(input_size, vocabulary_size + 1, embed_dim)
X = embedding_layer(inputs)
transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
transformer_block2 = TransformerBlock(embed_dim, num_heads, ff_dim)

X = transformer_block(X)
X = transformer_block2(X)
#X = layers.GlobalAvgPool1D()(X)
X = layers.Dropout(0.5)(X)
X = layers.Dense(20, activation='relu')(X)
X = layers.Dropout(0.1)(X)
outputs = layers.Dense(1, activation='linear')(X)


model = Model(inputs=inputs, outputs=outputs)
model.compile(optimizer='adam', loss='mse', metrics=['mse']) # Adam, categorical_crossentropy
model.summary()

# log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
#
# model.fit(np_data, y_data, epochs=10, batch_size= 64, validation_split=0.3, callbacks=[tensorboard_callback])

model.fit(np_data, y_data, epochs=100, batch_size= 64, validation_split=0.2, verbose=1)
