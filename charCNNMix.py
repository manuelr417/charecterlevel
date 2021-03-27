import tensorflow as tf
import pandas as pd
import numpy as np
from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Dense, Conv1D, MaxPool1D, Activation, Embedding, Flatten
from tensorflow.keras.models import Model
import os
from functools import reduce
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import cluster

def plot_loss(history, filename):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.ylim([0, 10])
    plt.xlabel('Epoch')
    plt.ylabel('Error [XLogP]')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)

os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ["CUDA_VISIBLE_DEVICES"]="0"

df = pd.read_csv("sample_training2.csv")
texts = df.iloc[:,0].to_list()

tk =  Tokenizer(num_words=None, char_level=True, oov_token='UNK')
tk.fit_on_texts(texts)
print(tk.word_index)
print("word index len: ", len(tk.word_index))
vocabulary_size = len(tk.word_index)


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
#print(data[0])

np_data = np.array(data)
print("Shape X ", np_data.shape)
xlogs = df.iloc[:, 1].to_list()
print(xlogs)

y_data = np.array(xlogs)

print("Shape Y ", y_data.shape)

####################################################
#Cluster
print("Shape Y ", y_data.shape)
y_data_c = np.reshape(y_data, (y_data.shape[0], 1))
print("Shape Y_c ", y_data_c.shape)
k_means = cluster.KMeans(n_clusters=2)
k_means.fit(y_data_c)
predict = k_means.predict(y_data_c)
print(predict[:20])
print(type(predict))
print("Shape Predict: ", predict.shape)


#####################################################
#
################################################
# Filter out
x_train = []
y_train = []
len = predict.shape[0]
for i in range(0, len):
    if predict[i] == 0:
        x_train.append(np_data[i])
        y_train.append(y_data[i])

x_train = np.array(x_train)
y_train = np.array(y_train)
print("Shape: x_train: ", x_train.shape)
print("Shape: y_train: ", y_train.shape)


################################################
# Neural net
input_size = 1400
dimension = 50

input_layer = Input(shape=(input_size,), name="input_layer")
embedding_layer = Embedding(vocabulary_size + 1, dimension, input_length=input_size, name="embedding")(input_layer)
num_filters = 64
filter_size = 3

conv_1 = Conv1D(num_filters, filter_size, activation='relu', name="conv1")(embedding_layer)
max_pool1 = MaxPool1D(pool_size=2, name="maxpool1")(conv_1)

num_filters = 128
filter_size = 3


conv_2 = Conv1D(num_filters, filter_size, activation='relu', name="conv2")(max_pool1)
max_pool2 = MaxPool1D(pool_size=2, name="maxpool2")(conv_2)

num_filters = 256
filter_size = 3


conv_3 = Conv1D(num_filters, filter_size, activation='relu', name="conv3")(max_pool2)
max_pool3 = MaxPool1D(pool_size=2, name="maxpool3")(conv_3)


X = Flatten()(max_pool3)

dense1 = Dense(64, activation='relu', name="dense1")(X)
dense2 = Dense(32, activation='relu', name="dense2")(dense1)

output = Dense(1, activation='linear', name="dense3")(dense2)

model = Model(inputs=input_layer, outputs=output)
model.compile(optimizer='adam', loss='mse', metrics=['mse', 'mae']) # Adam, categorical_crossentropy
model.summary()

#log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
#tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

#model.fit(np_data, y_data, epochs=50, batch_size= 64, validation_split=0.3, callbacks=[tensorboard_callback])

#history = model.fit(np_data, y_data, epochs=10, batch_size= 64, validation_split=0.3)
history = model.fit(x_train, y_train, epochs=10, batch_size= 64, validation_split=0.3)

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()

plot_loss(history=history, filename="plots/train.png")
