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
from sklearn import cluster, datasets

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
xlogs = df.iloc[:, 1].to_list()
print("array: ", xlogs[:20])
first_ones = np.array(xlogs[:20])
first_ones_rs = np.reshape(first_ones, (first_ones.shape[0], 1))

y_data = np.array(xlogs)
print("2: ", y_data[2])
print("1000: ", y_data[1000])
print("0: ", y_data[0])

s1 = np.array([y_data[2]])
s1 = np.reshape(s1, (1,1))
s2 = np.array([y_data[1000]])
s2 = np.reshape(s2, (1,1))
s3 = np.array([y_data[0]])
s3 = np.reshape(s3, (1,1))

print("Shape Y ", y_data.shape)
y_data = np.reshape(y_data, (y_data.shape[0], 1))

k_means = cluster.KMeans(n_clusters=2)
k_means.fit(y_data)

p1 = k_means.predict(s1)
print("P1: ", p1)

p2 = k_means.predict(s2)
print("P2: ", p2)

p3 = k_means.predict(s3)
print("P3: ", p3)

p4 = k_means.predict(first_ones_rs)
print("first_ones: ", first_ones_rs)
print("P4: ", p4)

for i in range(0, 20):
    print(str(first_ones[i]) + ", " + str(p4[i]))
