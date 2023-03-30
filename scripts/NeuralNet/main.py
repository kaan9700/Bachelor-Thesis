import sys
sys.path.insert(0, "..")
from functions import load_np_array_pickle, generate_random_noise
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt
import numpy as np
import random
from tensorflow.keras.callbacks import EarlyStopping
import keras_tuner as kt
from keras_models import ffnn_model, cnn_model, rnn_model, f1_m, precision_m, recall_m
import math


tf.config.set_visible_devices([], 'GPU')

# Data preparation


data_dict = load_np_array_pickle('../../files/windows_files/final_windows.pickle')
data = [el['data'] for el in data_dict]


# Get Flatlines
all_flatlines = load_np_array_pickle('../../files/windows_files/windows_flatlines.pickle')
print(len(all_flatlines))

amount_flatlines = math.floor(len(data) / 2)
flatlines = random.sample(all_flatlines, amount_flatlines)
flatlines = [el['data'] for el in flatlines]
flatlines = np.array(flatlines)
noise = generate_random_noise(n_signals=math.ceil(len(data) / 2), time=len(data[0]), min_amp=0.3, max_amp=0.45)



positive_data = data
negative_data = np.concatenate((flatlines, noise), axis=0)


# Zielvariablen erstellen (1 für Signal, 0 für Nicht-Signal)
positive_labels = np.ones((len(positive_data), 1))
negative_labels = np.zeros((len(negative_data), 1))

# Daten und Labels zusammenfügen
data = np.concatenate((positive_data, negative_data), axis=0)
labels = np.concatenate((positive_labels, negative_labels), axis=0)

# Split data to Train and test
data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size=0.3, random_state=42)
data_train = np.array(data_train)
data_test = np.array(data_test)
labels_train = np.array(labels_train)
labels_test = np.array(labels_test)


print(len(positive_labels))
print(len(negative_labels))

output_dense = len(np.unique(labels_train))
epochs = 100
batch_size = 32

ann_type = input("Welche Netzarchitektur soll verwendet werden:\nFeedforward = 1\nConvolutional = 2\nRecurrent = 3\n\n")


if ann_type == "1":
    model_name = 'ffnn'
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(len(data[0]),)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(2, activation='softmax')
    ])


if ann_type == "2":
    model_name = 'cnn'
if ann_type == "3":
    model_name = 'rnn'

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=6e-5),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics='accuracy')

# Train the best model on the full dataset
history = model.fit(data, labels, batch_size=batch_size, epochs=epochs, validation_data=(data_test, labels_test))




model.save("./models/"+model_name+".h5")

plot_model(model, to_file='model_'+model_name+'.png', show_shapes=True, show_layer_names=True)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
