import sys
sys.path.insert(0, "..")
from functions import load_np_array_pickle
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt
import numpy as np
import random


#Data preparation
affinity_label = load_np_array_pickle('../../cluster_files/ap.pickle')
hierachical_label = load_np_array_pickle('../../cluster_files/hierachical.pickle')
kmeans_label = load_np_array_pickle('../../cluster_files/kmeans.pickle')
dbscan_label = load_np_array_pickle('../../cluster_files/dbscan.pickle')

data_dict = load_np_array_pickle('../../epochs_files/epoch_dict.pickle')
data = [el['data'] for el in data_dict]


cluster_type = input("Bitte wählen Sie Cluster-Algorithmus aus: \n1 = dbscan\n2 = hierachical\n3 = kmeans\n4 = affinity propagation\n")


labels = 0
if cluster_type == "1":
    labels = dbscan_label
if cluster_type == "2":
    labels = hierachical_label
if cluster_type == "3":
    labels = kmeans_label
if cluster_type == "4":
    labels = affinity_label
if len(labels) == 1:
    print('Bitte starte das Programm neu und gib eine zulässige Eingabe ein.')
    sys.exit()


# Get Flatlines
all_flatlines = load_np_array_pickle('../../epochs_files/flatlines.pickle')
amount_flatlines = len(data)
flatlines = random.sample(all_flatlines, amount_flatlines)
flatlines = [el['data'] for el in flatlines]

flatlines_label = len(np.unique(labels))
data += flatlines
labels = labels.tolist()
for i in range(len(flatlines)):
    labels.append(flatlines_label)


# Split data to Train and test
data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size=0.3, random_state=42)
data_train = np.array(data_train)
data_test = np.array(data_test)
labels_train = np.array(labels_train)
labels_test = np.array(labels_test)



# Hyperparameter
learning_rate = 3e-5
loss_function = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
batch_size = 16
epochs = 100
output_dense = len(np.unique(labels_train))

# Recurrent NN
model = tf.keras.Sequential([
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=True, dropout=0.2), input_shape=(len(data[0]), 1)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, dropout=0.1)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(output_dense, activation='softmax')
])


print(len(data_train))
print(len(labels_train))

# Compile the model
model.compile(optimizer=optimizer, loss=loss_function, metrics=['accuracy'])

# Train the model
history = model.fit(data_train, labels_train, batch_size=batch_size, epochs=epochs, validation_data=(data_test, labels_test))
print(model.summary())
model.save("./models/rnn.h5")

plot_model(model, to_file='model_rnn.png', show_shapes=True, show_layer_names=True)
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
