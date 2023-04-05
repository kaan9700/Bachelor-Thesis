import keras_tuner as kt
import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras import backend as K

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

# Define the model builder function for Keras Tuner


def ffnn_model(hp, inputShape, outputShape):
    # Hyperparameters
    learning_rate = hp.Choice('learning_rate', values=[1e-4, 1e-5])
    dropout_rate_2 = hp.Float('dropout_rate_2', min_value=0.0, max_value=0.5, step=0.1)
    dropout_rate_3 = hp.Float('dropout_rate_3', min_value=0.0, max_value=0.5, step=0.1)
    num_neurons_1 = hp.Int('num_neurons_1', min_value=16, max_value=64, step=16)
    num_neurons_2 = hp.Int('num_neurons_2', min_value=16, max_value=32, step=16)
    num_neurons_3 = hp.Int('num_neurons_3', min_value=16, max_value=32, step=16)

    # Define the model architecture
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(num_neurons_1, activation='relu', input_shape=(inputShape,)),
        tf.keras.layers.Dense(num_neurons_2, activation='relu'),
        tf.keras.layers.Dropout(dropout_rate_2),
        tf.keras.layers.Dense(num_neurons_3, activation='relu'),
        tf.keras.layers.Dropout(dropout_rate_3),
        tf.keras.layers.Dense(2, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy', f1_m, precision_m, recall_m])

    return model


def cnn_model(hp, inputShape, outputShape):
    # Hyperparameters
    learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4, 1e-5])
    filters = hp.Int('filters', min_value=16, max_value=128, step=16)
    kernel_size = hp.Int('kernel_size', min_value=1, max_value=4, step=1)
    pool_size = hp.Int('pool_size', min_value=1, max_value=4, step=1)
    dense_units = hp.Int('dense_units', min_value=16, max_value=64, step=16)
    dropout_rate = hp.Float('dropout_rate', min_value=0.0, max_value=0.5, step=0.1)


    # Define the model architecture
    model = tf.keras.Sequential([
        tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, activation='relu',
                               input_shape=(inputShape, 1)),
        tf.keras.layers.MaxPooling1D(pool_size=pool_size),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv1D(filters=filters // 2, kernel_size=kernel_size, activation='relu'),
        tf.keras.layers.MaxPooling1D(pool_size=pool_size),
        tf.keras.layers.Dropout(dropout_rate),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(dense_units, activation='relu'),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=['accuracy', f1_m, precision_m, recall_m])

    return model


def rnn_model(hp, inputShape, outputShape):
    # Hyperparameters
    learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4, 1e-5])
    num_lstm_1 = hp.Int('num_lstm_1', min_value=64, max_value=512, step=64)
    num_lstm_2 = hp.Int('num_lstm_2', min_value=64, max_value=256, step=64)
    dense_units = hp.Int('dense_units', min_value=32, max_value=256, step=32)
    dropout_rate_1 = hp.Float('dropout_rate_1', min_value=0.0, max_value=0.5, step=0.1)
    dropout_rate_2 = hp.Float('dropout_rate_2', min_value=0.0, max_value=0.5, step=0.1)
    dropout_rate_3 = hp.Float('dropout_rate_2', min_value=0.0, max_value=0.5, step=0.1)

    # Define the model architecture
    model = tf.keras.Sequential([
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(num_lstm_1, return_sequences=True, dropout=dropout_rate_1),
                                      input_shape=(inputShape, 1)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(num_lstm_2, dropout=dropout_rate_2)),
        tf.keras.layers.Dense(dense_units, activation='relu'),
        tf.keras.layers.Dropout(dropout_rate_3),
        tf.keras.layers.Dense(outputShape, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['acc', f1_m, precision_m, recall_m])

    return model
