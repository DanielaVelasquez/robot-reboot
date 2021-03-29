from tensorflow import keras
from tensorflow.keras import layers, Model
import tensorflow as tf


def get_cnn_model(input_shape=(31, 31, 9), n_outputs=4, convolutions=3, optimizer='adam', seed=26):
    tf.random.set_seed(seed=seed)
    _input = layers.Input(shape=input_shape, name='input')
    x = layers.Conv2D(64, (4, 4), activation='relu', padding='same', name=f'conv_0')(_input)
    x = layers.MaxPooling2D(2)(x)
    for i in range(convolutions - 1):
        x = layers.Conv2D(64, (4, 4), activation='relu', padding='same', name=f'conv_{i + 1}')(x)
        x = layers.MaxPooling2D(2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu', name='dense_1')(x)
    x = layers.Dropout(0.35, name='dropout_1')(x)
    x = layers.Dense(128, activation='relu', name='dense_2')(x)
    x = layers.Dropout(0.35, name='dropout_2')(x)
    p = layers.Dense(n_outputs, activation='tanh', name='p')(x)
    v = layers.Dense(1, activation='tanh', name='v')(x)
    cnn_model = Model(inputs=_input, outputs=[v, p])
    losses = {
        "v": 'mean_squared_error',
        "p": keras.losses.BinaryCrossentropy()
    }

    cnn_model.compile(loss=losses, optimizer=optimizer)
    return cnn_model


def get_model(input_shape=(31, 31, 9), n_outputs=4, convolutions=3, optimizer='adam', seed=26):
    tf.random.set_seed(seed=seed)
    _input = layers.Input(shape=input_shape, name='input')
    x = layers.Conv2D(64, (4, 4), activation='relu', padding='same', name=f'conv_0')(_input)
    x = layers.MaxPooling2D(2)(x)
    for i in range(convolutions - 1):
        x = layers.Conv2D(64, (4, 4), activation='relu', padding='same', name=f'conv_{i + 1}')(x)
        x = layers.MaxPooling2D(2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu', name='dense_1')(x)
    x = layers.Dropout(0.35, name='dropout_1')(x)
    x = layers.Dense(128, activation='relu', name='dense_2')(x)
    x = layers.Dropout(0.35, name='dropout_2')(x)
    p = layers.Dense(n_outputs, activation='tanh', name='p')(x)
    v = layers.Dense(1, activation='tanh', name='v')(x)
    cnn_model = Model(inputs=_input, outputs=[v, p])
    return cnn_model
