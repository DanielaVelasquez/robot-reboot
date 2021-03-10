from tensorflow.keras import models, layers


class CNN(models.Sequential):
    def __init__(self, input_shape=(31, 31, 9), n_outputs=4, convolutions=3):
        models.Sequential.__init__(self)
        self.add(layers.Input(input_shape, name='input'))
        for i in range(convolutions):
            self.add(layers.Conv2D(64, (4, 4), activation='relu', padding='same', name=f'conv_{i + 1}'))
            self.add(layers.MaxPooling2D(2))
        self.add(layers.Flatten())
        self.add(layers.Dense(128, activation='relu', name='dense_1'))
        self.add(layers.Dropout(0.35, name='dropout_1'))
        self.add(layers.Dense(64, activation='relu', name='dense_2'))
        self.add(layers.Dropout(0.35, name='dropout_2'))

        p = layers.Dense(n_outputs, activation='tanh', name='p')
        v = layers.Dense(1, activation='tanh', name='v')
        self.add
        self.add(layers.Concatenate([p, v]))
