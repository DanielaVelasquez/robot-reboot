from tensorflow.keras import models, layers


class CNN(models.Model):
    def __init__(self, input_shape=(31, 31, 9), n_outputs=4, convolutions=3):
        super(CNN, self).__init__()
        self.cnn_input = layers.Input(shape=input_shape, name='input')
        self.convolutions = list()
        for i in range(convolutions):
            self.convolutions.append(layers.Conv2D(64, (4, 4), activation='relu', padding='same', name=f'conv_{i + 1}'))
            self.convolutions.append(layers.MaxPooling2D(2))
        self.flatten = layers.Flatten()
        self.dense_1 = layers.Dense(128, activation='relu', name='dense_1')
        self.dropout_1 = layers.Dropout(0.35, name='dropout_1')
        self.dense_2 = layers.Dense(64, activation='relu', name='dense_2')
        self.dropout_2 = layers.Dropout(0.35, name='dropout_2')

        self.p_out = layers.Dense(n_outputs, activation='tanh', name='p')
        self.v_out = layers.Dense(1, activation='tanh', name='v')

    def call(self, inputs, training=None, mask=None):
        x = self.cnn_input(inputs)
        for cnn in self.convolutions:
            x = cnn(x)
        x = self.flatten(x)

        x = self.dense_1(x)
        if training:
            x = self.dropout_1(x)

        x = self.dense_2(x)
        if training:
            x = self.dropout_2(x)

        p = self.p_out(x)
        v = self.v_out(x)

        return [p, v]
