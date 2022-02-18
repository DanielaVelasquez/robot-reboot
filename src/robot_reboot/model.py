import tensorflow as tf
from keras import Model, Input
from keras.layers import Conv2D, Flatten, Dense


def get_model_v2(input_shape, outputs, convolutions=3):
    tf.random.set_seed(seed=26)
    maze_input = Input(shape=input_shape, name='maze_input')
    mb = maze_input
    for i in range(convolutions):
        mb = Conv2D(64, (3, 3), padding='same', activation='relu')(mb)

    prob_conv = Conv2D(2, (1, 1), activation='relu')(mb)
    prob_flat = Flatten()(prob_conv)
    prob_output = Dense(outputs, activation='softmax')(prob_flat)

    value_conv = Conv2D(1, (1, 1), activation='relu')(mb)
    value_flat = Flatten()(value_conv)
    value_hidden = Dense(256, activation='relu')(value_flat)
    value_output = Dense(1, activation='tanh')(value_hidden)
    return Model(inputs=[maze_input],
                 outputs=[prob_output, value_output])
