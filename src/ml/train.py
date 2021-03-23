import argparse
import logging
import os
import tensorflow as tf

from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint

logging.getLogger().setLevel(logging.INFO)

HEIGHT = 31
WIDTH = 31
DEPTH = 9
NUM_CLASSES = 16
SHUFFLE_BUFFER_SIZE = 100


def get_filenames(channel_name, channel):
    if channel_name in ['train', 'validation', 'eval']:
        return [os.path.join(channel, f'{channel_name}.tfrecords')]
    else:
        raise ValueError(f'Invalid data for {channel_name}')


def read_tfrecord(example):
    feature_desc = {
        'v': tf.io.FixedLenFeature([], tf.float32),
        'p': tf.io.VarLenFeature(tf.float32),
        's': tf.io.VarLenFeature(tf.float32)
    }
    parsed = tf.io.parse_single_example(example, feature_desc)
    x = tf.reshape(tf.sparse.to_dense(parsed['s']), (1, HEIGHT, WIDTH, DEPTH))
    p = tf.reshape(tf.sparse.to_dense(parsed['p']), (1, 16))
    v = tf.reshape(parsed['v'], (1, 1))

    y = {'v': v, 'p': p}
    return x, y


def _input(epochs, channel, channel_name):
    filenames = get_filenames(channel_name, channel)
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(read_tfrecord)#, num_parallel_calls=20)

    # if channel_name == 'train':
    #     dataset = dataset.repeat(epochs)
    #     dataset = dataset.shuffle(SHUFFLE_BUFFER_SIZE).prefetch(1)

    return dataset


def save_model(model, dir, version):
    output = os.path.join(dir, 'model_'+version)
    model.save(output)

    logging.info("Model successfully saved at: {}".format(output))


def train(args):
    logging.info("Getting data sets")
    train_ds = _input(args.epochs, args.train, 'train')
    valid_ds = _input(args.epochs, args.validation, 'validation')
    eval_ds = _input(args.epochs, args.eval, 'eval')

    logging.info("Loading model")
    model = tf.keras.models.load_model(args.model)
    checkpoint = ModelCheckpoint(args.model_dir + 'checkpoint_{epoch}.h5')

    logging.info("Starting to train")
    model.fit(train_ds)#, epochs=args.epochs, validation_data=valid_ds, callbacks=[checkpoint])
    #
    # logging.info("Evaluating the model")
    # v_eval, p_eval = model.evaluate(eval_ds)
    #
    # logging.info('Test loss v:{}'.format(v_eval))
    # logging.info('Tess loss p:{}'.format(p_eval))

    return save_model(model, args.model_dir, args.model_version)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--train',
        type=str,
        required=True,
        help='Directory where the robot reboot training data is stored'
    )
    parser.add_argument(
        '--validation',
        type=str,
        required=True,
        help='Directory where the robot reboot validation data is stored'
    )
    parser.add_argument(
        '--eval',
        type=str,
        required=True,
        help='Directory where the robot reboot evaluation data is stored'
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Directory where the current model is stored'
    )
    parser.add_argument(
        '--model_dir',
        type=str,
        required=True,
        help='Directory where the model will be stored'
    )
    parser.add_argument(
        '--model_version',
        type=str,
        required=True,
        help='Directory where the model will be stored'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=10,
        help='The number of steps to use for training.'
    )
    args = parser.parse_args()
    train(args)
