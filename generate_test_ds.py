import logging
import argparse
import numpy as np
import time
import tensorflow as tf
from src.robot_reboot.factory import RobotRebootFactory

logging.getLogger().setLevel(logging.INFO)
rows, cols, layers = 31, 31, 9


def generate_validation_ds(total):
    factory = RobotRebootFactory()
    with tf.io.TFRecordWriter('robot_reboot_data/test.tfrecords') as f:
        for i in range(total):
            game, state, quadrant_ids = factory.create(31, locate_robot_close_goal=True,
                                                       max_movements=np.random.randint(1, 6))
            state = state.get_matrix().reshape(rows * cols * layers, )
            sample = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        's': tf.train.Feature(float_list=tf.train.FloatList(value=state))
                    }
                )
            )
            f.write(sample.SerializeToString())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--total_samples',
        type=int,
        required=True,
        help='Number of samples to create'
    )
    parser.add_argument(
        '--seed',
        type=int,
        required=True,
        help='Seed to generate data'
    )
    args = parser.parse_args()
    logging.info(f'Generating {args.total_samples} samples with seed {args.seed}')
    np.random.seed(args.seed)
    start_time = time.time()
    generate_validation_ds(args.total_samples)
    logging.info('Finished after %s seconds' % (time.time() - start_time))
