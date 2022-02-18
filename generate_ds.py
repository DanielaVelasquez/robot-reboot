import argparse
import logging
import multiprocessing
import time

import numpy as np
import tensorflow as tf

from src.robot_reboot.dataset_factory import RobotRebootDataSetFactory
from src.robot_reboot.model import get_model

logging.getLogger().setLevel(logging.INFO)
rows, cols, layers = 31, 31, 9


def write_tf_record(i, max_movements, model_dir, iteration):
    cnn = get_model()
    cnn.load_weights(model_dir)
    ds_factory = RobotRebootDataSetFactory(31, cnn, max_depth=20, playouts=50)
    v, p, s = ds_factory.create(locate_robot_close_goal=True, max_movements=max_movements)
    state = s.reshape(rows * cols * layers, )
    sample = tf.train.Example(
        features=tf.train.Features(
            feature={
                'v': tf.train.Feature(float_list=tf.train.FloatList(value=[v])),
                'p': tf.train.Feature(float_list=tf.train.FloatList(value=p)),
                's': tf.train.Feature(float_list=tf.train.FloatList(value=state))
            }
        )
    )
    filename = f'robot_reboot_data/iter_{iteration}/robot_reboot_{i}.tfrecords'
    logging.info(f'Saving file {filename}')
    with tf.io.TFRecordWriter(filename) as f:
        f.write(sample.SerializeToString())


def create_tfrecord(offset, threshold, model_dir, time_threshold, iteration):
    processes = list()
    num_cores = multiprocessing.cpu_count()
    logging.info(f"Working with {num_cores} cores")
    start_time = time.time()
    total_seconds_running = time.time() - start_time
    i = 0
    while (time_threshold and total_seconds_running < threshold) or (not time_threshold and i < threshold):
        if i % num_cores == 0 and i != 0:
            if time_threshold:
                logging.info(f"Processing jobs {total_seconds_running}/{threshold}")
            else:
                logging.info(f"Processing jobs {i}/{threshold}")

            [t.join() for t in processes]
            processes = list()
        max_movements = np.random.randint(1, 6)
        process = multiprocessing.Process(target=write_tf_record, args=(i + offset, max_movements, model_dir, iteration))
        process.start()
        processes.append(process)
        total_seconds_running = time.time() - start_time
        if time_threshold:
            logging.info(f"Total time {total_seconds_running}/{threshold}")
        else:
            logging.info(f"Total samples {i}/{threshold}")
        i += 1
    logging.info(f"{i} jobs processed, waiting for them to finish")
    [t.join() for t in processes]
    logging.info(f'{i} records generated')
    logging.info("Finished after %s seconds ---" % (time.time() - start_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--threshold',
        type=int,
        required=True,
        help='Max number of seconds that samples will be generated'
    )
    parser.add_argument(
        '--time_threshold',
        type=int,
        required=False,
        default=1,
        help='Determine if the threshold is for time 1 or 0 for number of samples'
    )
    parser.add_argument(
        '--seed',
        type=int,
        required=True,
        help='Seed to generate data set samples'
    )
    parser.add_argument(
        '--offset',
        type=int,
        default=0,
        help='Id to start saving datasets'
    )

    parser.add_argument(
        '--model_dir',
        type=str,
        required=True,
        help='Model used to generate the dataset'
    )

    parser.add_argument(
        '--iter',
        type=int,
        required=False,
        default=0,
        help='Iteration number'
    )

    args = parser.parse_args()

    if args.time_threshold > 0:
        logging.info(f'Running for {args.threshold} seconds with seed {args.seed}')
    else:
        logging.info(f'Running to generate {args.threshold} samples with seed {args.seed}')
    np.random.seed(args.seed)

    create_tfrecord(args.offset, args.threshold, args.model_dir, args.time_threshold > 0, args.iter)
