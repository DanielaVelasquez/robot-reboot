import logging
import multiprocessing
import time

import numpy as np
import tensorflow as tf

from src.robot_reboot.dataset_factory import RobotRebootDataSetFactory

logging.getLogger().setLevel(logging.INFO)

rows, cols, layers = 31, 31, 9


def write_tf_record(i, max_movements):
    cnn = tf.keras.models.load_model('model/model_0')
    ds_factory = RobotRebootDataSetFactory(31, cnn, max_depth=20, seed=26, playouts=50)
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
    filename = f'robot_reboot_{i}_test.tfrecord'
    logging.info(f'Saving file {filename}')
    with tf.io.TFRecordWriter(filename) as f:
        f.write(sample.SerializeToString())


def create_tfrecord(batch, n):
    processes = list()
    num_cores = multiprocessing.cpu_count()
    for i in range(0, n):
        if i % num_cores == 0 and i != 0:
            logging.info("Processing jobs")
            [t.join() for t in processes]
            processes = list()
        max_movements = np.random.randint(1, 6)
        process = multiprocessing.Process(target=write_tf_record, args=(i * batch, max_movements))
        process.start()
        processes.append(process)
    logging.info(f"{n} jobs processed, waiting for them to finish")
    [t.join() for t in processes]
    logging.info("Finished processed")


if __name__ == '__main__':
    np.random.seed(27)
    start_time = time.time()
    create_tfrecord(0, 2)
    logging.info("--- %s seconds ---" % (time.time() - start_time))
