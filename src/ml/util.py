import tensorflow as tf

rows, cols, layers = 31, 31, 9


def get_feature_desc():
    return {
        'v': tf.io.FixedLenFeature([], tf.float32),
        'p': tf.io.VarLenFeature(tf.float32),
        's': tf.io.VarLenFeature(tf.float32)
    }


def get_tf_sample(v, p, s):
    state = s.reshape(rows * cols * layers, )
    return tf.train.Example(
        features=tf.train.Features(
            feature={
                'v': tf.train.Feature(float_list=tf.train.FloatList(value=[v])),
                'p': tf.train.Feature(float_list=tf.train.FloatList(value=p)),
                's': tf.train.Feature(float_list=tf.train.FloatList(value=state))
            }
        )
    )


def deserialize(serialized_sample):
    parsed = tf.io.parse_single_example(serialized_sample, get_feature_desc())
    s = tf.reshape(tf.sparse.to_dense(parsed['s']), (1, rows, cols, layers))
    p = tf.reshape(tf.sparse.to_dense(parsed['p']), (1, 16))
    v = tf.reshape(parsed['v'], (1, 1))
    return s, v, p


def get_test_data(filenames):
    dataset = tf.data.TFRecordDataset(filenames)
    states = list()
    for serialized in dataset:
        parsed = tf.io.parse_single_example(serialized, {'s': tf.io.VarLenFeature(tf.float32)})
        s = tf.reshape(tf.sparse.to_dense(parsed['s']), (rows, cols, layers)).numpy()
        states.append(s)
    return states
