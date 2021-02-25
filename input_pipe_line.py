"""

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf


def data_loader(mode=tf.estimator.ModeKeys.TRAIN,
                  dataset_dir='./',
                  preprocess_data=None,
                  batch_size=8):

    """
    
    :param mode: 
    :param dataset_dir: 
    :param preprocess_data: 
    :param batch_size: 
    :return: 
    """

    def parse(feature):
        features = tf.io.parse_single_example(
            feature,
            features={
                'image/encoded':
                    tf.FixedLenFeature((), tf.string, default_value=''),
                'image/format':
                    tf.FixedLenFeature((), tf.string, default_value='jpeg'),
                'image/filename':
                    tf.FixedLenFeature((), tf.string, default_value=''),
                'image/height':
                    tf.FixedLenFeature((), tf.int64, 1),
                'image/width':
                    tf.FixedLenFeature((), tf.int64, 1),
            }
        )

        image = tf.image.decode_jpeg(features['image/encoded'], channels=3)
        image = tf.reshape(image, tf.stack([features['image/height'], features['image/width'], 3]))
        image = tf.cast(image, tf.float32) / 255.0
        image_name = features['image/filename']
        image_height = features['image/height']
        image_width = features['image/width']

        tensor_dict = {
            'image': image,
            'height': image_height,
            'width': image_width,
            'filename': image_name,
        }
        return image

    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = False

    autotune = tf.data.experimental.AUTOTUNE
    tfrecord = tf.io.gfile.glob(dataset_dir + '/{}-*'.format(mode.lower()))
    dataset = tf.data.TFRecordDataset(tfrecord)

    dataset = dataset.map(parse, num_parallel_calls=autotune)

    dataset = dataset.batch(batch_size=batch_size, drop_remainder=True)
    if preprocess_data is not None:
        dataset = dataset.map(preprocess_data, num_parallel_calls=autotune)

    if mode == tf.estimator.ModeKeys.TRAIN:
        dataset = dataset.with_options(
            ignore_order
        )  # uses data as soon as it streams in, rather than in its original order
        dataset = dataset.shuffle(8 * batch_size)
        dataset = dataset.prefetch(buffer_size=autotune)
        dataset = dataset.repeat()
        # dataset = dataset.cache()

    elif mode == tf.estimator.ModeKeys.EVAL:
        dataset = dataset.apply(tf.data.experimental.ignore_errors())
        dataset = dataset.prefetch(autotune)

    else:
        dataset = dataset.apply(tf.data.experimental.ignore_errors())
        dataset = dataset.prefetch(autotune)
    return dataset
