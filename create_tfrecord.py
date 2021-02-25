"""

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
import sys
import threading

import numpy as np
from datetime import datetime
import tensorflow as tf
from six.moves import xrange
from dataset_utils import bytes_feature
from dataset_utils import int64_feature
from dataset_utils import ImageCoder

tf.app.flags.DEFINE_string('train_directory',
                           '/media/dat/68fa98f8-9d03-4c1e-9bdb-c71ea72ab6fa/dat/zero_DCE/data/train_data',
                           'Training data directory')
tf.app.flags.DEFINE_string('validation_directory', '/tmp/',
                           'Validation data directory')
tf.app.flags.DEFINE_string('output_directory', '/media/dat/68fa98f8-9d03-4c1e-9bdb-c71ea72ab6fa/dat/zero_DCE/data/tfrecord',
                           'Output data directory')

tf.app.flags.DEFINE_integer('train_shards', 24,
                            'Number of shards in training TFRecord files.')
tf.app.flags.DEFINE_integer('validation_shards', 128,
                            'Number of shards in validation TFRecord files.')

tf.app.flags.DEFINE_integer('num_threads', 8,
                            'Number of threads to preprocess the images.')

FLAGS = tf.app.flags.FLAGS


def image_to_tfexample_only(image_data, image_name, image_format, height, width):
    """

    :param image_data:
    :param image_format:
    :param height:
    :param width:
    :return:
    """
    return tf.train.Example(
        features=tf.train.Features(
            feature={
                'image/encoded': bytes_feature(image_data),
                'image/filename': bytes_feature(image_name),
                'image/format': bytes_feature(image_format),
                'image/height': int64_feature(height),
                'image/width': int64_feature(width),
            }
        )
    )


def _find_image_files(directory):
    """

    :param directory:
    :return:
    """
    list_img = [f for f in os.listdir(directory) if f.endswith('jpg')]
    shuffled_index = np.arange(len(list_img))
    random.seed(12345)
    random.shuffle(shuffled_index)
    return [os.path.join(directory, list_img[i]) for i in shuffled_index]


def _process_image(filename, coder):
    """Process a single image file.
    Args:
      filename: string, path to an image file e.g., '/path/to/example.JPG'.
      coder: instance of ImageCoder to provide TensorFlow image coding utils.
    Returns:
      image_buffer: string, JPEG encoding of RGB image.
      height: integer, image height in pixels.
      width: integer, image width in pixels.
    """
    # Read the image file.
    image_data = tf.gfile.GFile(filename, 'rb').read()
    # Decode the RGB JPEG.
    image = coder.decode_jpeg(image_data)

    # Check that image converted to RGB
    assert len(image.shape) == 3
    height = image.shape[0]
    width = image.shape[1]
    assert image.shape[2] == 3

    return image_data, height, width

def _convert_to_example(filename, image_buffer,
                        height, width):
    """

    :param filename:
    :param image_buffer:
    :param height:
    :param width:
    :return:
    """

    colorspace = 'RGB'
    channels = 3
    image_format = 'JPEG'.encode('utf-8')
    filename_encoded = filename.encode('utf-8')
    example = image_to_tfexample_only(image_buffer, filename_encoded, image_format, height, width)

    return example


def _process_image_files_batch(coder, thread_index, ranges, name, filenames, num_shards):
    """

    :param coder:
    :param thread_index:
    :param ranges:
    :param name:
    :param filenames:
    :param num_shards:
    :return:
    """
    num_threads = len(ranges)
    assert not num_shards % num_threads
    num_shards_per_batch = int(num_shards / num_threads)

    shard_ranges = np.linspace(ranges[thread_index][0],
                               ranges[thread_index][1],
                               num_shards_per_batch + 1).astype(int)
    num_files_in_thread = ranges[thread_index][1] - ranges[thread_index][0]

    counter = 0
    for s in xrange(num_shards_per_batch):
        # Generate a sharded version of the file name, e.g. 'train-00002-of-00010'
        shard = thread_index * num_shards_per_batch + s
        output_filename = '%s-%.5d-of-%.5d' % (name, shard, num_shards)
        output_file = os.path.join(FLAGS.output_directory, output_filename)
        writer = tf.python_io.TFRecordWriter(output_file)

        shard_counter = 0
        files_in_shard = np.arange(shard_ranges[s], shard_ranges[s + 1], dtype=int)
        for i in files_in_shard:
            filename = filenames[i]

            image_buffer, height, width = _process_image(filename, coder)

            example = _convert_to_example(filename, image_buffer,
                                          height, width)
            writer.write(example.SerializeToString())
            shard_counter += 1
            counter += 1

            if not counter % 1000:
                print('%s [thread %d]: Processed %d of %d images in thread batch.' %
                      (datetime.now(), thread_index, counter, num_files_in_thread))
                sys.stdout.flush()

        writer.close()
        print('%s [thread %d]: Wrote %d images to %s' %
              (datetime.now(), thread_index, shard_counter, output_file))
        sys.stdout.flush()
        shard_counter = 0
    print('%s [thread %d]: Wrote %d images to %d shards.' %
          (datetime.now(), thread_index, counter, num_files_in_thread))
    sys.stdout.flush()

def _process_image_files(name, filenames, num_shards):
    """

    :param name:
    :param filenames:
    :param num_shards:
    :return:
    """
    # Bread images into batches
    spacing = np.linspace(0, len(filenames), FLAGS.num_threads + 1).astype(np.int)

    ranges = []
    for i in xrange(len(spacing) - 1):
        ranges.append([spacing[i], spacing[i + 1]])

    # Launch a thread for each batch.
    print('Launching %d threads for spacings: %s' % (FLAGS.num_threads, ranges))
    sys.stdout.flush()

    # Create a mechanism for monitoring when all threads are finished.
    coord = tf.train.Coordinator()

    # Create a generic TensorFlow-based utility for converting all image codings.
    coder = ImageCoder()

    threads = []
    for thread_index in xrange(len(ranges)):
        args = (coder, thread_index, ranges, name, filenames, num_shards)
        t = threading.Thread(target=_process_image_files_batch, args=args)
        t.start()
        threads.append(t)

    # Wait for all the threads to terminate.
    coord.join(threads)
    print('%s: Finished writing all %d images in data set.' %
          (datetime.now(), len(filenames)))
    sys.stdout.flush()


def _process_dataset(name, directory, num_shards):
    """

    :param name:
    :param directory:
    :param num_shards:
    :return:
    """

    filenames = _find_image_files(directory)

    _process_image_files(name, filenames, num_shards)


def main(unused_argv):
    """

    :return:
    """
    _process_dataset('train', FLAGS.train_directory, FLAGS.train_shards)


if __name__ == '__main__':
    tf.app.run()
