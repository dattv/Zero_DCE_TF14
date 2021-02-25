"""

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import numpy as np
import input_pipe_line
import data_preprocessing
import tensorflow as tf
import unittest
import cv2 as cv

class test_input_pipe_line(unittest.TestCase):
    """

    """
    def test_data(self):
        """

        :return:
        """
        height = 256
        width = 256
        batch_size = 4
        folder_name = './test_input_pipe_line_results'
        tfrecord_dir = '/media/dat/68fa98f8-9d03-4c1e-9bdb-c71ea72ab6fa/dat/zero_DCE/data/tfrecord'
        train_data_generator = input_pipe_line.build_dataset(mode=tf.estimator.ModeKeys.TRAIN,
                                   dataset_dir=tfrecord_dir,
                                   preprocess_data=data_preprocessing.train_data_preprocess(target_height=height, target_width=width),
                                   batch_size=4)

        iterator = train_data_generator.make_one_shot_iterator()
        nex_element = iterator.get_next()
        with tf.Session() as sess:
            for i in range(1000):
                images = sess.run(nex_element[0])

                for j in range(len(images)):
                    first_vid = images[j]
                    name = str(j) + str(i) + '.jpg'
                    if os.path.isdir(folder_name) is False:
                        os.mkdir(folder_name)

                    cv.imwrite(folder_name + "/" + name, cv.cvtColor(first_vid, cv.COLOR_RGB2BGR))
                    print(folder_name)

if __name__ == '__main__':
    unittest.main()