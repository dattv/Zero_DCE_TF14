"""

"""
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import os
import numpy as np
import unittest
import tensorflow as tf
from tensorflow.python.keras.utils.layer_utils import count_params

from nets.dce_net import dce_inference

# ******************************************************************
model = dce_inference()
# ******************************************************************

class test_dce_net(unittest.TestCase):
    """

    """
    def test_output(self):
        """

        :return:
        """
        input = np.random.random_sample(size=[1, 256, 256, 3]).astype(np.float32)
        output = model(input)

    def test_trainable_parameters(self):
        trainable_count = count_params(model.trainable_variables)
        non_trainable_count = count_params(model.non_trainable_variables)

        """
        number of trainable parameters should be 79416 (paper reported)
        https://openaccess.thecvf.com/content_CVPR_2020/papers/Guo_Zero-Reference_Deep_Curve_Estimation_for_Low-Light_Image_Enhancement_CVPR_2020_paper.pdf
        """
        assert trainable_count == 79416, 'number of trainable parameters wrong'

if __name__ == '__main__':
    unittest.main()