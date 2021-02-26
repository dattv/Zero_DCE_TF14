"""

"""
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os
import tensorflow as tf
import yaml

from nets import dce_net
def convert_to_frozen(config):
    """

    :param output_file:
    :return:
    """
    frozen_file_name = config['DEPLOY']['FROZEN_FILE']
    height = config['INPUT_HEIGHT']
    width = config['INPUT_WIDTH']
    n_channel = config['N_CHANNEL']

    tf.keras.backend.clear_session()
    tf.keras.backend.set_learning_phase(0)

    input_shape = (height, width, n_channel)

    model = dce_net.dce_inference(input_shape=input_shape, training=False)



if __name__ == '__main__':
        file_config = './configs/config.yaml'

        with open(file_config) as file:
            cfg = yaml.full_load(file)

        convert_to_frozen(config=cfg)