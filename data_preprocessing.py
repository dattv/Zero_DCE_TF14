"""

"""
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import tensorflow as tf
import numpy as np
from tensorflow.python.ops import control_flow_ops
import functools

def _crop_frame(image, target_height, target_with):
    """

    :param image:
    :param target_with:
    :param target_height:
    :return:
    """

    image = tf.image.random_crop(image, [target_height, target_with, 3], name='random_crop_frame')
    # image = tf.image.crop_and_resize(image, boxes, name='crop_frame')
    return image

def _random_crop_video(video, target_width, target_height,
                       seed=None):
    """

    :param image:
    :param target_width:
    :param target_height:
    :return:
    """
    video_shape = tf.shape(video)

    return tf.random_crop(video, (video_shape[0], target_height, target_width, video_shape[-1]))

def _flip_left_right_img(img):
    """

    :param img:
    :return:
    """

    img = tf.image.flip_left_right(img)
    return img

def _flip_up_down_img(img):
    """

    :param img:
    :return:
    """
    img = tf.image.flip_up_down(img)
    return img

def _random_horizontal_flip(video,
                            seed=None):
    """

    :param video:
    :param bboxes:
    :param seed:
    :return:
    """
    with tf.name_scope('RandomHorizontalFlip', values=[video]):
        # random variable defining whether to do flip or not
        generator_func = functools.partial(tf.random_uniform, [], seed=seed)
        do_a_flip_random = tf.greater(generator_func(), 0.5)

        # flip image
        video = tf.cond(do_a_flip_random, lambda: _flip_left_right_img(video), lambda: video)

    return video

def _random_up_down_flip(img, seed=None):
    """

    :param img:
    :param seed:
    :return:
    """
    with tf.name_scope('RandomHorizontalFlip', values=[img]):
        # random variable defining whether to do flip or not
        generator_func = functools.partial(tf.random_uniform, [], seed=seed)
        do_a_flip_random = tf.greater(generator_func(), 0.5)

        # flip image
        img = tf.cond(do_a_flip_random, lambda: _flip_up_down_img(img), lambda: img)

    return img

def apply_with_random_selector(x, func, num_cases):
    """Computes func(x, sel), with sel sampled from [0...num_cases-1].

    Args:
      x: input Tensor.
      func: Python function to apply.
      num_cases: Python int32, number of cases to sample sel from.

    Returns:
      The result of func(x, sel), where func receives the value of the
      selector as a python integer, but sel is sampled dynamically.
    """
    sel = tf.random_uniform([], maxval=num_cases, dtype=tf.int32)
    # Pass the real x only to one of the func calls.
    return control_flow_ops.merge([
        func(control_flow_ops.switch(x, tf.equal(sel, case))[1], case)
        for case in range(num_cases)])[0]


def distort_color(image, color_ordering=0, fast_mode=True, scope=None):
    """Distort the color of a Tensor image.

    Each color distortion is non-commutative and thus ordering of the color ops
    matters. Ideally we would randomly permute the ordering of the color ops.
    Rather then adding that level of complication, we select a distinct ordering
    of color ops for each preprocessing thread.

    Args:
      image: 3-D Tensor containing single image in [0, 255].
      color_ordering: Python int, a type of distortion (valid values: 0-3).
      fast_mode: Avoids slower ops (random_hue and random_contrast)
      scope: Optional scope for name_scope.
    Returns:
      3-D Tensor color-distorted image on range [0, 1]
    Raises:
      ValueError: if color_ordering not in [0, 3]
    """
    with tf.name_scope(scope, 'distort_color', [image]):
        if fast_mode:
            if color_ordering == 0:
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            else:
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
        else:
            if color_ordering == 0:
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_hue(image, max_delta=0.2)
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
            elif color_ordering == 1:
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
                image = tf.image.random_hue(image, max_delta=0.2)
            elif color_ordering == 2:
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
                image = tf.image.random_hue(image, max_delta=0.2)
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            elif color_ordering == 3:
                image = tf.image.random_hue(image, max_delta=0.2)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
            else:
                raise ValueError('color_ordering must be in [0, 3]')

        # The random_* ops do not necessarily clamp.
        return tf.clip_by_value(image, 0.0, 1.0)

def train_data_preprocess(target_height, target_width):
    """

    :param input_height:
    :param input_width:
    :return:
    """

    def wraper(image):
        """

        :param image:
        :param label:
        :return:
        """

        # RANDOMLY FLIP LEFT RIGHT
        # image = _random_horizontal_flip(image) # DO NOT USE THIS --> NaN

        # RANDOMLY FLIP UP DOWN
        # image = _random_up_down_flip(image) # DO NOT USE THIS --> NaN

        # RANDOMLY CROP TO SPECIFIC SIZE
        # image = _random_crop_video(image, target_width=target_width, target_height=target_height) # DO NOT USE THIS --> NaN
        image = tf.image.resize_bilinear(image, size=[target_height, target_width])

        return image

    return wraper
