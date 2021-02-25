"""

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.layers import Conv2D, Concatenate
import numpy as np

INITIAL = {'mean': 0., 'std': 0.01}


def enhance_layer():
    """

    :param img_lowlight:
    :param output_tanh:
    :return:
    """

    def wraper(tensors):
        img_lowlight, output_tanh = tensors

        r1, r2, r3, r4, r5, r6, r7, r8 = tf.split(output_tanh, [3, 3, 3, 3, 3, 3, 3, 3], axis=-1)
        x = img_lowlight + r1 * (tf.pow(img_lowlight, 2) - img_lowlight)
        x = x + r2 * (tf.pow(x, 2) - x)
        x = x + r3 * (tf.pow(x, 2) - x)
        enhanced_image_1 = x + r4 * (tf.pow(x, 2) - x)
        x = enhanced_image_1 + r5 * (tf.pow(enhanced_image_1, 2) - enhanced_image_1)
        x = x + r6 * (tf.pow(x, 2) - x)
        x = x + r7 * (tf.pow(x, 2) - x)
        enhance_image = x + r8 * (tf.pow(x, 2) - x)
        return enhance_image

    return tf.keras.layers.Lambda(wraper, name='enhance_image')

def enhance_layer_np():
    def wraper(input):
        """

        :param input:
        :return:
        """
        img_lowlight, output_tanh = input

        r1, r2, r3, r4, r5, r6, r7, r8 = np.split(output_tanh, 8, axis=-1)
        x = img_lowlight + r1 * (np.power(img_lowlight, 2) - img_lowlight)
        x = x + r2 * (np.power(x, 2) - x)
        x = x + r3 * (np.power(x, 2) - x)
        enhanced_image_1 = x + r4 * (np.power(x, 2) - x)
        x = enhanced_image_1 + r5 * (np.power(enhanced_image_1, 2) - enhanced_image_1)
        x = x + r6 * (np.power(x, 2) - x)
        x = x + r7 * (np.power(x, 2) - x)
        enhance_image = x + r8 * (np.power(x, 2) - x)
        return enhance_image
    return wraper

enhance = enhance_layer()
enhance_np = enhance_layer_np()

def dce_inference(input_shape=(256, 256, 3), training=True, regularizer=1.e-7, initial=INITIAL):
    """

    :param input_shape:
    :param training:
    :return:
    """
    assert input_shape is not None, '{} input_shape must not be None'.format(input_shape)
    input_img = Input(shape=input_shape)
    conv1 = Conv2D(32, (3, 3), strides=(1, 1), activation='relu', padding='same',
                   kernel_regularizer=tf.keras.regularizers.l2(regularizer),
                   kernel_initializer=tf.keras.initializers.random_normal(mean=initial['mean'], stddev=initial['std']),
                   use_bias=True,
                   bias_initializer=tf.keras.initializers.random_normal(mean=initial['mean'], stddev=initial['std']))(
        input_img)

    conv2 = Conv2D(32, (3, 3), strides=(1, 1), activation='relu', padding='same',
                   kernel_regularizer=tf.keras.regularizers.l2(regularizer),
                   kernel_initializer=tf.keras.initializers.random_normal(mean=initial['mean'], stddev=initial['std']),
                   use_bias=True,
                   bias_initializer=tf.keras.initializers.random_normal(mean=initial['mean'], stddev=initial['std']))(
        conv1)

    conv3 = Conv2D(32, (3, 3), strides=(1, 1), activation='relu', padding='same',
                   kernel_regularizer=tf.keras.regularizers.l2(regularizer),
                   kernel_initializer=tf.keras.initializers.random_normal(mean=initial['mean'], stddev=initial['std']),
                   use_bias=True,
                   bias_initializer=tf.keras.initializers.random_normal(mean=initial['mean'], stddev=initial['std']))(
        conv2)

    conv4 = Conv2D(32, (3, 3), strides=(1, 1), activation='relu', padding='same',
                   kernel_regularizer=tf.keras.regularizers.l2(regularizer),
                   kernel_initializer=tf.keras.initializers.random_normal(mean=initial['mean'], stddev=initial['std']),
                   use_bias=True,
                   bias_initializer=tf.keras.initializers.random_normal(mean=initial['mean'], stddev=initial['std']))(
        conv3)

    int_con1 = Concatenate(axis=-1)([conv4, conv3])
    conv5 = Conv2D(32, (3, 3), strides=(1, 1), activation='relu', padding='same',
                   kernel_regularizer=tf.keras.regularizers.l2(regularizer),
                   kernel_initializer=tf.keras.initializers.random_normal(mean=initial['mean'], stddev=initial['std']),
                   use_bias=True,
                   bias_initializer=tf.keras.initializers.random_normal(mean=initial['mean'], stddev=initial['std']))(
        int_con1)

    int_con2 = Concatenate(axis=-1)([conv5, conv2])
    conv6 = Conv2D(32, (3, 3), strides=(1, 1), activation='relu', padding='same',
                   kernel_regularizer=tf.keras.regularizers.l2(regularizer),
                   kernel_initializer=tf.keras.initializers.random_normal(mean=initial['mean'], stddev=initial['std']),
                   use_bias=True,
                   bias_initializer=tf.keras.initializers.random_normal(mean=initial['mean'], stddev=initial['std']))(
        int_con2)

    int_con3 = Concatenate(axis=-1)([conv6, conv1])
    x_r = Conv2D(24, (3, 3), strides=(1, 1), activation='tanh', padding='same',
                 kernel_regularizer=tf.keras.regularizers.l2(regularizer),
                 kernel_initializer=tf.keras.initializers.random_normal(mean=initial['mean'], stddev=initial['std']),
                 use_bias=True,
                 bias_initializer=tf.keras.initializers.random_normal(mean=initial['mean'], stddev=initial['std']))(
        int_con3)

    # output = tf.keras.layers.Concatenate(-1)([
    #     tf.keras.layers.Lambda(tf.identity, name='identity')(input_img),
    #     enhance([input_img, x_r]),
    #     x_r,
    # ]
    # )
    model = Model(inputs=input_img, outputs=x_r)
    return model


if __name__ == '__main__':
    model = dce_inference()
    model.build(input_shape=(1, 256, 256, 3))

    input_np = np.random.randint(low=0, high=10, size=[1, 256, 256, 3]).astype(np.float32)
    output = model(input_np)
    print(model.summary())
    # model.save('abc.h5')

    # loaded_model = tf.keras.models.load_model("abc.h5")

    tf.keras.utils.plot_model(model, '{}.png'.format(model.name), show_shapes=True)
