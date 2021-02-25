"""

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import sys

np.set_printoptions(threshold=sys.maxsize)


def kernel_left(shape=(3, 3), dtype=np.float32):
    """

    :return:
    """
    kernel = np.asarray([[0, 0, 0], [-1, 1, 0], [0, 0, 0]], dtype=np.float32)
    kernel = np.expand_dims(
        np.expand_dims(
            kernel, axis=-1
        ), axis=-1
    )
    return kernel


def kernel_right(shape=(3, 3), dtype=np.float32):
    """

    :param shape:
    :param dtype:
    :return:
    """
    kernel = np.asarray([[0, 0, 0], [0, 1, -1], [0, 0, 0]])
    kernel = np.expand_dims(
        np.expand_dims(
            kernel, axis=-1
        ), axis=-1
    )
    return kernel


def kernel_up(shape=(3, 3), dtype=np.float32):
    """

    :param shape:
    :param dtype:
    :return:
    """
    kernel = np.asarray([[0, -1, 0], [0, 1, 0], [0, 0, 0]])
    kernel = np.expand_dims(
        np.expand_dims(
            kernel, axis=-1
        ), axis=-1
    )
    return kernel


def kernel_down(shape=(3, 3), dtype=np.float32):
    """

    :param shape:
    :param dtype:
    :return:
    """
    kernel = np.asarray([[0, 0, 0], [0, 1, 0], [0, -1, 0]])
    kernel = np.expand_dims(
        np.expand_dims(
            kernel, axis=-1
        ), axis=-1
    )
    return kernel


class spa_loss(tf.keras.losses.Loss):
    """

    """

    def __init__(self):
        super(spa_loss, self).__init__()

        self.w = 1.
        self.kernel_left = kernel_left

        self.kernel_right = kernel_right

        self.kernel_up = kernel_up

        self.kernel_down = kernel_down

        self.pool = tf.keras.layers.AveragePooling2D(pool_size=[4, 4])
        print('spa_loss')

    def call(self, y_true, y_pred):
        enhance_img, _ = tf.split(y_pred, [3, 24], axis=-1)

        batch_size, height, width, n_channel = y_true.shape

        org_mean = tf.reduce_mean(y_true, axis=-1, keepdims=True)
        enhance_mean = tf.reduce_mean(enhance_img, axis=-1, keepdims=True)
        #
        org_pool = self.pool(org_mean)
        enhance_pool = self.pool(enhance_mean)

        D_org_left = tf.keras.layers.Conv2D(
            kernel_initializer=self.kernel_left, padding='same', kernel_size=(3, 3), filters=1
        )(org_pool)

        D_org_right = tf.keras.layers.Conv2D(
            kernel_initializer=self.kernel_right, padding='same', kernel_size=(3, 3), filters=1
        )(org_pool)

        D_org_up = tf.keras.layers.Conv2D(
            kernel_initializer=self.kernel_up, padding='same', kernel_size=(3, 3), filters=1
        )(org_pool)

        D_org_down = tf.keras.layers.Conv2D(
            kernel_initializer=self.kernel_down, padding='same', kernel_size=(3, 3), filters=1
        )(org_pool)

        D_enhance_letf = tf.keras.layers.Conv2D(
            kernel_initializer=self.kernel_left, padding='same', kernel_size=(3, 3), filters=1
        )(enhance_pool)

        D_enhance_right = tf.keras.layers.Conv2D(
            kernel_initializer=self.kernel_right, padding='same', kernel_size=(3, 3), filters=1
        )(enhance_pool)

        D_enhance_up = tf.keras.layers.Conv2D(
            kernel_initializer=self.kernel_up, padding='same', kernel_size=(3, 3), filters=1
        )(enhance_pool)

        D_enhance_down = tf.keras.layers.Conv2D(
            kernel_initializer=self.kernel_down, padding='same', kernel_size=(3, 3), filters=1
        )(enhance_pool)

        D_left = tf.pow(D_org_left - D_enhance_letf, 2)
        D_right = tf.pow(D_org_right - D_enhance_right, 2)
        D_up = tf.pow(D_org_up - D_enhance_up, 2)
        D_down = tf.pow(D_org_down - D_enhance_down, 2)
        E = (D_left + D_right + D_up + D_down)
        return E


class color_loss(tf.keras.losses.Loss):
    """
    
    """

    def __init__(self):
        super(color_loss, self).__init__()
        print("color_loss")

    def call(self, y_true, y_pred):
        # y_true = tf.Print(y_true, [y_true], message='y_true', summarize=100)
        enhance_img, _ = tf.split(y_pred, [3, 24], axis=-1)
        mean_rgb = tf.reduce_mean(enhance_img, axis=[1, 2], keepdims=True)
        [mr, mg, mb] = tf.unstack(mean_rgb, axis=-1)
        Drg2 = tf.pow(mr - mg, 2)
        Drb2 = tf.pow(mr - mb, 2)
        Dgb2 = tf.pow(mb - mg, 2)
        return tf.pow(tf.pow(Drg2, 2) + tf.pow(Drb2, 2) + tf.pow(Dgb2, 2), 0.5)


class exp_loss(tf.keras.losses.Loss):
    """

    """

    def __init__(self, patch_size, mean_val):
        super(exp_loss, self).__init__()
        self.pool = tf.keras.layers.AveragePooling2D(pool_size=(patch_size, patch_size))
        self.mean_val = mean_val
        print('exp_loss')

    def call(self, y_true, y_pred):
        enhance_img, _ = tf.split(y_pred, [3, 24], axis=-1)
        x = tf.reduce_mean(enhance_img, axis=-1, keepdims=True)
        mean = self.pool(x)
        d = tf.reduce_mean(tf.pow(mean - self.mean_val, 2))
        return d


class illumination_loss(tf.keras.losses.Loss):
    def __init__(self, TVLoss_weight=1):
        super(illumination_loss, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def call(self, y_true, y_pred):
        _, A = tf.split(y_pred, [3, 24], axis=-1)

        shape_A = tf.shape(A)
        batch_size = tf.cast(shape_A[0], tf.float32)
        h_x = shape_A[1]  # get height of output tensor
        w_x = shape_A[2]  # get width of output tensor

        count_h = tf.cast((h_x - 1) * w_x, dtype=tf.float32)
        count_w = tf.cast((w_x - 1) * h_x, dtype=tf.float32)

        h_tv = tf.reduce_sum(
            tf.multiply(
                A[:, 1, :, :] - A[:, h_x - 1, :, :],
                A[:, 1, :, :] - A[:, h_x - 1, :, :]
            )
        )

        w_tv = tf.reduce_sum(
            tf.multiply(
                A[:, :, 1, :] - A[:, :, w_x - 1, :],
                A[:, :, 1, :] - A[:, :, w_x - 1, :]
            )
        )

        return self.TVLoss_weight * 2. * (h_tv / count_h + w_tv / count_w) / batch_size


class total_loss(tf.keras.losses.Loss):
    def __init__(self):
        super(total_loss, self).__init__()
        self.loss_TV = illumination_loss()
        self.loss_spa = spa_loss()
        self.loss_col = color_loss()
        self.loss_exp = exp_loss(16, 0.6)

    def call(self, y_true, y_pred):
        img_lowlight, A = y_pred
        r1, r2, r3, r4, r5, r6, r7, r8 = A[:, :, :, :3], A[:, :, :, 3:6], A[:, :, :, 6:9], A[:, :, :, 9:12], A[:, :, :,
                                                                                                             12:15], A[
                                                                                                                     :,
                                                                                                                     :,
                                                                                                                     :,
                                                                                                                     15:18], A[
                                                                                                                             :,
                                                                                                                             :,
                                                                                                                             :,
                                                                                                                             18:21], A[
                                                                                                                                     :,
                                                                                                                                     :,
                                                                                                                                     :,
                                                                                                                                     21:24]
        x = img_lowlight + r1 * (tf.pow(img_lowlight, 2) - img_lowlight)
        x = x + r2 * (tf.pow(x, 2) - x)
        x = x + r3 * (tf.pow(x, 2) - x)
        enhanced_image_1 = x + r4 * (tf.pow(x, 2) - x)
        x = enhanced_image_1 + r5 * (tf.pow(enhanced_image_1, 2) - enhanced_image_1)
        x = x + r6 * (tf.pow(x, 2) - x)
        x = x + r7 * (tf.pow(x, 2) - x)
        enhance_image = x + r8 * (tf.pow(x, 2) - x)

        # loss1 = 100. * self.loss_TV(y_true, y_pred)
        # loss2 = tf.reduce_mean(self.loss_spa(y_true, y_pred))
        # loss3 = 5 * tf.reduce_mean(self.loss_col(y_true, y_pred))
        # loss4 = 10 * tf.reduce_mean(self.loss_exp(y_true, y_pred))
        #
        # loss1 = tf.Print(loss1, [loss1], message='loss_TV')
        # loss2 = tf.Print(loss2, [loss2], message='loss_spa')
        # loss3 = tf.Print(loss3, [loss3], message='loss_col')
        # loss4 = tf.Print(loss4, [loss4], message='loss_exp')
        return enhance_image


pool = tf.keras.layers.AveragePooling2D(pool_size=(4, 4))
pool_16 = tf.keras.layers.AveragePooling2D(pool_size=(16, 16))
conv2D_left = tf.keras.layers.Conv2D(
    kernel_initializer=kernel_left, padding='same', kernel_size=(3, 3), filters=1, use_bias=False
)
conv2D_right = tf.keras.layers.Conv2D(
    kernel_initializer=kernel_right, padding='same', kernel_size=(3, 3), filters=1, use_bias=False
)
conv2D_up = tf.keras.layers.Conv2D(
    kernel_initializer=kernel_up, padding='same', kernel_size=(3, 3), filters=1, use_bias=False
)
conv2D_down = tf.keras.layers.Conv2D(
    kernel_initializer=kernel_down, padding='same', kernel_size=(3, 3), filters=1, use_bias=False
)


def dce_loss(input, custom_layer, outputs):
    """

    :param y_true:
    :param y_pred:
    :return:
    """

    def illumination_loss(A):
        """

        :param A:
        :return:
        """
        TVLoss_weight = 1.0
        shape_A = tf.shape(A)
        batch_size = tf.cast(shape_A[0], tf.float32)
        h_x = shape_A[1]  # get height of output tensor
        w_x = shape_A[2]  # get width of output tensor

        count_h = tf.cast(
            (h_x - 1) * w_x, dtype=tf.float32
        )
        count_w = tf.cast(
            (w_x - 1) * h_x,
            dtype=tf.float32
        )

        h_tv = tf.reduce_sum(
            tf.pow(
                A[:, 1, :, :] - A[:, h_x - 1, :, :], 2
            )
        )

        w_tv = tf.reduce_sum(
            tf.pow(
                A[:, :, 1, :] - A[:, :, w_x - 1, :], 2
            )
        )

        return tf.multiply(
            tf.multiply(TVLoss_weight, 2.), tf.divide((tf.divide(h_tv, count_h) + tf.divide(w_tv, count_w)), batch_size)
        )

    def spa_loss(enchance_image, lowlight_image):
        """

        :param enchance_image:
        :param lowlight_image:
        :return:
        """
        enchance_mean = tf.reduce_mean(enchance_image, axis=-1, keepdims=True)
        lowlight_mean = tf.reduce_mean(lowlight_image, axis=-1, keepdims=True)

        enchance_pool = pool(enchance_mean)
        lowlight_pool = pool(lowlight_mean)

        D_enhance_left = conv2D_left(enchance_pool)
        D_enhance_right = conv2D_right(enchance_pool)
        D_enhance_up = conv2D_up(enchance_pool)
        D_enhance_down = conv2D_down(enchance_pool)

        D_lowlight_left = conv2D_left(lowlight_pool)
        D_lowlight_right = conv2D_right(lowlight_pool)
        D_lowlight_up = conv2D_up(lowlight_pool)
        D_lowlight_down = conv2D_down(lowlight_pool)

        D_left = tf.pow(D_enhance_left - D_lowlight_left, 2)
        D_right = tf.pow(D_enhance_right - D_lowlight_right, 2)
        D_up = tf.pow(D_enhance_up - D_lowlight_up, 2)
        D_down = tf.pow(D_enhance_down - D_lowlight_down, 2)

        return D_left + D_right + D_up + D_down

    def color_loss(enhance_image):
        """

        :param enhance_image:
        :return:
        """
        mean_rgb = tf.reduce_mean(enhance_image, axis=[1, 2], keepdims=True)
        mr, mg, mb = tf.split(mean_rgb, [1, 1, 1], axis=-1)

        Drg = tf.pow(mr - mg, 2)
        Drb = tf.pow(mr - mb, 2)
        Dgb = tf.pow(mb - mg, 2)

        return tf.pow(
            tf.pow(Drg, 2) + tf.pow(Drb, 2) + tf.pow(Dgb, 2),
            0.5
        )

    def exp_loss(enhance_image, patch_size=16, mean_val=0.6):
        """

        :param enhance_image:
        :param patch_size:
        :param mean_val:
        :return:
        """
        enhance_image = tf.reduce_mean(enhance_image, axis=-1, keepdims=True)
        mean = pool_16(enhance_image)
        d = tf.reduce_mean(
            tf.pow(mean - mean_val, 2)
        )
        return d

    def wraper(y_true=None, y_pred=None):
        """

        :param y_true:
        :param y_pred:
        :return:
        """

        lowlight_image = input
        enhance_image = custom_layer([lowlight_image, outputs])
        A = outputs
        # [lowlight_image, enhance_image, A] = tf.split(outputs, [3, 3, 24], axis=-1)

        loss1 = 200. * tf.keras.layers.Lambda(
            illumination_loss, name='Illumination_Loss'
        )(A)

        def wraper_spa_loss(inputs):
            """

            :param inputs: [enhance_image, lowlight_image]
            :return:
            """
            enhance_image, lowlight_image = inputs

            return spa_loss(enhance_image, lowlight_image)

        loss2 = tf.reduce_mean(
            tf.keras.layers.Lambda(wraper_spa_loss, name='SPA_Loss')([enhance_image, lowlight_image])
        )

        loss3 = 5. * tf.reduce_mean(
            tf.keras.layers.Lambda(color_loss, name='Color_Loss')(enhance_image)
        )
        loss4 = 10. * tf.reduce_mean(
            tf.keras.layers.Lambda(exp_loss, name='Exp_Loss')(enhance_image)
        )

        return loss1, loss2, loss3, loss4

    return wraper
