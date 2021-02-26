"""

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2 as cv
import numpy as np
import skimage
from PIL import Image
import tensorflow as tf
from nets.dce_net import enhance_np
data = skimage.data

def colormap_jet(img):
    min = img.min()
    if min < 0:
        img -= min
    img *= (255.0/img.max())
    temp = cv.applyColorMap(np.uint8(img), 2)
    out = temp # cv.cvtColor(temp, cv.COLOR_BGR2RGB)
    return out


class conv2d_callback(tf.keras.callbacks.Callback):
    def __init__(self, log_dir='./logs/tmp/', feed_inputs_display=None):
        super(customModelCheckpoint, self).__init__()
        self.seen = 0
        self.feed_inputs_display = feed_inputs_display
        self.writer = tf.summary.FileWriter(log_dir)

    def custom_set_feed_input_to_display(self, feed_inputs_display):
        self.feed_inputs_display = feed_inputs_display

    # copied from the above answers;
    def make_image(self, numpy_img):
        height, width, channel = numpy_img.shape
        image = Image.fromarray(numpy_img)
        import io
        output = io.BytesIO()
        image.save(output, format='PNG')
        image_string = output.getvalue()
        output.close()
        return tf.Summary.Image(height=height, width=width, colorspace=channel, encoded_image_string=image_string)

    # A callback has access to its associated model through the class property self.model.
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.seen += 1
        if epoch % 1 == 0:  # every 200 iterations or batches, plot the costumed images using TensorBorad;
            summary_str = []
            for i in range(len(self.feed_inputs_display)):
                feature, _, _ = self.feed_inputs_display[i]

                summary_str.append(tf.Summary.Value(tag='Compare{}/lowlight_img'.format(i), image=self.make_image(
                    feature)))  # function colormap_jet(), defined above;

                lowlight_img = np.expand_dims(feature, axis=0)/255.0

                output_np = tf.keras.backend.get_session().run(
                            self.model.output, feed_dict={self.model.input: lowlight_img}
                        )

                enhance_img = enhance_np([lowlight_img, output_np])
                enhance_img = np.squeeze(enhance_img, axis=0)

                #
                min = enhance_img.min()
                if min < 0:
                    enhance_img -= min
                enhance_img *= (255.0 / enhance_img.max())
                enhance_img = enhance_img.astype(np.uint8)
                # enhance_img = cv.cvtColor(enhance_img, cv.COLOR_BGR2RGB)

                summary_str.append(tf.Summary.Value(tag='Compare{}/enhanced_img'.format(i), image=self.make_image(
                    enhance_img)))
                #

                layer_names = [layer.name for layer in self.model.layers if 'conv' in layer.name]
                for name in layer_names:
                    disp_pred, output_np = tf.keras.backend.get_session().run(
                            [self.model.get_layer(name).output, self.model.output], feed_dict={self.model.input: lowlight_img}
                        )
                    disp_pred = np.squeeze(disp_pred, axis=0)
                    # disp_pred = np.squeeze(self.model.predict_on_batch(feature), axis = 0)

                    disp_size = disp_pred.shape[-1]
                    for j in range(disp_size):
                        gray_img = disp_pred[:, :, j]
                        gray_img = np.expand_dims(gray_img, axis=-1)


                        summary_str.append(
                            tf.Summary.Value(tag='{}/img{}/out{}'.format(name, i, j), image=self.make_image(
                                colormap_jet(gray_img))))

            self.writer.add_summary(tf.Summary(value=summary_str), global_step=self.seen)
