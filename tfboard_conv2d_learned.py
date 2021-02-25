"""

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2 as cv
import numpy as np
import skimage
import tensorflow as tf
from nets.dce_net import enhance_np
data = skimage.data


def colormap_jet(img):
    min = img.min()
    if min < 0:
        img -= min
    img *= (255.0/img.max())
    temp = cv.applyColorMap(np.uint8(img), 2)
    out = cv.cvtColor(temp, cv.COLOR_BGR2RGB)
    return out


def summary_conv2d_ouputs(model):
    """

    :param model:
    :return:
    """
    img_test = './data/test_data/LIME/1.pmp'
    img = colormap_jet(cv.imread(img_test))
    img = cv.resize(img, (256, 256))

    conv2d_name = [layer.name for layer in model.layers if 'conv' in layer.name]
    input = model.inputs
    for name in conv2d_name:
        output_layer = model.get_layer(name)
        output_tensor = output_layer.output
        gray = tf.unstack(output_tensor, axis=-1)
        for i, tensor in enumerate(gray):
            tensor = tf.expand_dims(tensor, axis=-1)
            test_model = tf.keras.Model(inputs=input, outputs=tensor)
            test_model(img)

            # image = Image.fromarray(tensor)
            # output = io.BytesIO()
            # image.save(output, format='PNG')
            # image_string = output.getvalue()
            # output.close()
            tf.summary.image(name=name + '{}'.format(i),
                             tensor=tensor,
                             max_outputs=len(gray))

        print('djkfd')


# class TensorBoardImage(tf.keras.callbacks.Callback):
#     def __init__(self, model, logsdir):
#         super().__init__()
#         self.model = model
#         self.logsdir = logsdir
#
#
#     def on_epoch_end(self, epoch, logs={}):
#         summary_conv2d_ouputs(self.model)

def make_image(tensor):
    """
    Convert an numpy representation image to Image protobuf.
    Copied from https://github.com/lanpa/tensorboard-pytorch/
    """
    from PIL import Image
    height, width, channel = tensor.shape
    image = Image.fromarray(tensor)
    import io
    output = io.BytesIO()
    image.save(output, format='PNG')
    image_string = output.getvalue()
    output.close()
    return tf.Summary.Image(height=height,
                            width=width,
                            colorspace=channel,
                            encoded_image_string=image_string)


class TensorBoardImage(tf.keras.callbacks.Callback):
    def __init__(self, tag, model):
        super(TensorBoardImage, self).__init__()
        self.tag = tag
        self.model = model

    def on_epoch_end(self, epoch, logs={}):
        # Load image
        img = data.astronaut()
        # Do something to the image
        img = (255 * skimage.util.random_noise(img)).astype('uint8')

        image = make_image(img)
        summary = tf.Summary(value=[tf.Summary.Value(tag=self.tag, image=image)])
        writer = tf.summary.FileWriter('./logs')
        writer.add_summary(summary, epoch)
        writer.close()

        return


# make the 1 channel input image or disparity map look good within this color map. This function is not necessary for this Tensorboard problem shown as above. Just a function used in my own research project.
def colormap_jet(img):
    min = img.min()
    if min < 0:
        img -= min
    img *= (255.0/img.max())
    temp = cv.applyColorMap(np.uint8(img), 2)
    out = temp # cv.cvtColor(temp, cv.COLOR_BGR2RGB)
    return out


class customModelCheckpoint(tf.keras.callbacks.Callback):
    def __init__(self, log_dir='./logs/tmp/', feed_inputs_display=None):
        super(customModelCheckpoint, self).__init__()
        self.seen = 0
        self.feed_inputs_display = feed_inputs_display
        self.writer = tf.summary.FileWriter(log_dir)

    # this function will return the feeding data for TensorBoard visualization;
    # arguments:
    #  * feed_input_display : [(input_yourModelNeed, left_image, disparity_gt ), ..., (input_yourModelNeed, left_image, disparity_gt), ...], i.e., the list of tuples of Numpy Arrays what your model needs as input and what you want to display using TensorBoard. Note: you have to feed the input to the model with feed_dict, if you want to get and display the output of your model.
    def custom_set_feed_input_to_display(self, feed_inputs_display):
        self.feed_inputs_display = feed_inputs_display

    # copied from the above answers;
    def make_image(self, numpy_img):
        from PIL import Image
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
                # summary_str.append(
                #     tf.Summary.Value(tag='plot/disp/{}'.format(i), image=self.make_image(
                #         colormap_jet(disp_pred))))

            self.writer.add_summary(tf.Summary(value=summary_str), global_step=self.seen)
