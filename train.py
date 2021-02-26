"""

"""
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import os
import yaml
import tensorflow as tf
from nets.dce_net import dce_inference, enhance_layer
from losser import dce_loss
from input_pipe_line import data_loader
from data_preprocessing import train_data_preprocess
from tfboard_conv2d_learned import customModelCheckpoint
import cv2 as cv


def train(config=None):
    """

    :param tfrecord_dir:
    :param output_dir:
    :return:
    """
    assert config is not None, 'need to pass config'

    tfrecord_dir = config['DATA_DIR']
    output_dir = config['OUTPUTS']
    log_dir = config['LOG_DIR']
    training = config['SOLVER']['TRAINING']
    height = config['INPUT_HEIGHT']
    width = config['INPUT_WIDTH']
    n_channel = config['N_CHANNEL']
    weight_decay = config['SOLVER']['REGULARIZATION']
    optimizer_name = config['SOLVER']['OPTIMS']
    lr = config['SOLVER']['INIT_LR']
    momentum = config['SOLVER']['MOMENTUM']
    batch_size = config['SOLVER']['BATCH_SIZE']
    epochs = config['SOLVER']['EPOCHS']

    assert tfrecord_dir is not None, 'tfrecord_dir need to be not None'
    assert output_dir is not None, 'output dir need to be not None'

    if os.path.isdir(output_dir) is False:
        os.mkdir(output_dir)

    if os.path.isdir(tfrecord_dir) is False:
        raise Exception('there are not {} dir'.format(tfrecord_dir))

    if os.path.isdir(log_dir) is False:
        os.mkdir(log_dir)

    tf.keras.backend.clear_session()
    graph = tf.get_default_graph()
    session = tf.Session(graph=graph)
    tf.keras.backend.set_session(session)

    train_data = data_loader(mode=tf.estimator.ModeKeys.TRAIN,
                             dataset_dir=tfrecord_dir,
                             preprocess_data=train_data_preprocess(target_height=height, target_width=width),
                             batch_size=batch_size)

    model = dce_inference(
        input_shape=(height, width, n_channel), training=training, regularizer=weight_decay
    )

    print(model.summary())

    opt = tf.keras.optimizers.SGD(learning_rate=lr, momentum=momentum, clipnorm=0.1)
    custom_layer = enhance_layer()
    custom_loss = dce_loss(model.inputs[0], custom_layer, model.outputs[0])()
    model.add_loss(custom_loss[0])
    model.add_loss(custom_loss[1])
    model.add_loss(custom_loss[2])
    model.add_loss(custom_loss[3])
    model.compile(optimizer=opt, loss=[None] * len(model.outputs))

    checkpoint_h5_cb = tf.keras.callbacks.ModelCheckpoint(
        os.path.join(output_dir, "dce_net.h5"), save_weights_only=True
    )

    class CustomCallback(tf.keras.callbacks.Callback):

        def on_epoch_end(self, epoch, logs=None):
            """
            Returns a custom learning rate that decreases as epochs progress.
            """

            tf.summary.scalar('custom_loss[0]', custom_loss[0])
            tf.summary.scalar('custom_loss[1]', custom_loss[1])
            tf.summary.scalar('custom_loss[2]', custom_loss[2])
            tf.summary.scalar('custom_loss[3]', custom_loss[3])

    tfboard_cb = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, histogram_freq=1, write_graph=True, batch_size=batch_size,
        write_images=True, update_freq='epoch', profile_batch=2
    )

    csvlogger = tf.keras.callbacks.CSVLogger(os.path.join(log_dir, 'train.log'))
    saver = tf.train.Saver()

    class save_ckpt_callback(tf.keras.callbacks.Callback):
        def __init__(self, saver, session, save_path):
            super(save_ckpt_callback, self).__init__()
            self.saver = saver
            self.session = session
            self.save_path = save_path
        def on_epoch_end(self, epoch, logs=None):
            saver.save(self.session, self.save_path, global_step=epoch)

    checkpoint_ckpt_cb = save_ckpt_callback(
    saver, tf.keras.backend.get_session(), os.path.join(output_dir, model.name + '.ckpt')
    )

    # output_image_cb = TensorBoardImage(tag='test', model)
    test_folder = './data/test_data/LIME'
    list_file = [os.path.join(test_folder, f) for f in os.listdir(test_folder)]
    img_test = './data/test_data/LIME/1.bmp'

    input_display = []
    for file in list_file:
        img = cv.imread(file)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img = cv.resize(img, (height, width))
        # img = np.expand_dims(img, axis=0)
        input_display.append([img, img, img])

    output_image_cb = customModelCheckpoint(feed_inputs_display=input_display)
    history = model.fit(
        train_data,
        epochs=epochs,
        steps_per_epoch=2002 // batch_size,
        callbacks=[
            checkpoint_h5_cb,
            checkpoint_ckpt_cb,
            output_image_cb,
            # early_stopping_cb,
            tfboard_cb,
            CustomCallback(),
            csvlogger
        ]
    )
    print(history)


if __name__ == '__main__':
    with open('./configs/config.yaml') as file:
        cfg = yaml.full_load(file)
    train(config=cfg)
