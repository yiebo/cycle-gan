import tensorflow as tf
from ops import *
import glob
import tqdm

from read import _parse_function

TRAINING = True


def u_block(x, filter):

    with tf.variable_scope('layer_1'):
        x = tf.layers.conv2d(inputs=x, filters=filter,
                             kernel_size=3, strides=1, padding='SAME')
        # x = tf.layers.batch_normalization(x, training=TRAINING)
        x = tf.nn.relu(x)

    with tf.variable_scope('layer_2'):
        x = tf.layers.conv2d(inputs=x, filters=filter,
                             kernel_size=3, strides=1, padding='SAME')
        x = tf.layers.batch_normalization(x, training=TRAINING)
        x = tf.nn.relu(x)

    return x


def up_conv(x, kernel=2):
    with tf.variable_scope('layer_2'):
        x = tf.layers.conv2d(inputs=x, filters=filter,
                             kernel_size=3, strides=1, padding='SAME')
        x = tf.layers.batch_normalization(x, training=TRAINING)
        x = tf.nn.relu(x)


def u_net(x, name, reuse=False):
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()

        depth = [64, 128, 256, 512]
        skip_connections = []
        for idx, filter in enumerate(depth):
            with tf.variable_scope(f'down_block{idx}'):
                x = u_block(x, filter)
                skip_connections.append(x)
                x = tf.nn.max_pool(input=x, ksize=2, strides=2, padding='SAME')

        with tf.variable_scope(f'conv_block'):
            x = u_block(x, 1024)
        depth = [64, 128, 256, 512]

        for idx, (filter, skip_connection) in enumerate(zip(depth, skip_connections)):
            with tf.variable_scope(f'up_block{idx}'):
                with tf.variable_scope(f'sub_pixel'):
                    x = sub_pixel_conv(x, filter=depth, kernel=3, uprate=2)
                    x = tf.layers.batch_normalization(x, training=TRAINING)
                    x = tf.concat([x, skip_connection])
            x = u_block(x, filter)

        with tf.variable_scope(f'final_block'):
            x = tf.layers.conv2d(inputs=x, filters=filter,
                                 kernel_size=1, strides=1, padding='SAME')
            x = tf.layers.batch_normalization(x, training=TRAINING)
            x = tf.nn.sigmoid(x)
    return x


def discriminator(x, name, reuse=False):
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()

        depth = [64, 128, 256, 512, 1024]
        for idx, filter in enumerate(depth):
            with tf.variable_scope(f'conv_{idx}'):
                x = tf.layers.conv2d(inputs=x, filters=filter,
                                     kernel_size=1, strides=1, padding='SAME')
                x = tf.layers.batch_normalization(x, training=TRAINING)
                x = tf.nn.max_pool(input=x, ksize=2, strides=2, padding='SAME')

        with tf.variable_scope(f'fc_0'):
            x = tf.layers.dense(input=x, units=256)
            x = tf.layers.batch_normalization(x, training=TRAINING)
            x = tf.nn.relu(x)

        with tf.variable_scope(f'fc_1'):
            x = tf.layers.dense(input=x, units=1)
    return x
# 218x178
