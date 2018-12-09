import tensorflow as tf
from ops import *

TRAINING = True


def u_block(x, filters):

    with tf.variable_scope('layer_1'):
        x = tf.layers.conv2d(inputs=x, filters=filters,
                             kernel_size=3, strides=1, padding='SAME')
        x = tf.layers.batch_normalization(x, training=TRAINING)
        x = tf.nn.relu(x)

    with tf.variable_scope('layer_2'):
        x = tf.layers.conv2d(inputs=x, filters=filters,
                             kernel_size=3, strides=1, padding='SAME')
        x = tf.layers.batch_normalization(x, training=TRAINING)
        x = tf.nn.relu(x)

    return x


def u_net(x, name):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        depth = [64, 128, 256, 512]
        skip_connections = []
        for idx, filters in enumerate(depth):
            with tf.variable_scope('down_block{}'.format(idx)):
                x = u_block(x, filters)
                skip_connections.append(x)
                x = tf.layers.max_pooling2d(inputs=x, pool_size=2, strides=2, padding='SAME')

        with tf.variable_scope('conv_block'):
            x = u_block(x, 1024)
        depth = [512, 256, 128, 64]
        skip_connections.reverse()

        for idx, (filters, skip_connection) in enumerate(zip(depth, skip_connections)):
            with tf.variable_scope('up_block{}'.format(idx)):
                with tf.variable_scope('sub_pixel'):
                    x = sub_pixel_conv(x, filters=filters, kernel_size=3, uprate=2)
                    x = tf.layers.batch_normalization(x, training=TRAINING)
                    x = tf.nn.relu(x)
                    x = tf.concat([x, skip_connection], 3)
                x = u_block(x, filters)

        with tf.variable_scope('final_block'):
            x = tf.layers.conv2d(inputs=x, filters=3,
                                 kernel_size=1, strides=1, padding='SAME')
            x = tf.nn.relu(x)
    return x


def discriminator(x, name, reuse=False):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        depth = [64, 128, 256, 512, 1024]
        for idx, filters in enumerate(depth):
            with tf.variable_scope('conv_{}'.format(idx)):
                x = tf.layers.conv2d(inputs=x, filters=filters,
                                     kernel_size=3, strides=1, padding='SAME')
                x = tf.layers.batch_normalization(x, training=TRAINING)
                x = tf.layers.max_pooling2d(inputs=x, pool_size=2, strides=2, padding='SAME')

        with tf.variable_scope('fc_0'):
            x = tf.layers.dense(inputs=x, units=256)
            x = tf.layers.batch_normalization(x, training=TRAINING)
            x = tf.nn.relu(x)

        with tf.variable_scope('fc_1'):
            x = tf.layers.dense(inputs=x, units=1)
    return x
# 218x178
