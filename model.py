import tensorflow as tf
from ops import *

def u_block(x, filters, training=False):

    with tf.variable_scope('layer_1'):
        x = tf.layers.conv2d(inputs=x, filters=filters,
                             kernel_size=3, strides=1, padding='SAME')
        x = tf.layers.batch_normalization(x, training=training)
        x = tf.nn.relu(x)

    with tf.variable_scope('layer_2'):
        x = tf.layers.conv2d(inputs=x, filters=filters,
                             kernel_size=3, strides=1, padding='SAME')
        x = tf.layers.batch_normalization(x, training=training)
        x = tf.nn.relu(x)

    return x


def u_net(x, name, training=False):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        depth = [64, 128, 256, 512]
        skip_connections = []
        for idx, filters in enumerate(depth):
            with tf.variable_scope(f'down_block{idx}'):
                x = u_block(x, filters, training=training)
                skip_connections.append(x)
                x = tf.layers.max_pooling2d(inputs=x, pool_size=2, strides=2, padding='SAME')

        with tf.variable_scope('conv_block'):
            x = u_block(x, 1024, training=training)

        depth = [512, 256, 128, 64]
        skip_connections.reverse()

        for idx, (filters, skip_connection) in enumerate(zip(depth, skip_connections)):
            with tf.variable_scope(f'up_block{idx}'):
                with tf.variable_scope('sub_pixel'):
                    x = sub_pixel_conv(x, filters=filters, kernel_size=3, uprate=2)
                    x = tf.layers.batch_normalization(x, training=training)
                    x = tf.nn.relu(x)
                    x = tf.concat([x, skip_connection], 3)
                x = u_block(x, filters, training=training)

        with tf.variable_scope('final_block'):
            x = tf.layers.conv2d(inputs=x, filters=3,
                                 kernel_size=1, strides=1, padding='SAME')
            x = tf.nn.relu(x)
    return x


def discriminator(x, name, training=False):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        depth = [64, 128, 256, 512, 1028]
        for idx, filters in enumerate(depth):
            with tf.variable_scope(f'conv{idx}'):
                x = tf.layers.conv2d(inputs=x, filters=filters,
                                     kernel_size=3, strides=1, padding='SAME')
                x = tf.layers.batch_normalization(x, training=training)
                x = tf.nn.leaky_relu(x)
                x = tf.layers.max_pooling2d(inputs=x, pool_size=2, strides=2, padding='SAME')

        with tf.variable_scope('final_conv'):
            x = tf.layers.conv2d(inputs=x, filters=1,
                                 kernel_size=1, strides=1, padding='SAME')

        # with tf.variable_scope('fc_0'):
        #     x = tf.layers.flatten(x)
        #     x = tf.layers.dense(inputs=x, units=256)
        #     # x = tf.layers.batch_normalization(x, training=TRAINING)
        #     x = tf.nn.leaky_relu(x)

        # with tf.variable_scope('fc_1'):
        #     x = tf.layers.dense(inputs=x, units=1)
    return x
# 218x178
