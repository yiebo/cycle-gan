import tensorflow as tf


def weight_variable(shape):
    weight = tf.get_variable("weight", shape, initializer=tf.truncated_normal_initializer(stddev=0.01))
    return weight


def bias_variable(shape):
    bias = tf.get_variable("bias", shape, initializer=tf.truncated_normal_initializer(stddev=0.01))
    return bias


def conv2d_dilated(x, k, co, rate=1, bn=True, padding='SAME'):
    if isinstance(k, list):
        W = weight_variable([k[0], k[1], shape(x)[-1], co])
    else:
        W = weight_variable([k, k, shape(x)[-1], co])
    B = bias_variable([co])
    L = tf.nn.atrous_conv2d(x, W, rate=rate, padding=padding) + B
    if bn:
        L = batch_norm(L)
    return L


def deconv2d(x, W, shape, s=1, padding='SAME'):
    deconv = tf.nn.conv2d_transpose(value=x, filter=W, output_shape=shape,
                                    strides=[1, s, s, 1], padding='SAME')
    return tf.reshape(deconv, shape)


class Reverse_Gradient_Builder(object):
    def __init__(self):
        self.num_calls = 0

    def __call__(self, x, l=1.0):
        grad_name = "reverse_gradient{}".format(self.num_calls)

        @ops.RegisterGradient(grad_name)
        def _flip_gradients(op, grad):
            return [tf.negative(grad) * l]

        g = tf.get_default_graph()
        with g.gradient_override_map({"Identity": grad_name}):
            y = tf.identity(x)

        self.num_calls += 1
        return y


reverse_gradient = Reverse_Gradient_Builder()


def batch_norm(x, epsilon=1e-5):
    return tf.layers.batch_normalization(inputs=x, epsilon=epsilon)


def shape(tensor):
    return tensor.get_shape().as_list()


def concat_y(x, y):
    with tf.name_scope("concat_y"):
        yb = tf.tile(tf.reshape(y, [-1, 1, 1, shape(y)[-1]]),
                     [1, tf.shape(x)[1], tf.shape(x)[2], 1])
        xy = tf.concat([x, yb], 3)
    return xy


def sub_pixel_conv(x, filters, kernel=2, stride=1, padding='SAME', uprate=2):
    x = tf.layers.conv2d(inputs=x, filters=filters * uprate**2,
                         kernel_size=kernel, strides=stride,
                         padding=padding)
    x = tf.depth_to_space(input=x, block_size=uprate)
    return x
