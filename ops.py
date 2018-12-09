import tensorflow as tf
from tensorflow.python.framework import ops


def weight_variable(shape):
    weight = tf.get_variable(name="weight", shape=shape,
                             initializer=tf.glorot_uniform_initializer)
    return weight


def bias_variable(shape):
    bias = tf.get_variable(name="bias", shape=shape,
                           initializer=tf.glorot_uniform_initializer)
    return bias


def conv2d_dilated(x, k, co, rate=1, padding='SAME'):
    if isinstance(k, list):
        W = weight_variable([k[0], k[1], shape(x)[-1], co])
    else:
        W = weight_variable([k, k, shape(x)[-1], co])
    B = bias_variable([co])
    L = tf.nn.atrous_conv2d(x, W, rate=rate, padding=padding) + B
    return L


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


def float32_variable_storage_getter(getter, name, shape=None, dtype=None,
                                    initializer=None, regularizer=None,
                                    trainable=True,
                                    *args, **kwargs):
    """Custom variable getter that forces trainable variables to be stored in
    float32 precision and then casts them to the training precision.
    """
    storage_dtype = tf.float32 if trainable else dtype
    variable = getter(name, shape, dtype=storage_dtype,
                      initializer=initializer, regularizer=regularizer,
                      trainable=trainable,
                      *args, **kwargs)
    if trainable and dtype != tf.float32:
        variable = tf.cast(variable, dtype)
    return variable


def shape(tensor):
    return tensor.get_shape().as_list()


def concat_y(x, y):
    with tf.name_scope("concat_y"):
        yb = tf.tile(tf.reshape(y, [-1, 1, 1, shape(y)[-1]]),
                     [1, tf.shape(x)[1], tf.shape(x)[2], 1])
        xy = tf.concat([x, yb], 3)
    return xy


def sub_pixel_conv(x, filters, kernel_size=2, stride=1, padding='SAME', uprate=2):
    x = tf.layers.conv2d(inputs=x, filters=filters * uprate**2,
                         kernel_size=kernel_size, strides=stride,
                         padding=padding)
    x = tf.depth_to_space(input=x, block_size=uprate)
    return x
