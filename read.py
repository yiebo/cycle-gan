import tensorflow as tf


def _parse_function(example_proto):
    features = {'image': tf.FixedLenFeature((), tf.string, default_value="")
                }
    parsed = tf.parse_single_example(example_proto, features)

    # image = tf.image.decode_image(parsed['image'])
    image_raw = tf.decode_raw(parsed['image'], tf.uint8)
    image = tf.cast(image_raw, tf.float32)
    image = tf.reshape(image, [208, 176, 3]) / 255.0
    # image = tf.image.random_flip_left_right(image)

    return image


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
