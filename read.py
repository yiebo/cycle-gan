import tensorflow as tf


def _parse_function(example_proto):
    features = {'image': tf.FixedLenFeature((), tf.string, default_value="")
                }
    parsed = tf.parse_single_example(example_proto, features)

    # image = tf.image.decode_image(parsed['image'])
    image_raw = tf.decode_raw(parsed['image'], tf.uint8)
    image = tf.cast(image_raw, tf.float32)
    image = tf.reshape(image, [208, 176, 3]) / 255.0

    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=32. / 255.)
    image = tf.image.random_saturation(image, lower=0.5, upper=1.5)

    return image
