import tensorflow as tf
from model import u_net
import cv2
from ops import *
import os

x = tf.placeholder(tf.float16, shape=[None, None, None, 3])

with tf.variable_scope('generator', dtype=tf.float16,
                       custom_getter=float32_variable_storage_getter):
    fake_x = u_net(x, name="x2y")
    fake_y = u_net(x, name="y2x")


# output = tf.cond(y_real < x_real, lambda: fake_y, lambda: fake_x)
out_f = tf.minimum(tf.cast(fake_y, tf.float32), 1)
out_m = tf.minimum(tf.cast(fake_x, tf.float32), 1)
# cap = cv2.VideoCapture(0)
saver = tf.train.Saver()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

if not os.path.exists('input'):
    os.makedirs('input')
if not os.path.exists('output'):
    os.makedirs('output')

files = os.listdir('input')
images = []

for file in files:
    image = cv2.imread(f'input/{file}')
    image = cv2.resize(image, (208, 208))
    image = image / 255.0

    image = image[..., ::-1]
    images.append(image)

with tf.Session(config=config) as sess:
    saver = tf.train.import_meta_graph('checkpoints/model.ckpt-34000.meta')
    saver.restore(sess, tf.train.latest_checkpoint('checkpoints'))

    # Our operations on the frame come here
    for idx in range(0, len(images), 5):
        images_f, images_m = sess.run([out_f, out_m], feed_dict={x: images[idx:idx + 5]})
        images_f = images_f[..., ::-1] * 255
        images_m = images_m[..., ::-1] * 255
        for image_m, image_f, file in zip(images_m, images_f, files[idx:idx + 5]):
            name = os.path.splitext(file)[0]
            cv2.imwrite(f'output/{name}_female.jpg', image_f)
            cv2.imwrite(f'output/{name}_male.jpg', image_m)
