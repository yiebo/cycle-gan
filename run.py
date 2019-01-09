import tensorflow as tf
from model import u_net, discriminator
from ops import *
import cv2
import numpy as np


x = tf.placeholder(tf.float16, shape=[None, None, None, 3])

with tf.variable_scope('generator', dtype=tf.float16,
                       custom_getter=float32_variable_storage_getter):
    fake_y = u_net(x, name="x2y")
    fake_x = u_net(x, name="y2x")


# output = tf.cond(y_real < x_real, lambda: fake_y, lambda: fake_x)
out_f = tf.minimum(tf.cast(fake_y, tf.float32), 1)
out_m = tf.minimum(tf.cast(fake_x, tf.float32), 1)
# cap = cv2.VideoCapture(0)
saver = tf.train.Saver()



with tf.Session() as sess:
    # saver.restore(sess, 'checkpoints')
    saver = tf.train.import_meta_graph('checkpoints/model.ckpt-100002.meta')
    saver.restore(sess, tf.train.latest_checkpoint('checkpoints'))

    # Our operations on the frame come here
    image = cv2.imread('marco.jpg')
    print(image.shape)
    idx = np.argmax(image.shape)
    # scale = (np.array([208., 176., 3.]) / image.shape)[0]
    # print(scale)
    # image = cv2.resize(image, None, fx=scale, fy=scale)*0.8
    image = cv2.resize(image, (176, 208))
    image = image / 255.0
    print(image)

    
    cv2.imshow("monkaS1", cv2.resize(image, None, fx=3, fy=3))

    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image[...,::-1]
    image = np.expand_dims(image, 0)
    print(image.shape)
    image_f, image_m = sess.run([out_f, out_m], feed_dict={x: image})
    # image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    image_f = image_f[0][...,::-1]
    image_m = image_m[0][...,::-1]
    cv2.imshow("monkaSf", cv2.resize(image_f, None, fx=3, fy=3))
    cv2.imshow("monkaSm", cv2.resize(image_m, None, fx=3, fy=3))

    cv2.waitKey(0)
#         i += 1

# with tf.Session() as sess:
#     # saver.restore(sess, 'checkpoints')
#     saver = tf.train.import_meta_graph('checkpoints/model.ckpt-99964.meta')
#     saver.restore(sess, tf.train.latest_checkpoint('checkpoints'))
#     print("hmm")
#     i = 0
#     while(True):
#         ret, frame = cap.read()
#         print(frame)
#         if i==4:
#             i = 0
#             # Our operations on the frame come here
#             print()
#             image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#             image_start = (image.shape/2) - np.array([104, 88])
#             image = image[52:52+104, 88:88+44]
#             # image = np.expand_dims(image, 0)
#             # image_ = sess.run(output, feed_dict={x: image})[0]

#             cv2.imshow("monkaS", image)
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break
#         i += 1

# # When everything done, release the capture
# cap.release()
# cv2.destroyAllWindows()
