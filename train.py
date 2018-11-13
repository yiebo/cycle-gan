import tensorflow as tf
from ops import *
import glob
import tqdm
from model import u_net, discriminator

from read import _parse_function

TRAINING = True


file_list_m = glob.glob("TFRECORD/celebA_male_*.tfrecord")
dataset_m = tf.data.TFRecordDataset(file_list_m)
dataset_m = dataset_m.map(_parse_function)


# dataset = dataset.shuffle(buffer_size=1000)
# dataset = dataset.repeat()
dataset_m.apply(tf.contrib.data.shuffle_and_repeat(buffer_size=3200))
dataset_m = dataset_m.batch(32)
dataset_m = dataset_m.prefetch(32)
iterator_m = dataset_m.make_one_shot_iterator()


file_list_f = glob.glob("TFRECORD/celebA_female_*.tfrecord")
dataset_f = tf.data.TFRecordDataset(file_list_f)
dataset_f = dataset_f.map(_parse_function)


# dataset = dataset.shuffle(buffer_size=1000)
# dataset = dataset.repeat()
dataset_f.apply(tf.contrib.data.shuffle_and_repeat(buffer_size=3200))
dataset_f = dataset_f.batch(32)
dataset_f = dataset_f.prefetch(32)
iterator_f = dataset_f.make_one_shot_iterator()

x, y = iterator_m.get_next(), iterator_f.get_next()

fake_y = u_net(x, name="x2y_generator")
fake_x = u_net(x, name="y2x_generator")
fake_y_ = u_net(fake_x, name="x2y_generator", reuse=True)
fake_x_ = u_net(fake_y, name="y2x_generator", reuse=True)
l2_x = tf.reduce_mean((x - fake_x_)**2)
l2_y = tf.reduce_mean((y - fake_y_)**2)

discriminator(y, name='discriminator_y')
discriminator(x, name='discriminator_x')
discriminator(fake_y, name='discriminator_y', reuse=True)
discriminator(fake_x, name='discriminator_x', reuse=True)

with tf.variable_scope("discriminator") as D:
    #add gaussian noise
    # noisy_x = self.x + tf.random_uniform(tf.shape(self.x), -0.005, 0.005)

    D_real = self.__discriminator(self.x)
    D.reuse_variables()
    D_false = self.__discriminator(self.g_out)

    #gradient penalty for wasserstein gp
    epsilon = tf.random_uniform([tf.shape(self.x)[0],1,1], 0.0, 1.0)
    x_hat = self.x + epsilon*(self.g_out-self.x)

    D_false_w = self.__discriminator(x_hat)
    gradients = tf.gradients(D_false_w, [x_hat])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=1))
    self.gradient_penalty = 10*tf.reduce_mean(tf.square(slopes-1.0))

loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=y))


tf.train.create_global_step()
global_step = tf.train.get_global_step()

learning_rate_ = tf.train.exponential_decay(
    0.0001, global_step, decay_steps=10, decay_rate=0.95)

training_op = tf.train.AdamOptimizer(
    learning_rate_).minimize(loss, global_step=global_step)


tf.summary.scalar('loss', loss)
tf.summary.scalar('learning_rate', learning_rate_)

# save iterator states
saveable_m = tf.contrib.data.make_saveable_from_iterator(iterator_m)
saveable_f = tf.contrib.data.make_saveable_from_iterator(iterator_f)
tf.add_to_collection(tf.GraphKeys.SAVEABLE_OBJECTS, [saveable_m, saveable_f])


config = tf.ConfigProto(intra_op_parallelism_threads=6)


with tf.train.MonitoredTrainingSession(checkpoint_dir='log/4', save_summaries_steps=2, config=config) as sess:
    for i in tqdm.trange(200):
        sess.run(training_op)