import tensorflow as tf
from ops import *
import glob
import tqdm
from model import u_net, discriminator
import os

from read import _parse_function

# tf.enable_eager_execution()


TRAINING = True
LAMBDA1 = 10
LAMBDA2 = 10
BATCH_SIZE = 2


file_list_m = glob.glob("TFRECORD/celebA_male_*.tfrecord")
dataset_m = tf.data.TFRecordDataset(file_list_m)
dataset_m = dataset_m.map(_parse_function)

dataset_m.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=1000))
dataset_m = dataset_m.batch(BATCH_SIZE)
dataset_m = dataset_m.prefetch(1)
dataset_m = dataset_m.repeat()
iterator_m = dataset_m.make_one_shot_iterator()


file_list_f = glob.glob("TFRECORD/celebA_female_*.tfrecord")
dataset_f = tf.data.TFRecordDataset(file_list_f)
dataset_f = dataset_f.map(_parse_function)

dataset_f.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=1000))
dataset_f = dataset_f.batch(BATCH_SIZE)
dataset_f = dataset_f.prefetch(1)
dataset_f = dataset_f.repeat()
iterator_f = dataset_f.make_one_shot_iterator()

x, y = iterator_m.get_next(), iterator_f.get_next()
x16 = tf.cast(x, tf.float16)
y16 = tf.cast(y, tf.float16)


with tf.variable_scope('generator', dtype=tf.float16,
                       custom_getter=float32_variable_storage_getter):
    fake_y = u_net(x16, name="x2y")
    fake_x = u_net(y16, name="y2x")
    fake_y_ = u_net(fake_x, name="x2y")
    fake_x_ = u_net(fake_y, name="y2x")

loss_cycle_x = tf.reduce_mean(tf.abs(x - tf.cast(fake_x_, tf.float32)))
loss_cycle_y = tf.reduce_mean(tf.abs(y - tf.cast(fake_y_, tf.float32)))

loss_cycle = LAMBDA1 * loss_cycle_x + LAMBDA2 * loss_cycle_y

with tf.variable_scope('discriminator', dtype=tf.float16,
                       custom_getter=float32_variable_storage_getter):
    d_y_real = discriminator(y16, name='y')
    d_x_real = discriminator(x16, name='x')
    d_y_fake = discriminator(fake_y, name='y')
    d_x_fake = discriminator(fake_x, name='x')

d_y_real = tf.cast(d_y_real, tf.float32)
d_x_real = tf.cast(d_x_real, tf.float32)
d_y_fake = tf.cast(d_y_fake, tf.float32)
d_x_fake = tf.cast(d_x_fake, tf.float32)

loss_g_y = tf.reduce_mean(tf.squared_difference(d_y_fake, 0.9))
loss_g_x = tf.reduce_mean(tf.squared_difference(d_x_fake, 0.9))

loss_g = loss_g_x + loss_g_y + loss_cycle

loss_d_y_fake = tf.reduce_mean(tf.squared_difference(d_y_fake, 0.1))
loss_d_x_fake = tf.reduce_mean(tf.squared_difference(d_x_fake, 0.1))
loss_d_y_real = tf.reduce_mean(tf.squared_difference(d_y_real, 0.9))
loss_d_x_real = tf.reduce_mean(tf.squared_difference(d_x_real, 0.9))


loss_d = (loss_d_y_fake + loss_d_x_fake + loss_d_y_real + loss_d_x_real) / 2

tf.train.create_global_step()
global_step = tf.train.get_global_step()

learning_rate_ = tf.train.exponential_decay(
    0.00005, global_step, decay_steps=5000, decay_rate=0.95)

G_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
D_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')


D_optimizer = tf.train.AdamOptimizer(learning_rate_, beta1=0.5, name='D_solver')
G_optimizer = tf.train.AdamOptimizer(learning_rate_, beta1=0.5, name='G_solver')

loss_scale_manager_D = tf.contrib.mixed_precision.FixedLossScaleManager(5000)
loss_scale_optimizer_D = tf.contrib.mixed_precision.LossScaleOptimizer(D_optimizer,
                                                                       loss_scale_manager_D)

loss_scale_manager_G = tf.contrib.mixed_precision.FixedLossScaleManager(5000)
loss_scale_optimizer_G = tf.contrib.mixed_precision.LossScaleOptimizer(G_optimizer,
                                                                       loss_scale_manager_G)


D_solver = loss_scale_optimizer_D.minimize(loss_d, var_list=D_var, global_step=global_step)
G_solver = loss_scale_optimizer_G.minimize(loss_g, var_list=G_var)

training_op = tf.group(D_solver, G_solver)

tf.summary.scalar('loss/d/total', loss_d)
tf.summary.scalar('loss/g/total', loss_g)
tf.summary.scalar('loss/g/x', loss_g_x)
tf.summary.scalar('loss/g/y', loss_g_y)
tf.summary.scalar('loss/cycle_loss_x', loss_cycle_x)
tf.summary.scalar('loss/cycle_loss_y', loss_cycle_y)
tf.summary.scalar('learning_rate', learning_rate_)

tf.summary.image('x/x', x16, 1)
tf.summary.image('x/xy', tf.minimum(fake_y, 1), 1)
tf.summary.image('x/xyx', tf.minimum(fake_x_, 1), 1)
tf.summary.image('y/y', y16, 1)
tf.summary.image('y/yx', tf.minimum(fake_x, 1), 1)
tf.summary.image('y/yxy', tf.minimum(fake_y_, 1), 1)

# save iterator states
# saveable_m = tf.data.experimental.make_saveable_from_iterator(iterator_m)
# saveable_f = tf.data.experimental.make_saveable_from_iterator(iterator_f)
# tf.add_to_collection(tf.GraphKeys.SAVEABLE_OBJECTS, saveable_m)
# tf.add_to_collection(tf.GraphKeys.SAVEABLE_OBJECTS, saveable_f)

if not os.path.exists('logs'):
    os.makedirs('logs')
if not os.path.exists('checkpoints'):
    os.makedirs('checkpoints')

with tf.train.MonitoredTrainingSession(checkpoint_dir='checkpoints', summary_dir='logs',
                                       save_checkpoint_steps=1000, save_summaries_steps=100) as sess:
    step = 0
    total_steps = 100000
    with tqdm.tqdm(total=total_steps) as pbar:
        while(step <= total_steps):
            step_, _ = sess.run([global_step, training_op])
            pbar.update(step_ - step)
            step = step_
