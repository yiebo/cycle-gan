import tensorflow as tf
import glob
import tqdm
import os

from model import u_net, discriminator
from ops import *
from read import iterator


LAMBDA1 = 10
LAMBDA2 = 10
BATCH_SIZE = 4


file_list_m = glob.glob("TFRECORD/male_*.tfrecord")
file_list_f = glob.glob("TFRECORD/female_*.tfrecord")
iterator_m = iterator(file_list_m, BATCH_SIZE)
iterator_f = iterator(file_list_f, BATCH_SIZE)

x, y = iterator_f.get_next(), iterator_m.get_next()
x16 = tf.cast(x, tf.float16)
y16 = tf.cast(y, tf.float16)

with tf.variable_scope('generator', dtype=tf.float16,
                       custom_getter=float32_variable_storage_getter):
    fake_y = u_net(x16, name="x2y", training=True)
    fake_x = u_net(y16, name="y2x", training=True)
    fake_y_ = u_net(fake_x, name="x2y", training=True)
    fake_x_ = u_net(fake_y, name="y2x", training=True)

with tf.variable_scope('discriminator', dtype=tf.float16,
                       custom_getter=float32_variable_storage_getter):
    d_y_real = discriminator(y16, name='y', training=True)
    d_x_real = discriminator(x16, name='x', training=True)
    d_y_fake = discriminator(fake_y, name='y', training=True)
    d_x_fake = discriminator(fake_x, name='x', training=True)


tf.train.create_global_step()
global_step = tf.train.get_global_step()
learning_rate_ = tf.train.exponential_decay(0.0005, global_step,
                                            decay_steps=2000, decay_rate=0.9)

G_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
D_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')

with tf.name_scope('D_optimizer'):
    D_optimizer = tf.train.AdamOptimizer(learning_rate_, beta1=0.5, name='D_solver')
    loss_scale_manager_D = tf.contrib.mixed_precision.ExponentialUpdateLossScaleManager(1000, 1000)
    loss_scale_optimizer_D = tf.contrib.mixed_precision.LossScaleOptimizer(D_optimizer,
                                                                           loss_scale_manager_D)

with tf.name_scope('G_optimizer'):
    G_optimizer = tf.train.AdamOptimizer(learning_rate_, beta1=0.5, name='G_solver')
    loss_scale_manager_G = tf.contrib.mixed_precision.ExponentialUpdateLossScaleManager(1000, 1000)
    loss_scale_optimizer_G = tf.contrib.mixed_precision.LossScaleOptimizer(G_optimizer,
                                                                           loss_scale_manager_G)

# -------------GAN SIMPLE GP R1
with tf.name_scope('losses'):

    d_y_real = tf.reduce_mean(d_y_real)
    d_x_real = tf.reduce_mean(d_x_real)
    d_y_fake = tf.reduce_mean(d_y_fake)
    d_x_fake = tf.reduce_mean(d_x_fake)

    d_y_real = tf.cast(d_y_real, tf.float32)
    d_x_real = tf.cast(d_x_real, tf.float32)
    d_y_fake = tf.cast(d_y_fake, tf.float32)
    d_x_fake = tf.cast(d_x_fake, tf.float32)

    loss_cycle_x = tf.reduce_mean(tf.abs(x - tf.cast(fake_x_, tf.float32)))
    loss_cycle_y = tf.reduce_mean(tf.abs(y - tf.cast(fake_y_, tf.float32)))

    loss_cycle = LAMBDA1 * loss_cycle_x + LAMBDA2 * loss_cycle_y

    loss_g_y = tf.nn.softplus(-d_y_fake)
    loss_g_x = tf.nn.softplus(-d_x_fake)
    loss_g = loss_g_x + loss_g_y + loss_cycle

    gp_x = gradient_penalty_simple(d_x_real, x)
    gp_y = gradient_penalty_simple(d_y_real, y)

    loss_d_x = tf.nn.softplus(d_x_fake) + tf.nn.softplus(-d_x_real) + gp_x
    loss_d_y = tf.nn.softplus(d_y_fake) + tf.nn.softplus(-d_y_real) + gp_y

    loss_d = loss_d_x + loss_d_y

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    D_solver = loss_scale_optimizer_D.minimize(loss_d, var_list=D_var, global_step=global_step)
    G_solver = loss_scale_optimizer_G.minimize(loss_g, var_list=G_var)

    training_op = tf.group(D_solver, G_solver)

# losses
tf.summary.scalar('loss/d/total', loss_d)
tf.summary.scalar('loss/g/total', loss_g)
tf.summary.scalar('loss/g/x', loss_g_x)
tf.summary.scalar('loss/g/y', loss_g_y)
tf.summary.scalar('loss/d/x', loss_d_x)
tf.summary.scalar('loss/d/y', loss_d_y)
tf.summary.scalar('cycle_loss/x', loss_cycle_x)
tf.summary.scalar('cycle_loss/y', loss_cycle_y)
tf.summary.scalar('gp/x', gp_x)
tf.summary.scalar('gp/y', gp_y)

# images
tf.summary.image('x/x', x16, 1)
tf.summary.image('x/x2y', tf.minimum(fake_y, 1), 1)
tf.summary.image('x/x2y2x', tf.minimum(fake_x_, 1), 1)
tf.summary.image('y/y', y16, 1)
tf.summary.image('y/y2x', tf.minimum(fake_x, 1), 1)
tf.summary.image('y/y2x2y', tf.minimum(fake_y_, 1), 1)

# params
tf.summary.scalar('loss_scale/g', loss_scale_manager_G.get_loss_scale())
tf.summary.scalar('loss_scale/d', loss_scale_manager_D.get_loss_scale())
tf.summary.scalar('learning_rate', learning_rate_)

if not os.path.exists('logs'):
    os.makedirs('logs')
if not os.path.exists('checkpoints'):
    os.makedirs('checkpoints')

config = tf.ConfigProto()
config.gpu_options.allow_growth = True


with tf.train.MonitoredTrainingSession(checkpoint_dir='checkpoints', summary_dir='logs',
                                       save_checkpoint_steps=2000,
                                       save_summaries_steps=200, config=config) as sess:
    with tqdm.tqdm(total=200000, dynamic_ncols=True) as pbar:
        while True:
            _, step = sess.run([training_op, global_step])
            if step - pbar.n > 0:
                pbar.update(step - pbar.n)
                if step >= pbar.total:
                    break
