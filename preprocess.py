#!/usr/bin/env python
import tensorflow as tf
from tqdm import tqdm
import csv
import cv2


def int64_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def float_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def bytes_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def tfrecord_example(image):
    features = {'image': bytes_feature(tf.compat.as_bytes(image.tostring()))
                }

    example = tf.train.Example(features=tf.train.Features(feature=features))

    return example


if __name__ == '__main__':
    TFRECORD_SIZE = 1000
    tfrecord_idx = 0
    idx = 0
    males = []
    females = []

    with open('../DATASETS/celebA/list_attr_celeba.csv') as csvfile:
        spamreader = list(csv.reader(csvfile))
        for row in spamreader[1:]:
            if row[21] == '1':
                males.append(row[0])
            else:
                females.append(row[0])

    tfrecord_writer = tf.python_io.TFRecordWriter(
        'TFRECORD/celebA_male_{:03d}.tfrecord'.format(tfrecord_idx))

    for male in tqdm(males):
        if idx >= TFRECORD_SIZE:
            idx = 0
            tfrecord_idx += 1
            tfrecord_writer.close()
            tfrecord_writer = tf.python_io.TFRecordWriter(
                'TFRECORD/celebA_male_{:03d}.tfrecord'.format(tfrecord_idx))

        image = cv2.imread(f'../DATASETS/celebA/img_align_celeba/{male}')
        example = tfrecord_example(image)
        tfrecord_writer.write(example.SerializeToString())
        idx += 1
    tfrecord_writer.close()

    tfrecord_writer = tf.python_io.TFRecordWriter(
        'TFRECORD/celebA_female_{:03d}.tfrecord'.format(tfrecord_idx))

    for female in tqdm(females):
        if idx >= TFRECORD_SIZE:
            idx = 0
            tfrecord_idx += 1
            tfrecord_writer.close()
            tfrecord_writer = tf.python_io.TFRecordWriter(
                'TFRECORD/celebA_female_{:03d}.tfrecord'.format(tfrecord_idx))
        image = cv2.imread(f'../DATASETS/celebA/img_align_celeba/{female}')
        example = tfrecord_example(image)
        tfrecord_writer.write(example.SerializeToString())
        idx += 1
    tfrecord_writer.close()
