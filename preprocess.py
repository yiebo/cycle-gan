#!/usr/bin/env python
import tensorflow as tf
from tqdm import tqdm
import csv
import os


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


def tfrecord_example(path):
    features = {'path': bytes_feature(path.encode())
                }

    example = tf.train.Example(features=tf.train.Features(feature=features))

    return example


if __name__ == '__main__':
    TFRECORD_SIZE = 10000
    tfrecord_idx = 0
    idx = 0
    males = []
    females = []
    path = '../DATASETS/celebA'
    image_path = f'{path}/img_align_celeba'
    write_path = 'TFRECORD'

    if not os.path.exists(write_path):
        os.makedirs(write_path)

    with open(f'{path}/list_attr_celeba.txt') as csvfile:
        reader = list(csv.reader(csvfile, delimiter=" "))

        for row in reader[1:]:
            if int(row[21]):
                males.append(row[0])
            else:
                females.append(row[0])

    tfrecord_writer = tf.io.TFRecordWriter(
        f'{write_path}/male_{tfrecord_idx:03d}.tfrecord')

    for male in tqdm(males):
        if idx >= TFRECORD_SIZE:
            idx = 0
            tfrecord_idx += 1
            tfrecord_writer.close()
            tfrecord_writer = tf.io.TFRecordWriter(
                f'{write_path}/male_{tfrecord_idx:03d}.tfrecord')
        example = tfrecord_example(f'{image_path}/{male}')
        tfrecord_writer.write(example.SerializeToString())
        idx += 1
    tfrecord_writer.close()

    tfrecord_idx = 0
    idx = 0

    tfrecord_writer = tf.io.TFRecordWriter(
        f'{write_path}/female_{tfrecord_idx:03d}.tfrecord')

    for female in tqdm(females):
        if idx >= TFRECORD_SIZE:
            idx = 0
            tfrecord_idx += 1
            tfrecord_writer.close()
            tfrecord_writer = tf.io.TFRecordWriter(
                f'{write_path}/female_{tfrecord_idx:03d}.tfrecord')
        example = tfrecord_example(f'{image_path}/{female}')
        tfrecord_writer.write(example.SerializeToString())
        idx += 1
    tfrecord_writer.close()
