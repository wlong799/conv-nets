# coding=utf-8
"""Program for running analysis on CIFAR-10 dataset"""
import os
import shutil
import struct
import tarfile

import numpy as np
import tensorflow as tf

import cnn


def create_cifar10_datasets(dest_dir, force=False):
    """Creates a TFRecords file for both the CIFAR-10 train and test data.

    Args:
        dest_dir: Directory to store TFRecords files.
        force: bool. If True, recreates the TFRecords files, even if they
               already exist.
    """
    data_url = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'
    cifar_data_dir = 'cifar-10-batches-bin'
    train_data_names = [os.path.join(dest_dir, cifar_data_dir, filename) for
                        filename in
                        ['data_batch_{}.bin'.format(i) for i in range(1, 6)]]
    test_data_names = [os.path.join(dest_dir, cifar_data_dir,
                                    'test_batch.bin')]
    train_record_name = os.path.join(dest_dir, 'cifar10_train.tfrecords')
    test_record_name = os.path.join(dest_dir, 'cifar10_test.tfrecords')

    # Download file if necessary
    tar_filename = os.path.join(dest_dir, data_url.split('/')[-1])
    if not os.path.exists(tar_filename):
        cnn.dataset_utils.download_dataset([data_url], dest_dir)
    with tarfile.open(tar_filename, 'r:gz') as tar_file:
        tar_file.extractall(dest_dir)

    # Create TFRecords files
    if force:
        os.remove(train_record_name)
        os.remove(test_record_name)
    if not os.path.exists(train_record_name):
        create_cifar10_record_from_binaries(train_record_name,
                                            train_data_names)
    if not os.path.exists(test_record_name):
        create_cifar10_record_from_binaries(test_record_name, test_data_names)

    # Clean up binaries
    shutil.rmtree(os.path.join(dest_dir, cifar_data_dir))


def create_cifar10_record_from_binaries(record_filename, data_filenames):
    """Parses CIFAR-10 binaries and writes examples to TFRecords file.

    The CIFAR-10 binary format is described here:
    http://www.cs.toronto.edu/~kriz/cifar.html

    Args:
        record_filename: Name of TFRecords file to write to.
        data_filenames: List of CIFAR-10 binaries to read from.
    """
    label_bytes = 1
    image_width, image_height, image_channels = 32, 32, 3
    image_bytes = image_width * image_height * image_channels
    coder = cnn.dataset_utils.ImageCoder()
    example_format = '{}B{}s'.format(label_bytes, image_bytes)

    record_writer = tf.python_io.TFRecordWriter(record_filename)
    for filename in data_filenames:
        with open(filename, 'rb') as binary_reader:
            buffer = binary_reader.read()
        for label, image_string in struct.iter_unpack(example_format, buffer):
            # Reformat byte data into image and encode as PNG string
            image_1d = np.fromstring(image_string, np.uint8)
            image_3d = image_1d.reshape(
                (image_channels, image_height, image_width))
            image_3d = image_3d.transpose([1, 2, 0])
            encoded_image = coder.encode_png(image_3d)

            example = tf.train.Example(features=tf.train.Features(feature={
                'height': cnn.dataset_utils.int64_feature(image_height),
                'width': cnn.dataset_utils.int64_feature(image_width),
                'channels': cnn.dataset_utils.int64_feature(image_channels),
                'label': cnn.dataset_utils.int64_feature(label),
                'encoded_image': cnn.dataset_utils.bytes_feature(
                    tf.compat.as_bytes(encoded_image))
            }))
            record_writer.write(example.SerializeToString())
    record_writer.close()


if __name__ == '__main__':
    # cnn.cnn_app.run('cifar10.ini', phase='test', examples_per_epoch=10000)
    cnn.cnn_app.run('cifar10.ini', phase='train', examples_per_epoch='50000')
